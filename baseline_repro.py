import argparse
import copy
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim

from linear_repro import (
    LinearClassifier,
    evaluate_metrics,
    one_hot_encode,
    prepare_data,
    quadratic_loss,
    set_seed,
    train_classifier,
)


def batch_indices(length: int, batch_size: int, seed: int) -> Iterable[torch.Tensor]:
    order = torch.randperm(length, generator=torch.Generator().manual_seed(seed))
    for offset in range(0, length, batch_size):
        yield order[offset:offset + batch_size]


def clone_linear_model(model: nn.Module, device: str) -> nn.Module:
    cloned = LinearClassifier(model.fc.in_features, model.fc.out_features).to(device)
    cloned.load_state_dict(copy.deepcopy(model.state_dict()))
    return cloned


def wrong_random_labels(labels: torch.Tensor, num_classes: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    offsets = torch.randint(1, num_classes, labels.shape, generator=generator)
    return (labels.cpu() + offsets) % num_classes


def finetune_forget_only(model: nn.Module, features: torch.Tensor, labels: torch.Tensor, num_classes: int,
                         device: str, epochs: int, batch_size: int, learning_rate: float, seed: int,
                         objective: str, anchor_weight: float = 0.0) -> nn.Module:
    initial_weight = model.fc.weight.detach().clone()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_labels = labels
    if objective == 'random_labels':
        train_labels = wrong_random_labels(labels, num_classes, seed)

    for epoch in range(epochs):
        for indices in batch_indices(len(features), batch_size, seed + epoch):
            inputs = features[indices].to(device)
            targets = train_labels[indices].to(device)
            outputs = model(inputs)
            loss = quadratic_loss(outputs, one_hot_encode(targets, num_classes))
            if objective == 'neggrad':
                loss = -loss
            if anchor_weight > 0:
                loss = loss + anchor_weight * torch.mean((model.fc.weight - initial_weight) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def finetune_jit(model: nn.Module, features: torch.Tensor, labels: torch.Tensor, num_classes: int,
                 device: str, epochs: int, batch_size: int, learning_rate: float, seed: int,
                 perturb_std: float, lipschitz_weight: float, anchor_weight: float) -> nn.Module:
    initial_weight = model.fc.weight.detach().clone()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        generator = torch.Generator(device=device).manual_seed(seed + epoch)
        for indices in batch_indices(len(features), batch_size, seed + epoch):
            inputs = features[indices].to(device)
            targets = labels[indices].to(device)
            noise = torch.randn(inputs.shape, generator=generator, device=device) * perturb_std
            clean_outputs = model(inputs)
            noisy_outputs = model(inputs + noise)
            forget_loss = -quadratic_loss(clean_outputs, one_hot_encode(targets, num_classes))
            lipschitz_loss = torch.mean((clean_outputs - noisy_outputs) ** 2)
            anchor_loss = torch.mean((model.fc.weight - initial_weight) ** 2)
            loss = forget_loss + lipschitz_weight * lipschitz_loss + anchor_weight * anchor_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def weight_importance(model: nn.Module, features: torch.Tensor, labels: torch.Tensor, num_classes: int,
                      device: str, batch_size: int) -> torch.Tensor:
    importance = torch.zeros_like(model.fc.weight, device=device)
    count = 0
    for indices in batch_indices(len(features), batch_size, seed=2025):
        inputs = features[indices].to(device)
        targets = labels[indices].to(device)
        outputs = model(inputs)
        loss = quadratic_loss(outputs, one_hot_encode(targets, num_classes))
        model.zero_grad(set_to_none=True)
        loss.backward()
        importance += model.fc.weight.grad.detach() ** 2
        count += 1
    return importance / max(1, count)


def finetune_adversarial(model: nn.Module, features: torch.Tensor, labels: torch.Tensor, num_classes: int,
                         device: str, epochs: int, batch_size: int, learning_rate: float, seed: int,
                         epsilon: float, importance_weight: float) -> nn.Module:
    initial_weight = model.fc.weight.detach().clone()
    importance = weight_importance(model, features, labels, num_classes, device, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        for indices in batch_indices(len(features), batch_size, seed + epoch):
            inputs = features[indices].to(device).detach().requires_grad_(True)
            targets = labels[indices].to(device)
            outputs = model(inputs)
            clean_loss = quadratic_loss(outputs, one_hot_encode(targets, num_classes))
            feature_grad = torch.autograd.grad(clean_loss, inputs, retain_graph=False, create_graph=False)[0]
            adversarial_inputs = (inputs + epsilon * feature_grad.sign()).detach()
            adversarial_outputs = model(adversarial_inputs)
            ascent_loss = -quadratic_loss(adversarial_outputs, one_hot_encode(targets, num_classes))
            preserve_loss = torch.mean(importance * (model.fc.weight - initial_weight) ** 2)
            loss = ascent_loss + importance_weight * preserve_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def run_experiment(args):
    set_seed(args.seed)
    device = f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu'
    data, num_classes = prepare_data(
        args.dataset_id,
        args.split_rate,
        args.seed,
        args.bound_train,
        args.bound_test,
        device,
        args.cache_dir,
    )
    source_model = train_classifier(
        data.train_features,
        data.train_labels,
        num_classes,
        device,
        args.train_epochs,
        args.batch_size,
        args.learning_rate,
        args.seed,
    )
    retrained_model = train_classifier(
        data.remaining_features,
        data.remaining_labels,
        num_classes,
        device,
        args.retrain_epochs,
        args.batch_size,
        args.learning_rate,
        args.seed + 1000,
    )

    metrics = {
        'source_model': evaluate_metrics(source_model, data, device, num_classes, args.seed),
        'retrained': evaluate_metrics(retrained_model, data, device, num_classes, args.seed),
    }

    if 'neggrad' in args.methods:
        model = clone_linear_model(source_model, device)
        model = finetune_forget_only(
            model, data.forget_features, data.forget_labels, num_classes, device,
            args.neggrad_epochs, args.batch_size, args.neggrad_lr, args.seed + 2000,
            objective='neggrad', anchor_weight=args.anchor_weight,
        )
        metrics['neggrad'] = evaluate_metrics(model, data, device, num_classes, args.seed)

    if 'random_labels' in args.methods:
        model = clone_linear_model(source_model, device)
        model = finetune_forget_only(
            model, data.forget_features, data.forget_labels, num_classes, device,
            args.random_epochs, args.batch_size, args.random_lr, args.seed + 3000,
            objective='random_labels', anchor_weight=args.anchor_weight,
        )
        metrics['random_labels'] = evaluate_metrics(model, data, device, num_classes, args.seed)

    if 'jit' in args.methods:
        model = clone_linear_model(source_model, device)
        model = finetune_jit(
            model, data.forget_features, data.forget_labels, num_classes, device,
            args.jit_epochs, args.batch_size, args.jit_lr, args.seed + 4000,
            args.jit_perturb_std, args.jit_lipschitz_weight, args.jit_anchor_weight,
        )
        metrics['jit_approx'] = evaluate_metrics(model, data, device, num_classes, args.seed)

    if 'adversarial' in args.methods:
        model = clone_linear_model(source_model, device)
        model = finetune_adversarial(
            model, data.forget_features, data.forget_labels, num_classes, device,
            args.adv_epochs, args.batch_size, args.adv_lr, args.seed + 5000,
            args.adv_epsilon, args.adv_importance_weight,
        )
        metrics['adversarial_approx'] = evaluate_metrics(model, data, device, num_classes, args.seed)

    result = {
        'config': vars(args),
        'note': (
            'NegGrad and Random Labels are direct forget-data-only baselines from the paper description. '
            'JiT and Adversarial are local approximations because the source-free-unlearning repository '
            'does not include their original implementations or exact hyperparameters.'
        ),
        'metrics': metrics,
    }
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f'{args.dataset_id}_split{args.split_rate}_seed{args.seed}_baselines.json'
    out_file.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f'saved_result={out_file}')


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-id', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--split-rate', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--train-epochs', type=int, default=10)
    parser.add_argument('--retrain-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--bound-train', action='store_true')
    parser.add_argument('--bound-test', action='store_true')
    parser.add_argument('--cache-dir', type=str, default='artifacts/feature_cache')
    parser.add_argument('--save-dir', type=str, default='artifacts/baseline_repro')
    parser.add_argument('--methods', nargs='+', default=['neggrad', 'random_labels', 'jit', 'adversarial'],
                        choices=['neggrad', 'random_labels', 'jit', 'adversarial'])
    parser.add_argument('--anchor-weight', type=float, default=0.0)
    parser.add_argument('--neggrad-epochs', type=int, default=1)
    parser.add_argument('--neggrad-lr', type=float, default=0.001)
    parser.add_argument('--random-epochs', type=int, default=5)
    parser.add_argument('--random-lr', type=float, default=0.001)
    parser.add_argument('--jit-epochs', type=int, default=1)
    parser.add_argument('--jit-lr', type=float, default=0.001)
    parser.add_argument('--jit-perturb-std', type=float, default=0.05)
    parser.add_argument('--jit-lipschitz-weight', type=float, default=1.0)
    parser.add_argument('--jit-anchor-weight', type=float, default=0.0)
    parser.add_argument('--adv-epochs', type=int, default=1)
    parser.add_argument('--adv-lr', type=float, default=0.001)
    parser.add_argument('--adv-epsilon', type=float, default=0.02)
    parser.add_argument('--adv-importance-weight', type=float, default=1.0)
    return parser


def main():
    args = build_parser().parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
