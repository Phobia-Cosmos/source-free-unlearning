import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from linear_repro import LinearClassifier, evaluate_metrics, prepare_data, set_seed, train_classifier


def clone_linear_model(model: nn.Module, device: str) -> nn.Module:
    cloned = LinearClassifier(model.fc.in_features, model.fc.out_features).to(device)
    cloned.load_state_dict(copy.deepcopy(model.state_dict()))
    return cloned


def make_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def random_wrong_labels(labels: torch.Tensor, num_classes: int, generator: torch.Generator) -> torch.Tensor:
    labels_cpu = labels.detach().cpu()
    offsets = torch.randint(1, num_classes, labels_cpu.shape, generator=generator)
    return (labels_cpu + offsets) % num_classes


def l2_project(delta: torch.Tensor, eps: float) -> torch.Tensor:
    norms = delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
    factors = torch.clamp(eps / norms, max=1.0)
    return delta * factors


def targeted_l2_pgd(model: nn.Module, inputs: torch.Tensor, target_labels: torch.Tensor, eps: float,
                    eps_iter: float, nb_iter: int, generator: torch.Generator) -> torch.Tensor:
    device = inputs.device
    base = inputs.detach()
    if eps > 0:
        delta = torch.randn(base.shape, generator=generator, device=device)
        delta_norm = delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
        radii = torch.rand((base.shape[0], 1), generator=generator, device=device) * eps
        delta = delta / delta_norm * radii
    else:
        delta = torch.zeros_like(base)
    adv = (base + delta).detach()
    criterion = nn.CrossEntropyLoss()
    for _ in range(nb_iter):
        adv.requires_grad_(True)
        loss = criterion(model(adv), target_labels)
        grad = torch.autograd.grad(loss, adv)[0]
        grad_norm = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
        adv = adv - eps_iter * grad / grad_norm
        delta = l2_project(adv - base, eps)
        adv = (base + delta).detach()
    return adv


def build_adversarial_dataset(model: nn.Module, forget_features: torch.Tensor, forget_labels: torch.Tensor,
                              num_classes: int, device: str, batch_size: int, num_adv_images: int,
                              pgd_eps: float, pgd_alpha: float, pgd_iter: int, seed: int) -> TensorDataset:
    model.eval()
    label_generator = torch.Generator().manual_seed(seed)
    noise_generator = torch.Generator(device=device).manual_seed(seed)
    adv_features = []
    adv_targets = []
    loader = make_loader(forget_features, forget_labels, batch_size, shuffle=False)
    for rep in range(num_adv_images):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            target_labels = random_wrong_labels(labels, num_classes, label_generator).to(device)
            adv_batch = targeted_l2_pgd(model, features, target_labels, pgd_eps, pgd_alpha, pgd_iter, noise_generator)
            adv_features.append(adv_batch.detach().cpu())
            adv_targets.append(target_labels.detach().cpu())
    return TensorDataset(torch.cat(adv_features, dim=0), torch.cat(adv_targets, dim=0))


def estimate_parameter_importance(model: nn.Module, forget_features: torch.Tensor, forget_labels: torch.Tensor,
                                  device: str, batch_size: int) -> dict[str, torch.Tensor]:
    loader = make_loader(forget_features, forget_labels, batch_size, shuffle=False)
    importance = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    count = 0
    model.train()
    for features, labels in loader:
        features = features.to(device)
        logits = model(features)
        loss = torch.norm(logits, p=2, dim=1).mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                importance[name] += param.grad.abs() * len(labels)
        count += len(labels)
    for name, imp in importance.items():
        imp = imp / max(1, count)
        imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-12)
        importance[name] = 1.0 - imp
    return importance


def run_l2ul_adversarial(source_model: nn.Module, forget_features: torch.Tensor, forget_labels: torch.Tensor,
                         device: str, num_classes: int, batch_size: int, unlearn_lr: float, pgd_eps: float,
                         pgd_alpha: float, pgd_iter: int, num_adv_images: int, max_outer_loops: int,
                         reg_lamb: float | None, seed: int) -> nn.Module:
    model = clone_linear_model(source_model, device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=unlearn_lr, momentum=0.9, weight_decay=1e-4)
    origin_params = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    importance = None
    if reg_lamb is not None:
        importance = estimate_parameter_importance(model, forget_features, forget_labels, device, batch_size)
    adv_dataset = build_adversarial_dataset(
        model,
        forget_features,
        forget_labels,
        num_classes,
        device,
        batch_size,
        num_adv_images,
        pgd_eps,
        pgd_alpha,
        pgd_iter,
        seed + 11,
    )
    adv_loader = DataLoader(adv_dataset, batch_size=batch_size, shuffle=True)
    forget_loader = make_loader(forget_features, forget_labels, batch_size, shuffle=True)
    forget_eval_loader = make_loader(forget_features, forget_labels, 512, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    for _ in range(max_outer_loops):
        forget_iter = iter(forget_loader)
        for adv_features, adv_targets in adv_loader:
            try:
                forget_batch, forget_targets = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                forget_batch, forget_targets = next(forget_iter)
            adv_features = adv_features.to(device)
            adv_targets = adv_targets.to(device)
            forget_batch = forget_batch.to(device)
            forget_targets = forget_targets.to(device)
            optimizer.zero_grad()
            output_adv = model(adv_features)
            output_forget = model(forget_batch)
            total = adv_features.shape[0] + forget_batch.shape[0]
            loss_unlearn = -criterion(output_forget, forget_targets) * (forget_batch.shape[0] / total)
            loss_adv = criterion(output_adv, adv_targets) * (adv_features.shape[0] / total)
            loss = loss_unlearn + loss_adv
            if importance is not None:
                reg_loss = torch.tensor(0.0, device=device)
                for name, param in model.named_parameters():
                    if name in importance:
                        reg_loss = reg_loss + torch.sum(importance[name] * (param - origin_params[name]).pow(2)) / 2.0
                loss = loss + reg_lamb * reg_loss
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in forget_eval_loader:
                logits = model(features.to(device))
                pred = logits.argmax(dim=1).cpu()
                correct += (pred == labels).sum().item()
                total += len(labels)
        if correct == 0:
            break
        model.train()
    model.eval()
    return model


def run_jit_official(source_model: nn.Module, forget_features: torch.Tensor, forget_labels: torch.Tensor,
                     device: str, batch_size: int, learning_rate: float, lipschitz_weighting: float,
                     n_samples: int, passes: int, seed: int) -> nn.Module:
    model = clone_linear_model(source_model, device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loader = make_loader(forget_features, forget_labels, batch_size, shuffle=False)
    label_generator = torch.Generator().manual_seed(seed)
    noise_generator = torch.Generator(device=device).manual_seed(seed)
    model.train()
    for _ in range(passes):
        for features, _ in loader:
            features = features.to(device)
            outputs = model(features)
            loss = torch.tensor(0.0, device=device)
            for _ in range(n_samples):
                noise = torch.randn(features.shape, generator=noise_generator, device=device) * lipschitz_weighting
                noisy_features = features + noise
                with torch.no_grad():
                    noisy_outputs = model(noisy_features)
                input_norm = (features - noisy_features).norm(dim=1).clamp_min(1e-12)
                output_norm = (outputs - noisy_outputs).norm(dim=1)
                loss = loss + (output_norm / input_norm).abs().sum()
            loss = loss / n_samples
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
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
        args.source_batch_size,
        args.source_learning_rate,
        args.seed,
    )
    retrained_model = train_classifier(
        data.remaining_features,
        data.remaining_labels,
        num_classes,
        device,
        args.retrain_epochs,
        args.source_batch_size,
        args.source_learning_rate,
        args.seed + 1000,
    )
    metrics = {
        'source_model': evaluate_metrics(source_model, data, device, num_classes, args.seed),
        'retrained': evaluate_metrics(retrained_model, data, device, num_classes, args.seed),
    }
    if 'jit_official' in args.methods:
        jit_model = run_jit_official(
            source_model,
            data.forget_features,
            data.forget_labels,
            device,
            args.method_batch_size,
            args.jit_learning_rate,
            args.jit_lipschitz_weighting,
            args.jit_num_samples,
            args.jit_passes,
            args.seed + 2000,
        )
        metrics['jit_official'] = evaluate_metrics(jit_model, data, device, num_classes, args.seed)
    if 'adversarial_official' in args.methods:
        adv_model = run_l2ul_adversarial(
            source_model,
            data.forget_features,
            data.forget_labels,
            device,
            num_classes,
            args.method_batch_size,
            args.adv_unlearn_lr,
            args.adv_pgd_eps,
            args.adv_pgd_alpha,
            args.adv_pgd_iter,
            args.adv_num_adv_images,
            args.adv_max_outer_loops,
            None,
            args.seed + 3000,
        )
        metrics['adversarial_official'] = evaluate_metrics(adv_model, data, device, num_classes, args.seed)
    if 'l2ul_official' in args.methods:
        l2ul_model = run_l2ul_adversarial(
            source_model,
            data.forget_features,
            data.forget_labels,
            device,
            num_classes,
            args.method_batch_size,
            args.adv_unlearn_lr,
            args.adv_pgd_eps,
            args.adv_pgd_alpha,
            args.adv_pgd_iter,
            args.adv_num_adv_images,
            args.adv_max_outer_loops,
            args.l2ul_reg_lamb,
            args.seed + 4000,
        )
        metrics['l2ul_official'] = evaluate_metrics(l2ul_model, data, device, num_classes, args.seed)
    result = {
        'config': vars(args),
        'note': (
            'JiT is ported from jwf40/Information-Theoretic-Unlearning (lipschitz.py + forget_full_class_strategies.py); '
            'Adversarial/L2UL is ported from csm9493/L2UL (main_unlearn_cifar10_mixed_label_resnet18.py + utils.py). '
            'Both are adapted to the fixed-feature linear-classifier benchmark used in this repository.'
        ),
        'metrics': metrics,
    }
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f'{args.dataset_id}_split{args.split_rate}_seed{args.seed}_official_baselines.json'
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
    parser.add_argument('--source-batch-size', type=int, default=32)
    parser.add_argument('--source-learning-rate', type=float, default=0.001)
    parser.add_argument('--method-batch-size', type=int, default=128)
    parser.add_argument('--bound-train', action='store_true')
    parser.add_argument('--bound-test', action='store_true')
    parser.add_argument('--cache-dir', type=str, default='artifacts/feature_cache')
    parser.add_argument('--save-dir', type=str, default='artifacts/baseline_official_repro')
    parser.add_argument('--methods', nargs='+', default=['jit_official', 'adversarial_official', 'l2ul_official'],
                        choices=['jit_official', 'adversarial_official', 'l2ul_official'])
    parser.add_argument('--jit-learning-rate', type=float, default=0.0003)
    parser.add_argument('--jit-lipschitz-weighting', type=float, default=0.01)
    parser.add_argument('--jit-num-samples', type=int, default=25)
    parser.add_argument('--jit-passes', type=int, default=1)
    parser.add_argument('--adv-pgd-eps', type=float, default=4.0)
    parser.add_argument('--adv-pgd-alpha', type=float, default=0.1)
    parser.add_argument('--adv-pgd-iter', type=int, default=100)
    parser.add_argument('--adv-num-adv-images', type=int, default=20)
    parser.add_argument('--adv-unlearn-lr', type=float, default=0.001)
    parser.add_argument('--adv-max-outer-loops', type=int, default=50)
    parser.add_argument('--l2ul-reg-lamb', type=float, default=1.0)
    return parser


def main():
    args = build_parser().parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
