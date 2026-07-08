import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class ExperimentData:
    train_features: torch.Tensor
    train_labels: torch.Tensor
    test_features: torch.Tensor
    test_labels: torch.Tensor
    remaining_features: torch.Tensor
    remaining_labels: torch.Tensor
    forget_features: torch.Tensor
    forget_labels: torch.Tensor
    forget_indices: torch.Tensor
    remaining_indices: torch.Tensor


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x.view(x.size(0), -1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_hot_encode(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.zeros(labels.size(0), num_classes, device=labels.device).scatter_(1, labels.unsqueeze(1), 1)


def quadratic_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean(0.5 * (outputs - targets) ** 2)


def bound_norm(data: torch.Tensor, upper_bound: float) -> torch.Tensor:
    norms = data.norm(dim=1, keepdim=True)
    scale = upper_bound / norms.clamp_min(1e-12)
    scale[scale > 1] = 1
    return data * scale


def get_cache_file(cache_dir: Optional[str], dataset_id: str) -> Optional[Path]:
    if cache_dir is None:
        return None
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / f'{dataset_id}_resnet18_imagenet_features.pth'


@torch.no_grad()
def collect_features(dataset_id: str, device: str, batch_size: int = 128, cache_dir: Optional[str] = None):
    cache_file = get_cache_file(cache_dir, dataset_id)
    if cache_file is not None and cache_file.exists():
        cached = torch.load(cache_file, map_location='cpu')
        return (
            cached['train_features'].float(),
            cached['train_labels'].long(),
            cached['test_features'].float(),
            cached['test_labels'].long(),
            int(cached['num_classes']),
        )

    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    if dataset_id == 'cifar10':
        trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_id == 'cifar100':
        trainset = CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
    else:
        raise ValueError(dataset_id)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    feature_extractor = FeatureExtractor().to(device).eval()

    def run(loader):
        features = []
        labels = []
        for inputs, targets in loader:
            feats = feature_extractor(inputs.to(device)).detach().cpu()
            features.append(feats)
            labels.append(targets)
        return torch.cat(features), torch.cat(labels)

    train_features, train_labels = run(trainloader)
    test_features, test_labels = run(testloader)
    if cache_file is not None:
        torch.save({
            'train_features': train_features,
            'train_labels': train_labels,
            'test_features': test_features,
            'test_labels': test_labels,
            'num_classes': num_classes,
        }, cache_file)
    return train_features.float(), train_labels.long(), test_features.float(), test_labels.long(), num_classes


def prepare_data(dataset_id: str, split_rate: float, seed: int, bound_train: bool, bound_test: bool,
                 device: str, cache_dir: Optional[str]):
    train_features, train_labels, test_features, test_labels, num_classes = collect_features(
        dataset_id,
        device,
        cache_dir=cache_dir,
    )
    if bound_train:
        train_features = bound_norm(train_features, 1.0)
    if bound_test:
        test_features = bound_norm(test_features, 1.0)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(train_features), generator=generator)
    forget_count = int(len(train_features) * split_rate)
    forget_indices = permutation[:forget_count]
    remaining_indices = permutation[forget_count:]

    return ExperimentData(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        remaining_features=train_features[remaining_indices],
        remaining_labels=train_labels[remaining_indices],
        forget_features=train_features[forget_indices],
        forget_labels=train_labels[forget_indices],
        forget_indices=forget_indices,
        remaining_indices=remaining_indices,
    ), num_classes


def train_classifier(features: torch.Tensor, labels: torch.Tensor, num_classes: int, device: str,
                     epochs: int, batch_size: int, learning_rate: float, seed: int):
    model = LinearClassifier(features.shape[1], num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        order = torch.randperm(len(features), generator=torch.Generator().manual_seed(seed + epoch))
        for offset in range(0, len(features), batch_size):
            batch_indices = order[offset:offset + batch_size]
            inputs = features[batch_indices].to(device)
            targets = labels[batch_indices].to(device)
            outputs = model(inputs)
            loss = quadratic_loss(outputs, one_hot_encode(targets, num_classes))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def accuracy(model: nn.Module, features: torch.Tensor, labels: torch.Tensor, device: str, batch_size: int = 256):
    correct = 0
    total = 0
    for offset in range(0, len(features), batch_size):
        inputs = features[offset:offset + batch_size].to(device)
        targets = labels[offset:offset + batch_size].to(device)
        predicted = model(inputs).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += len(inputs)
    return correct / total


@torch.no_grad()
def collect_losses(model: nn.Module, features: torch.Tensor, labels: torch.Tensor, device: str,
                   num_classes: int, batch_size: int = 256):
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses = []
    classes = []
    for offset in range(0, len(features), batch_size):
        inputs = features[offset:offset + batch_size].to(device)
        targets = labels[offset:offset + batch_size].to(device)
        logits = model(inputs)
        losses.extend(criterion(logits, one_hot_encode(targets, num_classes)).cpu().numpy().tolist())
        classes.extend(targets.cpu().numpy().tolist())
    return losses, classes


def attack_score(estimator, X, y):
    return float(np.mean(estimator.predict(X) == y))


def membership_inference_attack(model: nn.Module, test_features: torch.Tensor, test_labels: torch.Tensor,
                                forget_features: torch.Tensor, forget_labels: torch.Tensor,
                                device: str, num_classes: int, seed: int):
    test_losses, test_classes = collect_losses(model, test_features, test_labels, device, num_classes)
    forget_losses, forget_classes = collect_losses(model, forget_features, forget_labels, device, num_classes)
    forget_class_set = set(forget_classes)
    filtered_test_losses = [loss for loss, label in zip(test_losses, test_classes) if label in forget_class_set]
    random.seed(seed)
    if len(filtered_test_losses) > len(forget_losses):
        filtered_test_losses = random.sample(filtered_test_losses, len(forget_losses))
    elif len(forget_losses) > len(filtered_test_losses):
        forget_losses = random.sample(forget_losses, len(filtered_test_losses))
    features = np.array(filtered_test_losses + forget_losses, dtype=np.float32).reshape(-1, 1)
    labels = np.array([0] * len(filtered_test_losses) + [1] * len(forget_losses), dtype=np.int64)
    attack_model = LogisticRegression(max_iter=1000)
    scores = cross_val_score(
        attack_model,
        features,
        labels,
        cv=StratifiedShuffleSplit(n_splits=5, random_state=seed),
        scoring=attack_score,
    )
    return {
        'mia_scores': scores.tolist(),
        'mia_mean': float(np.mean(scores)),
        'num_test_samples_used': len(filtered_test_losses),
        'num_forget_samples_used': len(forget_losses),
    }


@torch.no_grad()
def exact_unlearned_plus(source_model: nn.Module, data: ExperimentData, device: str, num_classes: int,
                         lambda_reg: float):
    weight = source_model.fc.weight.detach().cpu()
    total_count = len(data.train_features)
    feature_dim = data.train_features.shape[1]
    identity = torch.eye(feature_dim)
    retain_hessian = (data.remaining_features.T @ data.remaining_features) / total_count + lambda_reg * identity
    y_forget = one_hot_encode(data.forget_labels, num_classes).float()
    grad_f = (data.forget_features.T @ (data.forget_features @ weight.T - y_forget)) / total_count
    update = torch.linalg.solve(retain_hessian, grad_f)
    model = LinearClassifier(feature_dim, num_classes).to(device)
    model.load_state_dict(source_model.state_dict())
    model.fc.weight.data = (weight.T + update).T.to(device)
    return model


@torch.no_grad()
def evaluate_metrics(model: nn.Module, data: ExperimentData, device: str, num_classes: int, seed: int):
    metrics = {
        'test_acc': accuracy(model, data.test_features, data.test_labels, device),
        'remaining_acc': accuracy(model, data.remaining_features, data.remaining_labels, device),
        'forget_acc': accuracy(model, data.forget_features, data.forget_labels, device),
    }
    metrics.update(membership_inference_attack(
        model,
        data.test_features,
        data.test_labels,
        data.forget_features,
        data.forget_labels,
        device,
        num_classes,
        seed,
    ))
    return metrics


def compute_forget_hessian_lower_bound(data: ExperimentData, lambda_reg: float):
    total_count = len(data.train_features)
    feature_dim = data.train_features.shape[1]
    identity = torch.eye(feature_dim)
    return (data.forget_features.T @ data.forget_features) / total_count + lambda_reg * identity


@torch.no_grad()
def compute_forget_gradient(source_model: nn.Module, data: ExperimentData, num_classes: int):
    weight = source_model.fc.weight.detach().cpu()
    y_forget = one_hot_encode(data.forget_labels, num_classes).float()
    return (data.forget_features.T @ (data.forget_features @ weight.T - y_forget)) / len(data.train_features)


def approximate_retain_hessian_via_sdp(source_model: nn.Module, data: ExperimentData, lambda_reg: float,
                                       num_perturbations: int, perturb_scale: float,
                                       solver: str, solver_max_iters: int):
    feature_dim = data.train_features.shape[1]
    num_classes = source_model.fc.weight.shape[0]
    base_weight = source_model.fc.weight.detach().cpu().numpy()
    forget_targets = one_hot_encode(data.forget_labels, num_classes).float()
    base_outputs = data.forget_features @ torch.tensor(base_weight.T, dtype=torch.float32)
    base_loss = quadratic_loss(base_outputs, forget_targets).item()

    delta_w = torch.randn(num_perturbations, feature_dim) * perturb_scale
    delta_l = []
    for perturbation in delta_w:
        noisy_weight = base_weight + perturbation.view(1, -1).numpy()
        outputs = data.forget_features @ torch.tensor(noisy_weight.T, dtype=torch.float32)
        delta_l.append(quadratic_loss(outputs, forget_targets).item() - base_loss)

    delta_w_matrix = delta_w.numpy()
    delta_l = np.array(delta_l)
    X = cp.Variable((feature_dim, feature_dim), symmetric=True)
    quadratic_forms = 0.5 * cp.sum(cp.multiply(delta_w_matrix @ X, delta_w_matrix), axis=1)
    objective = cp.Minimize(cp.sum_squares(delta_l - quadratic_forms))
    forget_hessian_lower = compute_forget_hessian_lower_bound(data, lambda_reg).numpy()
    constraints = [X >> 0, X >> forget_hessian_lower]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=getattr(cp, solver), verbose=False, max_iters=solver_max_iters)
    if X.value is None:
        raise RuntimeError(f'SDP failed with status={problem.status}')
    retain_hessian = X.value - forget_hessian_lower
    retain_hessian = 0.5 * (retain_hessian + retain_hessian.T)
    return torch.tensor(retain_hessian, dtype=torch.float32), {
        'sdp_status': problem.status,
        'sdp_objective': float(problem.value),
    }


@torch.no_grad()
def source_free_unlearned_minus(source_model: nn.Module, data: ExperimentData, device: str, num_classes: int,
                                lambda_reg: float, num_perturbations: int, perturb_scale: float,
                                solver: str, solver_max_iters: int):
    feature_dim = data.train_features.shape[1]
    weight = source_model.fc.weight.detach().cpu()
    retain_hessian, sdp_info = approximate_retain_hessian_via_sdp(
        source_model,
        data,
        lambda_reg,
        num_perturbations,
        perturb_scale,
        solver,
        solver_max_iters,
    )
    gradient = compute_forget_gradient(source_model, data, num_classes)
    update = torch.linalg.solve(retain_hessian + 1e-5 * torch.eye(feature_dim), gradient)
    model = LinearClassifier(feature_dim, num_classes).to(device)
    model.load_state_dict(source_model.state_dict())
    model.fc.weight.data = (weight.T + update).T.to(device)
    return model, sdp_info


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
    unlearned_plus = exact_unlearned_plus(source_model, data, device, num_classes, args.lambda_reg)
    unlearned_minus, sdp_info = source_free_unlearned_minus(
        source_model,
        data,
        device,
        num_classes,
        args.lambda_reg,
        args.num_perturbations,
        args.perturb_scale,
        args.solver,
        args.solver_max_iters,
    )
    result = {
        'config': vars(args),
        'metrics': {
            'source_model': evaluate_metrics(source_model, data, device, num_classes, args.seed),
            'retrained': evaluate_metrics(retrained_model, data, device, num_classes, args.seed),
            'unlearned_plus': evaluate_metrics(unlearned_plus, data, device, num_classes, args.seed),
            'unlearned_minus': evaluate_metrics(unlearned_minus, data, device, num_classes, args.seed),
        },
        'sdp': sdp_info,
    }
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset_id}_split{args.split_rate}_pert{args.num_perturbations}_lam{args.lambda_reg}_seed{args.seed}"
    out_file = save_dir / f'{stem}.json'
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
    parser.add_argument('--lambda-reg', type=float, default=0.0)
    parser.add_argument('--num-perturbations', type=int, default=500)
    parser.add_argument('--perturb-scale', type=float, default=0.01)
    parser.add_argument('--solver', type=str, default='SCS')
    parser.add_argument('--solver-max-iters', type=int, default=4000)
    parser.add_argument('--bound-train', action='store_true')
    parser.add_argument('--bound-test', action='store_true')
    parser.add_argument('--cache-dir', type=str, default='artifacts/feature_cache')
    parser.add_argument('--save-dir', type=str, default='artifacts/linear_repro')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
