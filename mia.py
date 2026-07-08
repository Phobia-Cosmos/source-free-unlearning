import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

from dataset import split_user_train_dataset_to_remaining_forget, get_remaining_forget_loader, get_user_loader
from evaluate import infer_core_model_path
from model import get_core_model_params, get_trained_linear, freeze


def one_hot_from_target(target, num_classes):
    if target.ndim > 1:
        return target.float()
    return torch.nn.functional.one_hot(target.long(), num_classes=num_classes).float()


@torch.no_grad()
def collect_losses(model, feature_backbone, params, loader, device, activation_variant=False):
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses = []
    classes = []
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        if activation_variant:
            logits = model(params, data)
        else:
            logits = model(feature_backbone, params, data)

        if target.ndim > 1:
            class_target = torch.argmax(target, dim=1)
        else:
            class_target = target.long()

        one_hot_target = one_hot_from_target(class_target, logits.shape[1])
        batch_loss = criterion(logits, one_hot_target)
        losses.extend(batch_loss.detach().cpu().numpy().tolist())
        classes.extend(class_target.detach().cpu().numpy().tolist())
    return losses, classes


def attack_score(estimator, X, y):
    pred = estimator.predict(X)
    return float(np.mean(pred == y))


def evaluate_attack_model(sample_loss, members, n_splits=5, random_state=2023):
    unique_members = np.unique(members)
    if not np.array_equal(unique_members, np.array([0, 1])):
        raise ValueError('members should only have 0 and 1s')

    attack_model = LogisticRegression(max_iter=1000)
    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)
    return cross_val_score(attack_model, sample_loss, members, cv=cv, scoring=attack_score)


def membership_inference_attack(model, feature_backbone, params, test_loader, forget_loader, device,
                                activation_variant=False, seed=2023):
    test_losses, test_classes = collect_losses(model, feature_backbone, params, test_loader, device, activation_variant)
    forget_losses, forget_classes = collect_losses(model, feature_backbone, params, forget_loader, device, activation_variant)

    forget_class_set = set(forget_classes)
    filtered_test_losses = [loss for loss, cls in zip(test_losses, test_classes) if cls in forget_class_set]

    np.random.seed(seed)
    random.seed(seed)
    if len(forget_losses) > len(filtered_test_losses):
        forget_losses = random.sample(forget_losses, len(filtered_test_losses))
    elif len(filtered_test_losses) > len(forget_losses):
        filtered_test_losses = random.sample(filtered_test_losses, len(forget_losses))

    member_labels = np.array([0] * len(filtered_test_losses) + [1] * len(forget_losses))
    sample_loss = np.array(filtered_test_losses + forget_losses, dtype=np.float32).reshape(-1, 1)
    scores = evaluate_attack_model(sample_loss, member_labels, n_splits=5, random_state=seed)
    return {
        'mia_scores': scores.tolist(),
        'mia_mean': float(np.mean(scores)),
        'num_test_samples_used': len(filtered_test_losses),
        'num_forget_samples_used': len(forget_losses),
        'forget_classes': sorted(forget_class_set),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-file', required=True, type=str)
    parser.add_argument('--arch-id', required=True, type=str)
    parser.add_argument('--dataset-id', required=True, type=str)
    parser.add_argument('--number-of-linearized-components', required=True, type=int)
    parser.add_argument('--split-rate', required=True, type=float)
    parser.add_argument('--device-id', default=0, type=int)
    parser.add_argument('--activation-variant', action='store_true')
    parser.add_argument('--pretrained-model-path', default=None, type=str)
    parser.add_argument('--seed', default=2023, type=int)
    args = parser.parse_args()

    device = f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.checkpoint_file, map_location='cpu')
    pretrained_model_path = args.pretrained_model_path or checkpoint.get('pretrained_model_path')
    core_model_file = infer_core_model_path(args.checkpoint_file)

    feature_backbone, model = get_trained_linear(
        args.checkpoint_file,
        args.arch_id,
        args.dataset_id.split('-')[0],
        args.number_of_linearized_components,
        activation_variant=args.activation_variant,
        pretrained_model_path=pretrained_model_path,
    )
    params = get_core_model_params(core_model_file, device)

    if args.activation_variant:
        feature_backbone = None
    else:
        feature_backbone = feature_backbone.to(device)
        freeze(feature_backbone)

    model = model.to(device)
    freeze(model)

    _, test_loader = get_user_loader(
        args.dataset_id,
        args.arch_id,
        64,
        shuffle=False,
        number_of_linearized_components=args.number_of_linearized_components,
    )
    remaining_dataset, forget_dataset = split_user_train_dataset_to_remaining_forget(
        args.dataset_id,
        args.arch_id,
        args.split_rate,
        number_of_linearized_components=args.number_of_linearized_components,
    )
    _, forget_loader = get_remaining_forget_loader(remaining_dataset, forget_dataset, 64, shuffle=False)

    result = membership_inference_attack(
        model,
        feature_backbone,
        params,
        test_loader,
        forget_loader,
        device,
        activation_variant=args.activation_variant,
        seed=args.seed,
    )
    result.update({
        'checkpoint_file': args.checkpoint_file,
        'core_model_file': core_model_file,
        'pretrained_model_path': pretrained_model_path,
    })
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
