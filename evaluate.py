import argparse
import json
import os

import torch

from dataset import get_user_loader, split_user_train_dataset_to_remaining_forget, get_remaining_forget_loader
from model import get_core_model_params, get_trained_linear, freeze


@torch.no_grad()
def evaluate_loader(model, feature_backbone, params, loader, device, activation_variant=False):
    model.eval()
    correct = 0
    total = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        if activation_variant:
            logits = model(params, data)
        else:
            logits = model(feature_backbone, params, data)
        pred = torch.argmax(logits, dim=1)
        if target.ndim > 1:
            target = torch.argmax(target, dim=1)
        correct += torch.count_nonzero(pred == target).item()
        total += data.shape[0]
    return correct / total


def infer_core_model_path(checkpoint_file):
    dirname, filename = os.path.split(checkpoint_file)
    stem = '.'.join(filename.split('.')[:-1])
    local_core_model = os.path.join(dirname, f'{stem}_core_model.pth')
    if os.path.exists(local_core_model):
        return local_core_model

    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    source_checkpoint_path = checkpoint.get('source_checkpoint_path')
    if source_checkpoint_path is None:
        raise FileNotFoundError(
            f'Cannot infer core model path for {checkpoint_file}. '
            'No local *_core_model.pth found and checkpoint has no source_checkpoint_path.'
        )

    train_dir = os.path.abspath(source_checkpoint_path)
    core_candidates = sorted([
        os.path.join(train_dir, file_name)
        for file_name in os.listdir(train_dir)
        if file_name.endswith('_core_model.pth')
    ])
    if len(core_candidates) != 1:
        raise FileNotFoundError(
            f'Expected one *_core_model.pth in {train_dir}, got {core_candidates}'
        )
    return core_candidates[0]


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
    args = parser.parse_args()

    device = f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu'
    checkpoint_file = os.path.abspath(args.checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    core_model_file = infer_core_model_path(checkpoint_file)
    pretrained_model_path = args.pretrained_model_path or checkpoint.get('pretrained_model_path')

    feature_backbone, model = get_trained_linear(
        checkpoint_file,
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
    remaining_loader, forget_loader = get_remaining_forget_loader(
        remaining_dataset,
        forget_dataset,
        64,
        shuffle=False,
    )

    metrics = {
        'test_acc': evaluate_loader(model, feature_backbone, params, test_loader, device, args.activation_variant),
        'remaining_acc': evaluate_loader(model, feature_backbone, params, remaining_loader, device, args.activation_variant),
        'forget_acc': evaluate_loader(model, feature_backbone, params, forget_loader, device, args.activation_variant),
        'core_model_file': core_model_file,
        'checkpoint_file': checkpoint_file,
        'pretrained_model_path': pretrained_model_path,
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
