import argparse
import glob
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def latest_checkpoint(pattern):
    matches = sorted(glob.glob(str(ROOT / pattern)))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[-1]


def latest_model_file(pattern):
    matches = sorted(
        match for match in glob.glob(str(ROOT / pattern))
        if match.endswith('.pth')
        and not match.endswith('_core_model.pth')
        and '_remaining_grads' not in match
        and '_expected_hess_diags' not in match
        and '_expected_hess_diags_inv' not in match
        and '_forgetting_update' not in match
        and '_grads' not in match
        and '_v_param' not in match
    )
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[-1]


def run_cmd(cmd, env):
    print('[RUN]', ' '.join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def capture_json(cmd, env):
    print('[EVAL]', ' '.join(cmd))
    raw = subprocess.check_output(cmd, cwd=ROOT, env=env, text=True)
    print(raw)
    start = raw.find('{')
    end = raw.rfind('}')
    if start == -1 or end == -1 or end < start:
        raise ValueError(f'No JSON object found in output: {raw}')
    return json.loads(raw[start:end + 1])


def ensure_metadata(forget_model, train_dir, pretrain_file):
    python = str(ROOT / '.venv-sfu' / 'bin' / 'python')
    patch_script = f"""
import torch
path = {forget_model!r}
ckpt = torch.load(path, map_location='cpu')
changed = False
if ckpt.get('source_checkpoint_path') != {train_dir!r}:
    ckpt['source_checkpoint_path'] = {train_dir!r}
    changed = True
if ckpt.get('pretrained_model_path') != {pretrain_file!r}:
    ckpt['pretrained_model_path'] = {pretrain_file!r}
    changed = True
if changed:
    torch.save(ckpt, path)
    print('patched metadata', path)
else:
    print('metadata ok', path)
"""
    subprocess.run([python, '-c', patch_script], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--split-rate', type=float, default=0.1)
    parser.add_argument('--num-iter', type=int, default=20)
    parser.add_argument('--arch-id', type=str, default='resnet18')
    parser.add_argument('--dataset-id', type=str, default='cifar10')
    parser.add_argument('--number-of-linearized-components', type=int, default=1)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--skip-pretrain', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-forget', action='store_true')
    parser.add_argument('--run-mia', action='store_true')
    args = parser.parse_args()

    python = str(ROOT / '.venv-sfu' / 'bin' / 'python')
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    if not args.skip_pretrain:
        run_cmd([
            python, 'main.py',
            '--mode', 'pretrain',
            '--arch-id', args.arch_id,
            '--dataset-id', args.dataset_id,
            '--split-rate', str(args.split_rate),
            '--device-id', str(args.device_id),
        ], env)

    pretrain_file = latest_checkpoint(
        f'checkpoint/*-pretrain-{args.arch_id}-{args.dataset_id}-split{args.split_rate}/*.pth'
    )

    if not args.skip_train:
        run_cmd([
            python, 'main.py',
            '--mode', 'train-user-data',
            '--arch-id', args.arch_id,
            '--dataset-id', args.dataset_id,
            '--number-of-linearized-components', str(args.number_of_linearized_components),
            '--pretrained-model-path', pretrain_file,
            '--split-rate', str(args.split_rate),
            '--weight-decay', str(args.weight_decay),
            '--device-id', str(args.device_id),
        ], env)

    train_dir = latest_checkpoint(
        f'checkpoint/*-train-user-data-{args.arch_id}-{args.dataset_id}-last{args.number_of_linearized_components}-split{args.split_rate}'
    )

    if not args.skip_forget:
        run_cmd([
            python, 'main.py',
            '--mode', 'forget-by-diag',
            '--arch-id', args.arch_id,
            '--dataset-id', args.dataset_id,
            '--number-of-linearized-components', str(args.number_of_linearized_components),
            '--split-rate', str(args.split_rate),
            '--weight-decay', str(args.weight_decay),
            '--checkpoint-path', train_dir,
            '--device-id', str(args.device_id),
            '--num-iter-for-diag', str(args.num_iter),
            '--pretrained-model-path', pretrain_file,
        ], env)

    train_model = latest_model_file(f'{os.path.relpath(train_dir, ROOT)}/*.pth')
    forget_model = latest_model_file(
        f'checkpoint/*-forget-by-diag-{args.arch_id}-{args.dataset_id}-last{args.number_of_linearized_components}-split{args.split_rate}-iter{args.num_iter}/*.pth'
    )

    ensure_metadata(forget_model, train_dir, pretrain_file)

    summary = {
        'config': vars(args),
        'artifacts': {
            'pretrain_file': pretrain_file,
            'train_dir': train_dir,
            'train_model': train_model,
            'forget_model': forget_model,
        },
        'metrics': {},
    }

    for label, checkpoint_file in [('train_user_data', train_model), ('forget_by_diag', forget_model)]:
        summary['metrics'][label] = capture_json([
            python, 'evaluate.py',
            '--checkpoint-file', checkpoint_file,
            '--arch-id', args.arch_id,
            '--dataset-id', args.dataset_id,
            '--number-of-linearized-components', str(args.number_of_linearized_components),
            '--split-rate', str(args.split_rate),
            '--device-id', str(args.device_id),
            '--pretrained-model-path', pretrain_file,
        ], env)

        if args.run_mia:
            summary['metrics'][f'{label}_mia'] = capture_json([
                python, 'mia.py',
                '--checkpoint-file', checkpoint_file,
                '--arch-id', args.arch_id,
                '--dataset-id', args.dataset_id,
                '--number-of-linearized-components', str(args.number_of_linearized_components),
                '--split-rate', str(args.split_rate),
                '--device-id', str(args.device_id),
                '--pretrained-model-path', pretrain_file,
            ], env)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
