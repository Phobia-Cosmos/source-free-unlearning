import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_case(python_bin, common_args, case_args):
    cmd = [python_bin, 'linear_repro.py'] + common_args + case_args
    print('[RUN]', ' '.join(cmd))
    out = subprocess.check_output(cmd, cwd=ROOT, text=True)
    print(out)
    start = out.find('{')
    end = out.rfind('}')
    return json.loads(out[start:end + 1])


def summarize_row(result):
    metrics = result['metrics']
    return {
        'retrained': {
            'test': metrics['retrained']['test_acc'],
            'remaining': metrics['retrained']['remaining_acc'],
            'forget': metrics['retrained']['forget_acc'],
            'mia': metrics['retrained']['mia_mean'],
        },
        'unlearned_minus': {
            'test': metrics['unlearned_minus']['test_acc'],
            'remaining': metrics['unlearned_minus']['remaining_acc'],
            'forget': metrics['unlearned_minus']['forget_acc'],
            'mia': metrics['unlearned_minus']['mia_mean'],
        },
        'unlearned_plus': {
            'test': metrics['unlearned_plus']['test_acc'],
            'remaining': metrics['unlearned_plus']['remaining_acc'],
            'forget': metrics['unlearned_plus']['forget_acc'],
            'mia': metrics['unlearned_plus']['mia_mean'],
        },
        'sdp': result['sdp'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--cache-dir', type=str, default='artifacts/feature_cache')
    parser.add_argument('--save-dir', type=str, default='artifacts/linear_tables')
    parser.add_argument('--bound-train', action='store_true')
    parser.add_argument('--train-epochs', type=int, default=10)
    parser.add_argument('--retrain-epochs', type=int, default=20)
    parser.add_argument('--dataset-id', type=str, default='cifar10')
    args = parser.parse_args()

    python_bin = str(ROOT / '.venv-sfu' / 'bin' / 'python')
    common_args = [
        '--dataset-id', args.dataset_id,
        '--device-id', str(args.device_id),
        '--cache-dir', args.cache_dir,
        '--save-dir', args.save_dir,
        '--train-epochs', str(args.train_epochs),
        '--retrain-epochs', str(args.retrain_epochs),
    ]
    if args.bound_train:
        common_args.append('--bound-train')

    results = {'table1': {}, 'table2': {}, 'table3': {}}

    for split in [0.05, 0.1, 0.15]:
        result = run_case(
            python_bin,
            common_args,
            ['--split-rate', str(split), '--num-perturbations', '500', '--lambda-reg', '0.0005'],
        )
        results['table1'][str(split)] = summarize_row(result)

    for perturb in [250, 500, 1000]:
        result = run_case(
            python_bin,
            common_args,
            ['--split-rate', '0.1', '--num-perturbations', str(perturb), '--lambda-reg', '0.0005'],
        )
        results['table2'][str(perturb)] = summarize_row(result)

    for lambda_reg in [0.0, 0.0005, 0.001]:
        result = run_case(
            python_bin,
            common_args,
            ['--split-rate', '0.1', '--num-perturbations', '500', '--lambda-reg', str(lambda_reg)],
        )
        results['table3'][str(lambda_reg)] = summarize_row(result)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / 'linear_tables_summary.json'
    output_file.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f'saved_summary={output_file}')


if __name__ == '__main__':
    main()
