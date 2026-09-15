"""Repeat frozen-encoder baselines across support seeds; summarize completed runs.

Does not select a winning model on test data. Full-data runs can be expensive.
Use the same locked dataset for all encoders. Head experiments require their
own matched training split and are intentionally not mixed into this script.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--encoders', nargs='+', choices=['classical', 'dinov2'], default=['dinov2'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--k-shot', type=int, default=3)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    output = Path(args.out) if args.out else ROOT / 'var/thesis/experiments' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    output.mkdir(parents=True, exist_ok=False)
    runs = []
    for encoder in args.encoders:
        for seed in args.seeds:
            env = os.environ.copy()
            env.update({'RPV_DATA_DIR': str(Path(args.data_dir).resolve()), 'RPV_ENCODER': encoder,
                        'RPV_PROJECTION_CKPT': '', 'RPV_CLASS_SPLIT_SEED': str(seed),
                        'RPV_REPORTS': str(output), 'PYTHONIOENCODING': 'utf-8'})
            command = [sys.executable, '-m', 'rpv.cli', 'evaluate', '--k-shot', str(args.k_shot)]
            print(f'{encoder}, support seed={seed}', flush=True)
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
            (output / f'{encoder}_{seed}.log').write_text(result.stderr, encoding='utf-8')
            if result.returncode:
                raise RuntimeError(f'Run thất bại; xem {output / f"{encoder}_{seed}.log"}')
            response = json.loads(result.stdout)
            runs.append({'encoder': encoder, 'seed': seed, 'reports': response['reports'],
                         'accuracy': response['closed_set']['accuracy_all_classes'],
                         'open_auroc': response['open_set']['auroc'],
                         'coverage': response['selective']['empirical_coverage'],
                         'verification_far': response['verification']['false_accept_rate']})
            (output / 'runs.json').write_text(json.dumps(runs, indent=2), encoding='utf-8')
    summary = {'note': 'Mean/std across support seeds on the same test set; not an independent population confidence interval.',
               'runs': runs, 'summary': {}}
    for encoder in args.encoders:
        group = [r for r in runs if r['encoder'] == encoder]
        summary['summary'][encoder] = {
            metric: {'mean': statistics.mean(r[metric] for r in group),
                     'std': statistics.stdev(r[metric] for r in group) if len(group) > 1 else 0.0}
            for metric in ('accuracy', 'open_auroc', 'coverage', 'verification_far')}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(output / 'summary.json')


if __name__ == '__main__':
    main()
