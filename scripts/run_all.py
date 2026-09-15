"""Chạy toàn bộ pipeline trên Windows, macOS và Linux."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*args: str, env: dict[str, str]) -> None:
    command = [sys.executable, *args]
    print("\n==>", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("dinov2", "classical"), default=None)
    parser.add_argument('--data-dir', default=None, help='dataset known/unknown đã chuẩn bị')
    parser.add_argument('--train-head', action='store_true', help='train head trước khi hiệu chỉnh/đánh giá')
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from rpv.config import settings
    env = os.environ.copy()
    if args.encoder:
        env["RPV_ENCODER"] = args.encoder
    if args.data_dir:
        env['RPV_DATA_DIR'] = args.data_dir
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    encoder = env.get("RPV_ENCODER", "dinov2")
    if args.train_head and encoder != 'dinov2':
        parser.error('--train-head yêu cầu --encoder dinov2; classical chỉ là baseline.')
    print(f"RPV pipeline, encoder={encoder}, Python={sys.version.split()[0]}")

    data_dir = Path(env.get('RPV_DATA_DIR', str(settings.paths.data_dir)))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    env['RPV_DATA_DIR'] = str(data_dir.resolve())
    if not (data_dir / 'manifest.json').is_file():
        parser.error(f'Không có manifest dataset thật tại {data_dir}. Chạy prepare-data trước.')
    # A fresh head starts from raw backbone embeddings, even if the shell still
    # points at a checkpoint deleted during reset. Baseline also has no head.
    if args.train_head or encoder == 'classical':
        env['RPV_PROJECTION_CKPT'] = ''
    steps = [
        ['-m', 'rpv.cli', 'dataset-info'],
        ["-m", "rpv.cli", "init-db"],
        ["-m", "rpv.cli", "warm-cache"],
    ]
    if args.train_head:
        head_path = env.get('RPV_HEAD_OUT', str(settings.paths.head_path))
        if not Path(head_path).is_absolute():
            head_path = str((ROOT / head_path).resolve())
        steps.append(['-m', 'rpv.cli', 'train-head', '--out', head_path])
    steps.extend([
        ["-m", "rpv.cli", "calibrate"],
        ["-m", "rpv.cli", "evaluate"],
        ["-m", "unittest", "discover", "-s", "tests", "-t", "."],
        ["-m", "rpv.cli", "audit-verify"],
    ])
    try:
        for step in steps:
            run(*step, env=env)
            if len(step) > 2 and step[2] == 'train-head':
                env['RPV_PROJECTION_CKPT'] = head_path
    except subprocess.CalledProcessError as exc:
        print(f"\nDỪNG: bước trên thất bại (exit code {exc.returncode}).", file=sys.stderr)
        return exc.returncode or 1

    print(f"\nXong. Báo cáo nằm trong {env.get('RPV_REPORTS', str(settings.paths.reports_dir))}/.")
    if args.train_head:
        print(f'Git Bash trước khi mở API/demo: export RPV_PROJECTION_CKPT="{head_path}"')
    print("API : python -m uvicorn api.main:app --reload")
    print("Demo: python -m app.demo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
