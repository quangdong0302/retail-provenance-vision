"""Remove disposable Python/linter caches; preserve datasets and experiment artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ('rpv', 'api', 'app', 'scripts', 'tests')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    candidates = [ROOT / name for name in ('__pycache__', '.pytest_cache', '.ruff_cache')]
    for name in SOURCE_DIRS:
        candidates.extend((ROOT / name).rglob('__pycache__'))
    for path in sorted(set(candidates)):
        if not path.exists():
            continue
        resolved = path.resolve()
        relative = resolved.relative_to(ROOT)
        if path.is_symlink() or path.absolute() != resolved:
            raise ValueError(f'Refusing redirected cache path: {path}')
        if relative.parts[0] not in (*SOURCE_DIRS, '__pycache__', '.pytest_cache', '.ruff_cache'):
            raise ValueError(f'Cache outside allowed directories: {path}')
        print(f'{"Would remove" if args.dry_run else "Removing"}: {relative}')
        if not args.dry_run:
            shutil.rmtree(resolved)


if __name__ == '__main__':
    main()
