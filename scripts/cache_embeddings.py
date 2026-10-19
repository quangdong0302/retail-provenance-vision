"""Persist raw backbone vectors once, reusable across projection-head training."""
from __future__ import annotations
import csv
import json
import sys
import time
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpv import config
from rpv.vision.encoders import load_encoder,image_sha256
from rpv.vision.cache import cache_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--color',action='store_true',help='Cache the independent HS cue instead of DINO.')
    args = parser.parse_args()
    if args.color:
        from rpv.vision.color import HueSaturationEncoder
        encoder = HueSaturationEncoder()
    else:
        encoder = load_encoder(checkpoint=False)
    root = config.settings.paths.data_dir.resolve()
    with (root / 'metadata.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    labels = sorted({r['label'] for r in rows if r['pool'] == '_extra_unknown'})
    np.random.default_rng(20261012).shuffle(labels)
    development = set(labels[12:24])
    # Head training/selection can start while final evaluation vectors are built.
    rows.sort(key=lambda r: (0 if r['pool'] == 'known' and r['role'] in
                            {'train', 'support', 'validation'} else
                            1 if r['label'] in development else
                            2 if r['pool'] == 'known' else
                            3 if r['label'] in set(labels[:12]) else 4))
    status_path = config.settings.paths.bank_path.parent / ('color_cache_status.json' if args.color else 'embedding_cache_status.json')
    status_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    def color_vector(path):
        key = cache_key(image_sha256(path),False)
        vector = encoder._cache().get(key)
        if vector is None:
            vector = encoder.embed_arrays([encoder._load(path)])[0]
        return key,vector

    for i in range(0, len(rows), 256):
        paths = [(root / r['path']).resolve() for r in rows[i:i+256]]
        if any(root not in p.parents for p in paths):
            raise ValueError('Image outside dataset root.')
        if args.color:
            try:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    for key,vector in executor.map(color_vector,paths):
                        encoder._cache().put(key,vector)
            finally:
                encoder._cache().flush()
        else:
            encoder.backbone_features(paths)
        done = min(i + 256, len(rows))
        status = dict(state='completed' if done == len(rows) else 'running',
                      completed=done, total=len(rows), seconds=time.monotonic()-start,
                      cache=encoder._cache().info())
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(status, indent=2), encoding='utf8')
        tmp.replace(status_path)
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
