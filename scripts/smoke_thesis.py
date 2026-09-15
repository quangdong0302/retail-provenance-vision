"""Bounded real-image train/evaluation check, explicitly not a thesis benchmark."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import numpy as np

from rpv import config as app_config
from rpv.data.dataset import index_dataset, build_splits
from rpv.data.managed import split_manifest
from rpv.evaluation.pipeline import evaluate_system
from rpv.evaluation.report import write_report
from rpv.util import json_dumps
from rpv.vision.encoders import DinoV2Encoder
from rpv.vision.projection import train_projection_head


def main():
    cfg = app_config.settings.calibration
    full = build_splits(index_dataset(app_config.settings.paths.data_dir), cfg.k_shot,
                        cfg.class_split_seed, cfg.calib_test_seed)
    known = [full.train_classes[i] for i in np.linspace(0, len(full.train_classes)-1, min(4, len(full.train_classes))).astype(int)]
    novel = full.novel_classes[:2]
    subset = replace(full,
        protocol=full.protocol + '_smoke_only',
        train={c: full.train[c][:16] for c in known},
        validation={c: full.validation[c][:12] for c in known},
        support={c: full.support[c] for c in known},
        calib={c: full.calib[c][:12] for c in known},
        test={c: full.test[c][:12] for c in known},
        train_classes=known, novel_classes=novel,
        novel_support={c: full.novel_support[c][:5] for c in novel},
        novel_test={c: full.novel_test[c][:12] for c in novel},
        unknown_test=[p for c in novel for p in full.novel_support[c][:5] + full.novel_test[c][:12]],
    )
    subset.assert_disjoint()
    out = ROOT / 'var/thesis/smoke' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    out.mkdir(parents=True)
    manifest = split_manifest(subset)
    (out / 'splits.json').write_text(json_dumps(manifest), encoding='utf-8')
    print('SMOKE ONLY:', subset.summary(), flush=True)
    raw = DinoV2Encoder()
    features = {c: raw.backbone_features(paths) for c, paths in subset.train.items()}
    validation = {c: raw.backbone_features(paths) for c, paths in subset.validation.items()}
    trained = train_projection_head(
        features, validation_features=validation, out_dim=raw.backbone_dim,
        epochs=2, episodes_per_epoch=5, val_episodes=5,
        n_way=min(4, len(known)), k_shot=3, q_query=4, device=raw.device,
        backbone_tag=raw.backbone_tag, dataset_fingerprint=subset.dataset_fingerprint,
        split_fingerprint=manifest['fingerprint'], save_path=out / 'head.pt',
    )
    (out / 'training.json').write_text(json_dumps(trained), encoding='utf-8')
    encoder = DinoV2Encoder(checkpoint=out / 'head.pt')
    result, bank, calibration = evaluate_system(
        encoder=encoder, splits=subset, k_shot=cfg.k_shot,
        save_bank_to=out / 'prototypes.json', save_calibration_to=out / 'calibration.json',
    )
    result.warnings.insert(0, 'SMOKE TEST ONLY: a small selected subset and 2 short epochs; not full-data accuracy or thesis evidence.')
    paths = write_report(result.to_json(), out_dir=out, stem='smoke_only')
    assert len(result.novel_evaluation['runs']) == 3
    assert calibration.bank_fingerprint == bank.fingerprint()
    assert np.isfinite(trained['history'][-1]['train_loss'])
    print(json_dumps({'status': 'passed', 'scope': 'smoke_only', 'reports': paths}))


if __name__ == '__main__':
    main()
