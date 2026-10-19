"""Quick DINOv2 integration check on a small real-image subset."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import os
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import numpy as np

from rpv import config as app_config
from rpv.audit.hash_chain import AuditLog
from rpv.data.dataset import build_splits, DatasetIndex
from rpv.data.managed import split_manifest, fingerprint, sha256
from rpv.evaluation.performance import benchmark_checkout
from rpv.evaluation.pipeline import evaluate_system
from rpv.evaluation.report import write_report
from rpv.evaluation.robustness import evaluate_robustness
from rpv.evaluation.scenarios import build_provenance_scenarios, evaluate_scenarios
from rpv.provenance.repository import ProvenanceRepository
from rpv.services.checkout import CheckoutService
from rpv.util import json_dumps
from rpv.vision.encoders import DinoV2Encoder


def main():
    cfg = app_config.settings.calibration
    print('SMOKE ONLY: verifying manifest; only used images will be hash-checked', flush=True)
    root = app_config.settings.paths.data_dir.resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    expected = manifest.pop('fingerprint')
    if manifest.get('schema_version') != 1 or fingerprint(manifest) != expected:
        raise ValueError('Manifest changed or is unsupported.')
    classes = sorted(manifest['known_classes'])
    if len(classes) < 8:
        raise ValueError('Review smoke requires at least 8 known classes.')
    known = [classes[i] for i in np.linspace(0, len(classes) - 1, 8).astype(int)]
    novel = sorted(manifest['unknown_classes'])[:2]
    selected = set(known + novel)
    index = DatasetIndex(unknown_classes=novel, fingerprint=expected, grouping=manifest['grouping'])
    for record in manifest['records']:
        if record['label'] in selected:
            path = str(root / record['path'])
            index.setdefault(record['label'], []).append(path)
            index.records[path] = record
    full = build_splits(index, cfg.k_shot, cfg.class_split_seed, cfg.calib_test_seed)
    subset = replace(full,
                     train={c: full.train[c][:16] for c in known},
                     validation={c: full.validation[c][:12] for c in known},
                     support={c: full.support[c] for c in known},
                     calib={c: full.calib[c][:16] for c in known},
                     test={c: full.test[c][:6] for c in known}, train_classes=known, novel_classes=novel,
                     novel_support={c: full.novel_support[c][:5] for c in novel},
                     novel_test={c: full.novel_test[c][:8] for c in novel},
                     unknown_test=[p for c in novel for p in full.novel_support[c][:5] + full.novel_test[c][:8]])
    subset.assert_disjoint()
    used = set(subset.unknown_test)
    for role in ('support', 'calib', 'test', 'novel_support', 'novel_test'):
        used.update(p for paths in getattr(subset, role).values() for p in paths)
    for path in used:
        resolved = Path(path).resolve()
        if root not in resolved.parents or sha256(resolved) != index.records[path]['sha256']:
            raise ValueError(f'Used image missing/changed/outside dataset: {path}')
    output = ROOT / 'var/thesis/review_smoke' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    output.mkdir(parents=True)
    (output / 'splits.json').write_text(json_dumps(split_manifest(subset)), encoding='utf-8')
    print('Loading real DINOv2 and fitting support/calibration', flush=True)
    encoder = DinoV2Encoder()
    result, bank, calibration = evaluate_system(encoder=encoder, splits=subset, k_shot=cfg.k_shot,
                                               save_bank_to=output / 'prototypes.json',
                                               save_calibration_to=output / 'calibration.json')
    print('Checking perturbations, policy scenarios and timing', flush=True)
    evaluation = result.to_json()
    evaluation['warnings'].insert(0, 'SMOKE ONLY: small selected subset, frozen DINOv2; manifest and used-image hashes checked, not the entire dataset. Not a full-data evaluation.')
    evaluation['robustness_on_test_split'] = evaluate_robustness(encoder, bank, calibration, subset, max_per_class=2)
    with ProvenanceRepository(output / 'scenarios.db') as repo:
        repo.init_schema()
        cases = build_provenance_scenarios(repo, subset.test, subset.unknown_test)
        service = CheckoutService(encoder, bank, repo, calibration, audit=AuditLog(output / 'audit.jsonl'))
        scenarios = evaluate_scenarios(service, cases)
        performance = benchmark_checkout(service, [cases[0].image_path], repeats=10, warmup=2,
                                         claimed_sku=cases[0].claimed_sku, claimed_lot=cases[0].claimed_lot)
        (output / 'performance.json').write_text(json_dumps(performance), encoding='utf-8')
        assert service.audit.verify()['valid']
    paths = write_report(evaluation, scenarios, out_dir=output, stem='review_smoke_only')
    assert calibration.ood['method'] == 'energy'
    assert len(scenarios['cases']) == 12
    print(json_dumps({'status': 'passed', 'scope': 'smoke_only', 'reports': paths,
                      'scenarios_passed': scenarios['n_passed'], 'performance': output / 'performance.json'}))


if __name__ == '__main__':
    main()
