"""Validation-only confidence study for the current frozen prototype model.

No training, test evaluation, deployment or production-artifact replacement.
All features must already exist in the shared embedding cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpv import config as app_config
from rpv.data.dataset import build_splits, index_dataset
from rpv.data.extra import extra_partition, risk_extra_partition
from rpv.vision.calibration import CalibrationModel
from rpv.vision.encoders import load_encoder
from rpv.vision.prototypes import PrototypeBank
from rpv.vision.rejection import fit_risk_threshold
from scripts.improve_prototypes import energy, operating


def mixture_risk(coverage, known_error_rate, unknown_accept_rate, unknown_fraction):
    """Risk among accepted items for an explicit unknown prevalence."""
    denominator = (1 - unknown_fraction) * coverage + unknown_fraction * unknown_accept_rate
    if denominator == 0:
        return None
    return ((1 - unknown_fraction) * known_error_rate + unknown_fraction * unknown_accept_rate) / denominator


def score(distances, energies, coefficient):
    if coefficient is None:
        return -energies
    nearest = np.sort(distances, axis=1)[:, :2]
    ratio = np.float32(1) - nearest[:, 0] / np.maximum(nearest[:, 1], np.float32(1e-9))
    return ratio.astype(float) - coefficient * energies


def measure(distances, truth, unknown, energies, unknown_energies, ood_threshold, coefficient, threshold):
    def accepted(d, e):
        nearest = np.sort(d, axis=1)[:, :2]
        eligible = (nearest[:, 1] - nearest[:, 0] > 1e-8) & (e <= ood_threshold)
        if threshold is None:
            return np.zeros(len(d), bool)
        return eligible & (score(d, e, coefficient) >= threshold)
    answers = accepted(distances, energies)
    unknown_answers = accepted(unknown, unknown_energies)
    wrong = answers & (distances.argmin(1) != truth)
    coverage, error_rate, far = float(answers.mean()), float(wrong.mean()), float(unknown_answers.mean())
    count = int(answers.sum() + unknown_answers.sum())
    return dict(known_accuracy=float((distances.argmin(1) == truth).mean()),
                known_coverage=coverage, known_risk=float(wrong.sum() / answers.sum()) if answers.any() else None,
                n_known_accepted=int(answers.sum()), n_known_errors=int(wrong.sum()),
                n_unknown_accepted=int(unknown_answers.sum()), unknown_false_accept_rate=far,
                natural_mixture_risk=float((wrong.sum() + unknown_answers.sum()) / count) if count else None,
                balanced_mixture_risk=mixture_risk(coverage, error_rate, far, .5),
                stress_mixture_risk_75pct_unknown=mixture_risk(coverage, error_rate, far, .75))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    settings = app_config.settings
    if settings.calibration.decision_rule != 'prototype_selective':
        parser.error('This study requires the current prototype_selective model.')
    output = args.out.resolve()
    if output in {settings.paths.bank_path.resolve(), settings.paths.calib_path.resolve(),
                  Path(settings.vision.checkpoint).resolve()}:
        parser.error('Output must not replace model artifacts.')
    source = {name: path for name, path in [('head', Path(settings.vision.checkpoint)),
              ('bank', settings.paths.bank_path), ('calibration', settings.paths.calib_path)]}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source.items()}
    print('Verifying dataset and fixed class partitions...', flush=True)
    cfg = settings.calibration
    splits = build_splits(index_dataset(settings.paths.data_dir), k_shot=cfg.k_shot,
                          class_seed=cfg.class_split_seed, calib_test_seed=cfg.calib_test_seed)
    partition = extra_partition(settings.paths.data_dir)
    risk = risk_extra_partition(partition)
    encoder = load_encoder()
    bank = PrototypeBank(encoder).load(settings.paths.bank_path)
    calibration = CalibrationModel.load(settings.paths.calib_path, encoder.tag)
    if calibration.bank_fingerprint != bank.fingerprint() or any(e.requires_operator for e in bank.entries.values()):
        raise ValueError('Study requires the unchanged base catalog, with no operator-only SKU.')
    if set(bank.sku_ids) != set(splits.calib) or set(bank.sku_ids) != set(splits.validation):
        raise ValueError('Current catalog does not match the fixed development split.')
    def refuse(*_args, **_kwargs):
        raise RuntimeError('Uncached image: warm the vector cache first; this study does not recompute features.')
    getattr(encoder, 'base', encoder)._backbone_arrays = refuse
    if hasattr(encoder, 'color'):
        encoder.color.embed_arrays = refuse
    classes = bank.sku_ids
    roles = {name: [p for c in classes for p in getattr(splits, name)[c]] for name in ('calib', 'validation')}
    roles.update(extra_cal=[str((settings.paths.data_dir / row['path']).resolve()) for row in risk['calibration']],
                 extra_val=[str((settings.paths.data_dir / row['path']).resolve()) for row in risk['selection']])
    matrices = {}
    for name, paths in roles.items():
        print(f'Cached vectors: {name}, n={len(paths)}', flush=True)
        matrices[name], labels = bank.distances_batch(encoder.embed_paths(paths))
        assert labels == classes
    truths = {name: np.array([i for i, c in enumerate(classes) for _ in getattr(splits, name)[c]])
              for name in ('calib', 'validation')}
    temperature = calibration.ood['temperature']
    energies = {name: energy(values, temperature) for name, values in matrices.items()}
    # Confirm numerical parity of the old score before studying a new one.
    probe = matrices['validation'][:100]
    for method in ('energy', 'nearest_ratio'):
        _, _, confidence = operating(probe, np.ones(len(classes)), energies['validation'][:100],
                                      calibration.ood['threshold'], 'prototype_selective', method)
        expected = score(probe, energies['validation'][:100], None if method == 'energy' else 0.)
        np.testing.assert_allclose(confidence, expected, atol=1e-7, rtol=1e-7)
    def validate(threshold, coefficient, gate):
        return measure(matrices['validation'], truths['validation'], matrices['extra_val'],
                       energies['validation'], energies['extra_val'], threshold, coefficient, gate)
    current_method = calibration.risk_control.get('confidence_method', 'energy')
    current = validate(calibration.ood['threshold'], None if current_method == 'energy' else 0.,
                       calibration.risk_control['confidence_threshold'])
    profiles = []
    for recall in (.8, .9, 1.):
        threshold = float(np.sort(energies['calib'])[int(np.ceil(len(energies['calib']) * recall)) - 1])
        for coefficient in (None, 0., .25, .5, 1., 2., 4., 8.):
            all_distances = np.concatenate([matrices['calib'], matrices['extra_cal']])
            all_energies = np.concatenate([energies['calib'], energies['extra_cal']])
            confidences = score(all_distances, all_energies, coefficient)
            nearest = np.sort(all_distances, axis=1)[:, :2]
            eligible = (nearest[:, 1] - nearest[:, 0] > 1e-8) & (all_energies <= threshold)
            correct = np.r_[matrices['calib'].argmin(1) == truths['calib'], np.zeros(len(matrices['extra_cal']), bool)]
            for target in (.02, .03, .04):
                gate = fit_risk_threshold(correct, confidences, eligible, target)
                profiles.append(dict(method='energy' if coefficient is None else 'ratio_minus_energy',
                    coefficient=coefficient, target_known_recall=recall, target_risk=target,
                    ood_threshold=threshold, gate=gate, validation=validate(threshold, coefficient, gate['confidence_threshold'])))
    risk_keys = ('known_risk', 'natural_mixture_risk', 'balanced_mixture_risk', 'stress_mixture_risk_75pct_unknown')
    feasible = [p for p in profiles if p['validation']['n_known_accepted'] >= 20 and
                all(p['validation'][k] is not None and p['validation'][k] < .05 for k in risk_keys)]
    winner = max(feasible, key=lambda p: (p['validation']['known_coverage'],
                  -p['validation']['stress_mixture_risk_75pct_unknown'])) if feasible else None
    unchanged = all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name] for name, path in source.items())
    assert unchanged, 'Production artifacts changed during the study.'
    result = dict(state='completed', head_sha256=hashes['head'], production_unchanged=unchanged,
        n_profiles=len(profiles), n_feasible=len(feasible), current_validation=current, proposed_validation=winner,
        test_used=False, deployment_performed=False, roles={name: len(paths) for name, paths in roles.items()},
        risk_partition_fingerprint=risk['fingerprint'], cache=encoder.describe(), profiles=profiles,
        scope='Empirical development comparison only. 75% unknown is a declared stress scenario, not measured POS prevalence. Known/novel final queries and extra holdout never score or select candidates. Accuracy is unchanged because the head/prototypes are frozen; post-enrollment risk remains a separate study.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('n_profiles', 'n_feasible', 'current_validation', 'proposed_validation')}, ensure_ascii=False, indent=2))
    print(f'Report: {output}', flush=True)


if __name__ == '__main__':
    main()
