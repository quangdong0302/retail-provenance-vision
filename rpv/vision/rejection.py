"""Prototype energy OOD and empirical selective-risk calibration.

Energy follows Liu et al. (2020), with logits -squared prototype distance.
Main unknown-test and extra holdout never fit operating thresholds. Reserved
extra calibration classes may calibrate the risk gate under an explicit protocol.
"""
from __future__ import annotations

import numpy as np


def energy_score(distances, temperature: float = 0.05) -> float:
    values = np.asarray(list(distances.values()) if isinstance(distances, dict) else distances, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Energy requires finite, nonnegative class distances.')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Energy temperature must be positive and finite.')
    logits = -(values ** 2) / temperature
    maximum = logits.max()
    return float(-temperature * (maximum + np.log(np.exp(logits - maximum).sum())))


def fit_risk_threshold(correct, confidence, eligible, target_risk=0.01) -> dict:
    """Maximize empirical calibration coverage, retaining whole score ties.

    This is an empirical constraint, not a finite-sample population guarantee.
    An empty feasible set explicitly disables automatic answers.
    """
    correct = np.asarray(correct, dtype=bool)
    confidence = np.asarray(confidence, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if not 0 <= target_risk < 1 or correct.ndim != 1 or confidence.shape != correct.shape or eligible.shape != correct.shape:
        raise ValueError('Invalid selective calibration inputs.')
    if not np.isfinite(confidence).all():
        raise ValueError('Confidence must be finite.')
    indices = np.flatnonzero(eligible)
    order = indices[np.argsort(-confidence[indices], kind='stable')]
    selected = 0
    threshold = None
    errors_selected = 0
    if order.size:
        scores = confidence[order]
        endpoints = np.r_[np.flatnonzero(scores[1:] != scores[:-1]) + 1, len(order)]
        errors = np.cumsum(~correct[order])
        for count in endpoints:
            if errors[count - 1] / count <= target_risk:
                selected = int(count)
                threshold = float(scores[count - 1])
                errors_selected = int(errors[count - 1])
    return {'enabled': True, 'confidence_threshold': threshold,
            'target_risk': float(target_risk), 'n_calib': len(correct),
            'n_answered': selected, 'n_errors': errors_selected,
            'coverage_on_calib': selected / len(correct) if len(correct) else 0.0,
            'risk_on_calib': errors_selected / selected if selected else None,
            'scope': 'Empirical known-calibration risk only; no population/OOD guarantee.'}
