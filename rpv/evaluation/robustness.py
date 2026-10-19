"""Held-out image perturbations; synthetic stress tests do not replace POS data."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from rpv import config as app_config
from rpv.vision.encoders import load_rgb
from rpv.vision.recognizer import Recognizer, STATUS_MATCH, STATUS_UNKNOWN
from rpv.vision.verification import Verifier

CONDITIONS = ('clean', 'plastic_bag', 'mesh_bag', 'color_shift', 'dim_light', 'defocus_blur')


def perturb_image(array, condition):
    image = Image.fromarray(array).convert('RGB')
    width, height = image.size
    if condition == 'clean':
        return np.asarray(image)
    if condition == 'plastic_bag':
        overlay = Image.new('RGB', image.size, 'white')
        draw = ImageDraw.Draw(overlay)
        for x in range(0, width, max(1, width // 6)):
            draw.line((x, 0, min(width - 1, x + width // 4), height - 1), fill=(190, 190, 190), width=max(1, width // 45))
        image = Image.blend(image, overlay, .22)
    elif condition == 'mesh_bag':
        draw = ImageDraw.Draw(image)
        step = max(4, min(width, height) // 8)
        for x in range(-height, width + height, step):
            draw.line((x, 0, x + height, height), fill=(160, 85, 40), width=max(1, step // 9))
            draw.line((x, 0, x - height, height), fill=(160, 85, 40), width=max(1, step // 9))
    elif condition == 'color_shift':
        values = np.asarray(image, dtype=float) * np.array([1.15, .85, .9])
        image = Image.fromarray(np.clip(values, 0, 255).astype(np.uint8))
    elif condition == 'dim_light':
        image = ImageEnhance.Brightness(image).enhance(.55)
    elif condition == 'defocus_blur':
        image = image.filter(ImageFilter.GaussianBlur(max(.5, min(width, height) / 80)))
    else:
        raise ValueError(f'Unknown perturbation: {condition}')
    return np.asarray(image)


def evaluate_robustness(encoder, bank, calibration, splits, max_per_class=5, hard_pairs_csv=None):
    if max_per_class < 1:
        raise ValueError('max_per_class must be positive.')
    splits.assert_disjoint()
    recognizer = Recognizer(encoder, bank, calibration)
    verifier = Verifier(encoder, bank, calibration)
    samples = [(label, p) for label, paths in sorted(splits.test.items()) for p in paths[:max_per_class]]
    samples.extend(('__unknown__', p) for p in splits.unknown_test[:max_per_class])
    pairs = []
    if hard_pairs_csv:
        with Path(hard_pairs_csv).open(encoding='utf-8-sig', newline='') as stream:
            pairs = list(csv.DictReader(stream))
        for pair in pairs:
            if not {'true_sku', 'claimed_sku', 'kind'} <= pair.keys() or pair['true_sku'] not in bank.entries or pair['claimed_sku'] not in bank.entries or pair['true_sku'] == pair['claimed_sku']:
                raise ValueError('Hard pairs require two distinct known SKUs and kind (cultivar/origin).')
            if pair['kind'] not in {'cultivar', 'origin'}:
                raise ValueError('Hard pair kind must be cultivar or origin.')
    rows = []
    for condition in CONDITIONS:
        truth, top1, answered, answered_correct, rejected_unknown, hard_accepts = [], [], [], [], [], []
        for label, path in samples:
            array = perturb_image(load_rgb(path), condition)
            # Use exactly the same TTA and head path as normal inference.
            embedding = encoder._postprocess(encoder._compute_from_arrays([array], app_config.settings.vision.use_tta))[0]
            distances = bank.distances(embedding)
            result = recognizer.recognize_distances(distances)
            if label == '__unknown__':
                rejected_unknown.append(result.status == STATUS_UNKNOWN)
            else:
                truth.append(label)
                top1.append(min(distances, key=distances.get))
                for pair in pairs:
                    if pair['true_sku'] == label:
                        hard_accepts.append(verifier.verify_distances(distances, pair['claimed_sku']).accepted)
            answered.append(result.status == STATUS_MATCH)
            if result.status == STATUS_MATCH:
                answered_correct.append(label != '__unknown__' and result.prediction_set == [label])
        rows.append({'condition': condition, 'n_known': len(truth), 'n_unknown': len(rejected_unknown),
                     'accuracy_known': float(np.mean(np.array(truth) == np.array(top1))) if truth else None,
                     'coverage_known_and_unknown': float(np.mean(answered)) if answered else None,
                     'selective_risk_known_and_unknown': 1 - float(np.mean(answered_correct)) if answered_correct else None,
                     'unknown_reject_rate': float(np.mean(rejected_unknown)) if rejected_unknown else None,
                     'hard_negative_far': float(np.mean(hard_accepts)) if hard_accepts else None,
                     'n_hard_claims': len(hard_accepts)})
    return {'scope': 'Synthetic perturbations of held-out images; fixed model and thresholds. Color shift is not physical ripening.',
            'max_per_class': max_per_class, 'hard_pairs': pairs, 'conditions': rows,
            'limitations': 'Collect real bag/mesh/ripeness images by independent object/session. Origin cannot be proven from appearance.'}
