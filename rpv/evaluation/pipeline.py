"""Chạy đánh giá thị giác end-to-end trên một bộ dữ liệu.

Thứ tự bắt buộc (và là điểm sửa quan trọng nhất so với bản trước của dự án):

    support -> đăng ký prototype
    calib   -> CHỌN ngưỡng / phân vị conformal
    test    -> BÁO CÁO số liệu

Không bao giờ đo trên tập đã dùng để chọn ngưỡng.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from rpv import config as app_config
from rpv.util import log, stable_hash_int
from rpv.data.dataset import DatasetSplits, build_splits, index_dataset
from rpv.evaluation.metrics import (
    closed_set_accuracy, coverage_at_risk, confusion_pairs, macro_f1, open_set_metrics,
    per_class_recall, risk_coverage_curve, verification_metrics,
)
from rpv.vision.calibration import CalibrationModel, Calibrator
from rpv.vision.encoders import BaseEncoder, load_encoder
from rpv.vision.prototypes import PrototypeBank
from rpv.vision.recognizer import Recognizer, STATUS_MATCH, STATUS_AMBIGUOUS, STATUS_UNKNOWN
from rpv.vision.verification import Verifier
from rpv.vision.rejection import energy_score, fit_risk_threshold

UNKNOWN_LABEL = "__unknown__"


@dataclass
class EvaluationOutput:
    encoder_tag: str
    splits_summary: dict
    calibration: dict
    closed_set: dict
    open_set: dict
    selective: dict
    verification: dict
    errors: dict
    config: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    novel_evaluation: dict = field(default_factory=dict)
    predictions: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "encoder_tag": self.encoder_tag,
            "config": self.config,
            "splits": self.splits_summary,
            "calibration_on_calib_split": self.calibration,
            "closed_set_on_test_split": self.closed_set,
            "open_set_on_test_split": self.open_set,
            "selective_prediction_on_test_split": self.selective,
            "verification_on_test_split": self.verification,
            "error_analysis": self.errors,
            "warnings": self.warnings,
            "novel_after_enrollment": self.novel_evaluation,
            "predictions": self.predictions,
            "methodology_note": (
                "Prototype dựng từ tập support; ngưỡng và phân vị conformal chọn trên tập calib; "
                "metric đo trên test tách khỏi support/calib trong từng phép thử. "
                "Main unknown-test không tham gia học/chọn head hoặc hiệu chỉnh. "
                "Danh mục dựng từ support; head chỉ học trên các lớp base đã công bố. "
                "six_extra_cal dùng sáu lớp extra-calibration, sáu lớp khác cho lựa chọn; holdout tách riêng. "
                "Hai phép thử unknown trước và sau đăng ký dùng cùng pool và được báo riêng."
            ),
        }


def _embed_map(encoder: BaseEncoder, mapping: dict[str, list[str]]) -> dict[str, np.ndarray]:
    return {c: encoder.embed_paths(paths) for c, paths in mapping.items() if paths}


def fit_catalog(encoder, splits, alpha=None):
    """Fit on support/calib only. This function never reads test/unknown-test."""
    splits.assert_disjoint()
    if getattr(encoder, 'head', None) is not None:
        from rpv.data.managed import split_manifest
        metadata = getattr(encoder, 'head_metadata', {})
        if metadata.get('split_fingerprint') != split_manifest(splits)['fingerprint']:
            raise ValueError('Head không thuộc manifest split hiện tại. Dùng đúng dữ liệu/seed/k-shot của lần train hoặc train lại head.')
    bank = PrototypeBank(encoder)
    for label, paths in splits.support.items():
        entry = bank.register(label, paths, label=label)
        if entry.n_shots != len(paths):
            raise RuntimeError(f'{label}: ảnh support lỗi, không thể báo cáo đúng k-shot.')
    cfg = app_config.settings.calibration
    scorer = CalibrationModel(encoder.tag, cfg.alpha, 0., conformal_score=cfg.conformal_score)
    true_distances, nearest, calibration_rows = {}, [], []
    for label, paths in splits.calib.items():
        distances, labels = bank.distances_batch(encoder.embed_paths(paths))
        true_distances[label] = [scorer.nonconformity(dict(zip(labels, map(float, row))))[label] for row in distances]
        nearest.extend(distances.min(axis=1).tolist())
        calibration_rows.extend((label, dict(zip(labels, map(float, row)))) for row in distances)
    calibration = Calibrator(encoder.tag, alpha=alpha).fit(
        true_distances, known_detection_distances=nearest)
    calibration.bank_fingerprint = bank.fingerprint()
    calibration.conformal_score = cfg.conformal_score
    calibration.decision_rule = cfg.decision_rule
    if cfg.ood_method not in {'energy', 'distance'}:
        raise ValueError('RPV_OOD_METHOD must be energy or distance.')
    scores = [energy_score(row, cfg.energy_temperature) if cfg.ood_method == 'energy'
              else min(row.values()) for _, row in calibration_rows]
    threshold = float(np.sort(scores)[int(np.ceil(len(scores) * cfg.target_known_recall)) - 1])
    calibration.ood = {'method': cfg.ood_method, 'temperature': cfg.energy_temperature,
                       'threshold': threshold, 'n_calib': len(scores),
                       'fit_split': 'known-calibration', 'score_direction': 'higher_is_ood'}
    calibration_recognizer = Recognizer(encoder, bank, calibration)
    risk_rows = list(calibration_rows)
    if cfg.risk_calibration in {'joint_extra', 'reserved_extra', 'six_extra_cal'}:
        from rpv.data.extra import extra_partition
        root = app_config.settings.paths.data_dir
        partition = extra_partition(root)
        if getattr(encoder, 'head', None) is not None:
            fitted_extra = encoder.head_metadata.get('development_fingerprint')
            if fitted_extra and fitted_extra != partition['fingerprint']:
                raise ValueError('Extra development metadata changed after head training; train again.')
        main_labels = set(splits.support) | set(splits.novel_classes)
        if main_labels & {r['label'] for role in ('reserved','development','holdout') for r in partition[role]}:
            raise ValueError('Extra classes overlap known/main unknown classes.')
        if cfg.risk_calibration == 'six_extra_cal':
            from rpv.data.extra import risk_extra_partition
            risk_records = risk_extra_partition(partition)['calibration']
        else:
            risk_records = partition['development' if cfg.risk_calibration == 'joint_extra' else 'reserved']
        paths = [str((Path(root) / r['path']).resolve()) for r in risk_records]
        distances, labels = bank.distances_batch(encoder.embed_paths(paths))
        risk_rows.extend((None, dict(zip(labels, map(float, row)))) for row in distances)
        calibration.development_fingerprint = partition['fingerprint']
    correct, eligible, confidences = [], [], []
    calibration.risk_control = {'confidence_method': cfg.confidence_method}
    for label, row in risk_rows:
        result = calibration_recognizer.recognize_distances(row)
        if calibration.decision_rule == 'prototype_selective':
            ordered = sorted(row, key=row.get)
            correct.append(ordered[0] == label)
            eligible.append(not calibration.is_ood(row) and
                            row[ordered[1]] - row[ordered[0]] > 1e-8)
        else:
            correct.append(result.prediction_set == [label])
            eligible.append(result.status == STATUS_MATCH)
        confidences.append(calibration.confidence(row))
    calibration.risk_control = fit_risk_threshold(correct, confidences, eligible, cfg.target_selective_risk)
    calibration.risk_control.update(confidence_method=cfg.confidence_method,
        fit_split=cfg.risk_calibration, n_known_calib=len(calibration_rows),
        n_extra_development=len(risk_rows)-len(calibration_rows),
        scope='Empirical calibration constraint; no population guarantee. Unknown-test and extra-holdout excluded.')
    calibration.calib_metrics['risk_calibration'] = cfg.risk_calibration
    calibration.calib_metrics['n_extra_development'] = len(risk_rows)-len(calibration_rows)
    return bank, calibration


def evaluate_enrollment(encoder, bank, calibration, splits, shots=(1, 2, 3, 5)):
    """A separate, post-enrollment experiment; never changes the open-set bank."""
    if not splits.novel_support:
        return {}
    queries = _embed_map(encoder, splits.novel_test)
    results = []
    for k in shots:
        if any(len(paths) < k for paths in splits.novel_support.values()):
            results.append({'k_shot': k, 'skipped': 'Không đủ nhóm support ở mọi lớp novel.'})
            continue
        enrolled = PrototypeBank(encoder, bank.centroids_per_sku)
        enrolled.entries = dict(bank.entries)
        for label, paths in splits.novel_support.items():
            # Novel registration uses the same prototype geometry as deployment.
            novel_bank = PrototypeBank(encoder, bank.centroids_per_sku)
            entry = novel_bank.register(label, paths[:k], label=label)
            enrolled.entries[label] = entry
            if entry.n_shots != k:
                raise RuntimeError(f'{label}: không đọc đủ {k} ảnh novel support.')
        # Catalog size changes logsumexp energy; known thresholds cannot be reused.
        transferred = replace(calibration, bank_fingerprint=enrolled.fingerprint(), ood={}, risk_control={})
        recognizer = Recognizer(encoder, enrolled, transferred)
        truth, predicted, covered = [], [], []
        for label, vectors in queries.items():
            distances, labels = enrolled.distances_batch(vectors)
            for row in distances:
                result = recognizer.recognize_distances(dict(zip(labels, map(float, row))))
                truth.append(label)
                predicted.append(labels[int(np.argmin(row))])
                covered.append(label in result.conformal_prediction_set)
        results.append({'k_shot': k, 'n_query': len(truth), 'n_catalog_classes': len(enrolled),
                        'accuracy': closed_set_accuracy(truth, predicted),
                        'macro_f1': macro_f1(truth, predicted, labels=sorted(queries)),
                        'macro_f1_labels': sorted(queries),
                        'macro_f1_scope': 'Novel true classes only; base-class predictions count as false negatives.',
                        'macro_f1_catalog': macro_f1(truth, predicted, labels=enrolled.sku_ids),
                        'empirical_coverage_transferred_threshold': float(np.mean(covered))})
    return {'protocol': 'Known catalog + enrolled unknown classes; fixed query images across k.',
            'threshold_note': 'Pooled known conformal calibration transferred to novel classes; no class-conditional guarantee for novel. Energy/risk gates disabled because the catalog changed.',
            'runs': results}


def evaluate_system(data_dir: str | Path | None = None,
                    encoder: BaseEncoder | None = None,
                    k_shot: int | None = None,
                    alpha: float | None = None,
                    splits: DatasetSplits | None = None,
                    save_bank_to: str | Path | None = None,
                    save_calibration_to: str | Path | None = None,
                    verbose: bool = True,
                    catalog: tuple[PrototypeBank, CalibrationModel] | None = None,
                    ) -> tuple[EvaluationOutput, PrototypeBank, CalibrationModel]:
    cfg = app_config.settings.calibration
    k_shot = cfg.k_shot if k_shot is None else k_shot
    alpha = cfg.alpha if alpha is None else alpha
    encoder = encoder or load_encoder()
    data_dir = Path(data_dir or app_config.settings.paths.data_dir)

    if splits is None:
        splits = build_splits(
            index_dataset(data_dir), k_shot=k_shot,
            class_seed=cfg.class_split_seed,
            calib_test_seed=cfg.calib_test_seed,
        )
    warnings: list[str] = []
    if splits.grouping == 'content_hash':
        warnings.append('Split theo ảnh/hash: chưa có object/session metadata để loại trừ frame gần trùng. Không suy rộng kết quả này thành hiệu năng tại quầy.')
    if verbose:
        log(f"[eval] encoder={encoder.tag} splits={splits.summary()}")
    splits.assert_disjoint()
    # Fit uses only support/calibration, never the unknown test pool.
    if catalog is None:
        bank, calibration = fit_catalog(encoder, splits, alpha)
    else:
        bank, calibration = catalog
        if bank.encoder.tag != encoder.tag or set(bank.sku_ids) != set(splits.support):
            raise ValueError('Frozen evaluation catalog does not match encoder or known classes.')
        if alpha != calibration.alpha:
            raise ValueError('Frozen calibration uses a different alpha.')
        # Recognizer validates the exact bank fingerprint and encoder identity.
        Recognizer(encoder, bank, calibration)

    # 2) Embedding
    test_emb = _embed_map(encoder, splits.test)
    unk_test = encoder.embed_paths(splits.unknown_test) if splits.unknown_test else np.zeros((0, encoder.dim), np.float32)

    # 4) ĐO TRÊN TEST
    recognizer = Recognizer(encoder, bank, calibration)
    verifier = Verifier(encoder, bank, calibration)

    pred_set_sizes: list[int] = []
    y_true_closed, y_pred_closed = [], []
    y_true_open, y_pred_open = [], []
    known_dists: list[float] = []
    known_ood, unknown_ood, operational_correct = [], [], []
    correct_flags, confidences = [], []
    status_counter: dict[str, int] = {}
    ver_records: list[dict] = []
    hard_records: list[dict] = []
    covered, filtered_covered, decision_covered, singleton_correct, prediction_rows = [], [], [], [], []
    all_skus = bank.sku_ids

    for cname, embs in test_emb.items():
        D, skus = bank.distances_batch(embs)
        for row_i in range(D.shape[0]):
            row = D[row_i]
            pred = skus[int(np.argmin(row))]
            y_true_closed.append(cname)
            y_pred_closed.append(pred)
            known_dists.append(float(row.min()))

            distances = dict(zip(skus, map(float, row)))
            res = recognizer.recognize_distances(distances)
            known_ood.append(calibration.ood_score(distances))
            if res.status == STATUS_MATCH:
                operational_correct.append(res.prediction_set[0] == cname)
            covered.append(cname in res.conformal_prediction_set)
            conformal_after_ood = [] if calibration.is_ood(distances) else res.conformal_prediction_set
            filtered_covered.append(cname in conformal_after_ood)
            decision_covered.append(cname in res.prediction_set)
            if res.status == STATUS_MATCH:
                singleton_correct.append(res.prediction_set[0] == cname)
            prediction_rows.append({'path': splits.test[cname][row_i], 'truth': cname,
                                    'top1': pred, 'status': res.status,
                                    'prediction_set': res.prediction_set,
                                    'conformal_prediction_set': res.conformal_prediction_set,
                                    'conformal_prediction_set_after_ood': conformal_after_ood,
                                    'confidence': calibration.confidence(distances),
                                    'nearest_distance': float(row.min()),
                                    'ood_score': res.ood_score, 'rejection_reason': res.rejection_reason})
            status_counter[res.status] = status_counter.get(res.status, 0) + 1
            pred_set_sizes.append(len(res.conformal_prediction_set))
            y_true_open.append(cname)
            if res.status == STATUS_UNKNOWN or not res.prediction_set:
                y_pred_open.append(UNKNOWN_LABEL)
            else:
                y_pred_open.append(res.prediction_set[0] if res.status == STATUS_MATCH else pred)
            # Diagnostic curve for the top-1 classifier; deployed risk gate is
            # independently fitted on known-calibration, never on this curve.
            conf = calibration.confidence(distances)
            correct_flags.append(pred == cname)
            confidences.append(conf)

            # Kiểm chứng: 1 lời khai đúng + 1 lời khai sai (mô phỏng đổi mã)
            vr = verifier.verify_distances(distances, cname)
            ver_records.append({"genuine": True, "accepted": vr.accepted, "score": vr.claimed_distance})
            hard_records.append(ver_records[-1])
            others = [s for s in all_skus if s != cname]
            if others:
                # hash() của Python bị random hoá theo tiến trình: dùng nó ở đây
                # nghĩa là mỗi lần chạy chọn một impostor khác và FAR trong báo
                # cáo không tái lập được. stable_hash_int cố định giữa các lần chạy.
                impostor = others[(row_i + stable_hash_int(cname)) % len(others)]
                vi = verifier.verify_distances(distances, impostor)
                ver_records.append({"genuine": False, "accepted": vi.accepted, "score": vi.claimed_distance})
                hardest = min(others, key=distances.get)
                vh = verifier.verify_distances(distances, hardest)
                hard_records.append({'genuine': False, 'accepted': vh.accepted, 'score': vh.claimed_distance})

    unk_test_dists: list[float] = []
    n_unknown_rejected, n_unknown_auto_accepted = 0, 0
    if unk_test.shape[0]:
        D, labels = bank.distances_batch(unk_test)
        unk_test_dists = [float(x) for x in D.min(axis=1)]
        for i in range(unk_test.shape[0]):
            distances = dict(zip(labels, map(float, D[i])))
            res = recognizer.recognize_distances(distances)
            unknown_ood.append(calibration.ood_score(distances))
            if res.status == STATUS_MATCH:
                operational_correct.append(False)
                n_unknown_auto_accepted += 1
            prediction_rows.append({'path': splits.unknown_test[i], 'truth': UNKNOWN_LABEL,
                                    'top1': labels[int(np.argmin(D[i]))], 'status': res.status,
                                    'prediction_set': res.prediction_set,
                                     'conformal_prediction_set': res.conformal_prediction_set,
                                     'conformal_prediction_set_after_ood': [] if calibration.is_ood(distances) else res.conformal_prediction_set,
                                     'confidence': calibration.confidence(distances),
                                    'nearest_distance': float(D[i].min()),
                                    'ood_score': res.ood_score, 'rejection_reason': res.rejection_reason})
            y_true_open.append(UNKNOWN_LABEL)
            if res.status == STATUS_UNKNOWN or not res.prediction_set:
                # ``not prediction_set`` bắt cả trường hợp danh mục rỗng: bản
                # trước lấy prediction_set[0] và ném IndexError ở đúng nhánh
                # đáng ra phải được báo cáo tử tế.
                y_pred_open.append(UNKNOWN_LABEL)
                n_unknown_rejected += 1
            else:
                y_pred_open.append(res.prediction_set[0])

    curve = risk_coverage_curve(correct_flags, confidences)
    recalls = per_class_recall(y_true_closed, y_pred_closed)
    worst = sorted(recalls.items(), key=lambda kv: kv[1])[:8]

    target_risk = calibration.risk_control.get('target_risk', cfg.target_selective_risk)
    known_risk = 1 - float(np.mean(singleton_correct)) if singleton_correct else None
    mixed_risk = 1 - float(np.mean(operational_correct)) if operational_correct else None
    output = EvaluationOutput(
        encoder_tag=encoder.tag,
        splits_summary=splits.summary(),
        calibration={
            "alpha": calibration.alpha,
            "global_threshold": calibration.global_threshold,
            "global_conformal_quantile": calibration.global_conformal_quantile,
            "n_per_class_thresholds": len(calibration.per_class_threshold),
            "n_fallback_to_global": len(calibration.fallback_classes),
            "metrics_on_calib": calibration.calib_metrics,
            "warnings": calibration.warnings,
            'ood': calibration.ood,
            'risk_control': calibration.risk_control,
            'conformal_score': calibration.conformal_score,
            'development_fingerprint': calibration.development_fingerprint,
        },
        closed_set={
            "accuracy_all_classes": closed_set_accuracy(y_true_closed, y_pred_closed),
            "macro_f1_all_classes": macro_f1(y_true_closed, y_pred_closed),
            "macro_f1_with_unknown": macro_f1(y_true_open, y_pred_open),
            "n_test_images": len(y_true_closed),
        },
        open_set={**open_set_metrics(known_ood, unknown_ood, calibration.ood['threshold']),
                  'score': calibration.ood['method'],
                  'distance_baseline': open_set_metrics(known_dists, unk_test_dists, calibration.global_threshold)},
        selective={
            "status_distribution": status_counter,
            "mean_prediction_set_size": float(np.mean(pred_set_sizes)) if pred_set_sizes else float("nan"),
            "empirical_coverage": float(np.mean(covered)) if covered else float('nan'),
            'empirical_coverage_after_ood': float(np.mean(filtered_covered)) if filtered_covered else None,
            'decision_set_label_coverage': float(np.mean(decision_covered)) if decision_covered else None,
            'set_metrics_scope': 'Raw conformal sets before Energy/risk gates; combined gate has no conformal coverage guarantee.',
            "nonempty_set_rate": float(np.mean(np.asarray(pred_set_sizes) > 0)) if pred_set_sizes else float('nan'),
            "singleton_accuracy": float(np.mean(singleton_correct)) if singleton_correct else float('nan'),
            "n_singleton": len(singleton_correct),
            'n_singleton_conformal_before_gates': sum(size == 1 for size in pred_set_sizes),
            "target_coverage": 1.0 - calibration.alpha,
            "abstention_rate": (status_counter.get(STATUS_UNKNOWN, 0) + status_counter.get(STATUS_AMBIGUOUS, 0))
                               / max(1, len(y_true_closed)),
            "unknown_items_correctly_rejected": (n_unknown_rejected / unk_test.shape[0]) if unk_test.shape[0] else float("nan"),
            'unknown_auto_accept_rate': n_unknown_auto_accepted / unk_test.shape[0] if unk_test.shape[0] else None,
            'unknown_handoff_rate': 1 - n_unknown_auto_accepted / unk_test.shape[0] if unk_test.shape[0] else None,
            "risk_coverage_curve": curve,
            "coverage_at_risk_1pct": coverage_at_risk(curve, 0.01),
            "coverage_at_risk_5pct": coverage_at_risk(curve, 0.05),
            "coverage_at_risk_10pct": coverage_at_risk(curve, 0.10),
            'risk_coverage_scope': 'Known-test diagnostic sweep; never used to select the deployed threshold.',
            'operating_point_known': {
                'target_risk': target_risk,
                'coverage': len(singleton_correct) / max(1, len(y_true_closed)),
                'selective_risk': known_risk,
                'n_answered': len(singleton_correct),
                'target_met_on_test': known_risk is not None and known_risk <= target_risk,
            },
            'operating_point_known_and_unknown': {
                'target_risk': target_risk,
                'coverage': len(operational_correct) / max(1, len(y_true_open)),
                'selective_risk': mixed_risk,
                'n_answered': len(operational_correct),
                'target_met_on_test': mixed_risk is not None and mixed_risk <= target_risk,
            },
        },
        verification={**verification_metrics(ver_records),
                      'nearest_wrong_claim': verification_metrics(hard_records),
                      'scope': 'Verifier confirmed only; not final policy auto-allow.'},
        errors={
            "top_confusion_pairs": confusion_pairs(y_true_closed, y_pred_closed),
            "worst_per_class_recall": [{"sku": k, "recall": v} for k, v in worst],
        },
        config={
            'metric_schema': 'selective_sets_v2',
            "k_shot": k_shot, "alpha": alpha, "data_dir": str(data_dir),
            "centroids_per_sku": bank.centroids_per_sku,
            "class_split_seed": cfg.class_split_seed, "calib_test_seed": cfg.calib_test_seed,
            "use_tta": app_config.settings.vision.use_tta,
            "encoder": encoder.describe(),
            "novel_classes": sorted(splits.novel_classes),
            'protocol': splits.protocol, 'dataset_fingerprint': splits.dataset_fingerprint,
            'grouping': splits.grouping,
            'open_set_score': calibration.ood['method'],
            'conformal_score': calibration.conformal_score,
            'confidence_method': calibration.risk_control.get('confidence_method',cfg.confidence_method),
            'risk_calibration': calibration.risk_control.get('fit_split',cfg.risk_calibration),
            'decision_rule': calibration.decision_rule,
        },
        warnings=warnings + list(calibration.warnings),
        novel_evaluation=evaluate_enrollment(encoder, bank, calibration, splits),
        predictions=prediction_rows,
    )
    if save_bank_to:
        bank.save(save_bank_to)
    if save_calibration_to:
        calibration.save(save_calibration_to)
    return output, bank, calibration
