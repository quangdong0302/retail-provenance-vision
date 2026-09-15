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
                "Ở protocol v2, unknown không tham gia học/chọn head hoặc hiệu chỉnh. "
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
    true_distances, nearest = {}, []
    for label, paths in splits.calib.items():
        distances, labels = bank.distances_batch(encoder.embed_paths(paths))
        true_distances[label] = distances[:, labels.index(label)].tolist()
        nearest.extend(distances.min(axis=1).tolist())
    calibration = Calibrator(encoder.tag, alpha=alpha).fit(
        true_distances, known_detection_distances=nearest)
    calibration.bank_fingerprint = bank.fingerprint()
    return bank, calibration


def evaluate_enrollment(encoder, bank, calibration, splits, shots=(1, 3, 5)):
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
            entry = enrolled.register(label, paths[:k], label=label)
            if entry.n_shots != k:
                raise RuntimeError(f'{label}: không đọc đủ {k} ảnh novel support.')
        transferred = replace(calibration, bank_fingerprint=enrolled.fingerprint())
        recognizer = Recognizer(encoder, enrolled, transferred)
        truth, predicted, covered = [], [], []
        for label, vectors in queries.items():
            distances, labels = enrolled.distances_batch(vectors)
            for row in distances:
                result = recognizer.recognize_distances(dict(zip(labels, map(float, row))))
                truth.append(label)
                predicted.append(labels[int(np.argmin(row))])
                covered.append(label in result.prediction_set)
        results.append({'k_shot': k, 'n_query': len(truth), 'n_catalog_classes': len(enrolled),
                        'accuracy': closed_set_accuracy(truth, predicted),
                        'macro_f1': macro_f1(truth, predicted),
                        'empirical_coverage_transferred_threshold': float(np.mean(covered))})
    return {'protocol': 'Known catalog + enrolled unknown classes; fixed query images across k.',
            'threshold_note': 'Pooled known calibration transferred to novel classes; no class-conditional guarantee for novel.',
            'runs': results}


def evaluate_system(data_dir: str | Path | None = None,
                    encoder: BaseEncoder | None = None,
                    k_shot: int | None = None,
                    alpha: float | None = None,
                    splits: DatasetSplits | None = None,
                    save_bank_to: str | Path | None = None,
                    save_calibration_to: str | Path | None = None,
                    verbose: bool = True) -> tuple[EvaluationOutput, PrototypeBank, CalibrationModel]:
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
    bank, calibration = fit_catalog(encoder, splits, alpha)

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
    correct_flags, confidences = [], []
    status_counter: dict[str, int] = {}
    ver_records: list[dict] = []
    hard_records: list[dict] = []
    covered, singleton_correct, prediction_rows = [], [], []
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
            covered.append(cname in res.prediction_set)
            if res.status == STATUS_MATCH:
                singleton_correct.append(res.prediction_set[0] == cname)
            prediction_rows.append({'path': splits.test[cname][row_i], 'truth': cname,
                                    'top1': pred, 'status': res.status,
                                    'prediction_set': res.prediction_set,
                                    'nearest_distance': float(row.min())})
            status_counter[res.status] = status_counter.get(res.status, 0) + 1
            pred_set_sizes.append(len(res.prediction_set))
            y_true_open.append(cname)
            if res.status == STATUS_UNKNOWN or not res.prediction_set:
                y_pred_open.append(UNKNOWN_LABEL)
            else:
                y_pred_open.append(res.prediction_set[0] if res.status == STATUS_MATCH else pred)
            # Độ tin cậy cho risk-coverage = khoảng cách tới prototype gần nhất,
            # CHUẨN HOÁ theo ngưỡng của chính lớp được dự đoán.
            #
            # This score is fixed by the protocol. Compare alternatives on
            # validation; do not choose confidence definitions using test errors.
            srt = np.sort(row)
            d1 = float(srt[0])
            thr = calibration.quantile_for(pred)
            conf = float(1.0 - d1 / thr) if thr > 0 else 0.0
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
    n_unknown_rejected = 0
    if unk_test.shape[0]:
        D, labels = bank.distances_batch(unk_test)
        unk_test_dists = [float(x) for x in D.min(axis=1)]
        for i in range(unk_test.shape[0]):
            res = recognizer.recognize_distances(dict(zip(labels, map(float, D[i]))))
            prediction_rows.append({'path': splits.unknown_test[i], 'truth': UNKNOWN_LABEL,
                                    'top1': labels[int(np.argmin(D[i]))], 'status': res.status,
                                    'prediction_set': res.prediction_set,
                                    'nearest_distance': float(D[i].min())})
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
        },
        closed_set={
            "accuracy_all_classes": closed_set_accuracy(y_true_closed, y_pred_closed),
            "macro_f1_all_classes": macro_f1(y_true_closed, y_pred_closed),
            "macro_f1_with_unknown": macro_f1(y_true_open, y_pred_open),
            "n_test_images": len(y_true_closed),
        },
        open_set=open_set_metrics(known_dists, unk_test_dists, calibration.global_threshold),
        selective={
            "status_distribution": status_counter,
            "mean_prediction_set_size": float(np.mean(pred_set_sizes)) if pred_set_sizes else float("nan"),
            "empirical_coverage": float(np.mean(covered)) if covered else float('nan'),
            "nonempty_set_rate": float(np.mean(np.asarray(pred_set_sizes) > 0)) if pred_set_sizes else float('nan'),
            "singleton_accuracy": float(np.mean(singleton_correct)) if singleton_correct else float('nan'),
            "n_singleton": len(singleton_correct),
            "target_coverage": 1.0 - calibration.alpha,
            "abstention_rate": (status_counter.get(STATUS_UNKNOWN, 0) + status_counter.get(STATUS_AMBIGUOUS, 0))
                               / max(1, len(y_true_closed)),
            "unknown_items_correctly_rejected": (n_unknown_rejected / unk_test.shape[0]) if unk_test.shape[0] else float("nan"),
            "risk_coverage_curve": curve,
            "coverage_at_risk_1pct": coverage_at_risk(curve, 0.01),
            "coverage_at_risk_5pct": coverage_at_risk(curve, 0.05),
            "coverage_at_risk_10pct": coverage_at_risk(curve, 0.10),
        },
        verification={**verification_metrics(ver_records),
                      'nearest_wrong_claim': verification_metrics(hard_records),
                      'scope': 'Verifier confirmed only; not final policy auto-allow.'},
        errors={
            "top_confusion_pairs": confusion_pairs(y_true_closed, y_pred_closed),
            "worst_per_class_recall": [{"sku": k, "recall": v} for k, v in worst],
        },
        config={
            "k_shot": k_shot, "alpha": alpha, "data_dir": str(data_dir),
            "centroids_per_sku": bank.centroids_per_sku,
            "class_split_seed": cfg.class_split_seed, "calib_test_seed": cfg.calib_test_seed,
            "use_tta": app_config.settings.vision.use_tta,
            "encoder": encoder.describe(),
            "novel_classes": sorted(splits.novel_classes),
            'protocol': splits.protocol, 'dataset_fingerprint': splits.dataset_fingerprint,
            'grouping': splits.grouping,
            'open_set_score': 'nearest_prototype_euclidean_for_both_known_and_unknown',
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
