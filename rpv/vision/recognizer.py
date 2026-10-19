"""Nhận diện: ảnh -> top-k ứng viên + quyết định chấp nhận/từ chối/mơ hồ.

Khi nhiều nhãn cùng phù hợp, kết quả giữ lại tập ứng viên và yêu cầu nhân
viên xác nhận tại quầy. Các trường hợp chưa đủ độ tin cậy không được ép
thành một nhãn duy nhất để xử lý tự động.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Sequence

import numpy as np

from rpv import config as app_config
from rpv.vision.calibration import CalibrationModel
from rpv.vision.encoders import BaseEncoder, image_sha256
from rpv.vision.prototypes import PrototypeBank

# Trạng thái nhận diện thuần thị giác (chưa xét nguồn gốc).
STATUS_MATCH = "match"            # đúng 1 ứng viên trong tập conformal
STATUS_AMBIGUOUS = "ambiguous"    # nhiều ứng viên hợp lệ -> cần người chọn
STATUS_UNKNOWN = "unknown"        # tập rỗng -> ngoài danh mục
STATUS_EMPTY_CATALOG = "empty_catalog"


def validate_distances(bank, calibration, distances):
    if calibration and calibration.bank_fingerprint and calibration.bank_fingerprint != bank.fingerprint():
        raise RuntimeError('Danh mục đã đổi sau khi hiệu chỉnh; không được dùng ngưỡng cũ.')
    if set(distances) != set(bank.sku_ids):
        raise ValueError('Khoảng cách phải chứa đúng toàn bộ danh mục, kể cả khi đầu vào rỗng.')
    if distances:
        values = np.asarray(list(distances.values()), dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Khoảng cách phải hữu hạn, không âm và chứa đúng toàn bộ danh mục.')


@dataclass
class Candidate:
    sku_id: str
    label: str
    distance: float
    normalized_score: float   # distance / ngưỡng của SKU: <=1 là trong bán kính
    in_prediction_set: bool
    p_value: float = float("nan")   # p-value conformal (nan khi chưa hiệu chỉnh)

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class RecognitionResult:
    status: str
    candidates: list[Candidate]
    prediction_set: list[str]
    image_sha256: str = ""
    encoder_tag: str = ""
    alpha: float = 0.0
    top1: Candidate | None = None
    note: str = ""
    ood_score: float | None = None
    ood_threshold: float | None = None
    rejection_reason: str = ""
    conformal_prediction_set: list[str] = field(default_factory=list)
    decision_rule: str = 'conformal'
    score_semantics: str = 'euclidean_distance'

    @property
    def is_confident(self) -> bool:
        return self.status == STATUS_MATCH

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "prediction_set": self.prediction_set,
            "prediction_set_size": len(self.prediction_set),
            "candidates": [c.to_json() for c in self.candidates],
            "top1": self.top1.to_json() if self.top1 else None,
            "image_sha256": self.image_sha256,
            "encoder_tag": self.encoder_tag,
            "alpha": self.alpha,
            "note": self.note,
            "ood_score": self.ood_score,
            "ood_threshold": self.ood_threshold,
            "rejection_reason": self.rejection_reason,
            'conformal_prediction_set': self.conformal_prediction_set,
            'decision_rule': self.decision_rule,
            'score_semantics': self.score_semantics,
        }


class Recognizer:
    """Kết hợp encoder + bank + calibration thành một API gọn."""

    def __init__(self, encoder: BaseEncoder, bank: PrototypeBank,
                 calibration: CalibrationModel | None = None, top_k: int | None = None) -> None:
        self.encoder = encoder
        self.bank = bank
        self.calibration = calibration
        self.top_k = app_config.settings.vision.top_k if top_k is None else top_k
        if self.top_k < 0:
            raise ValueError('top_k must be nonnegative; 0 shows all candidates.')
        if calibration is not None and calibration.encoder_tag != encoder.tag:
            raise RuntimeError(
                f"Calibration thuộc encoder {calibration.encoder_tag!r} nhưng recognizer đang "
                f"dùng {encoder.tag!r}. Từ chối chạy để tránh quyết định dựa trên ngưỡng sai."
            )
        if calibration and calibration.bank_fingerprint and calibration.bank_fingerprint != bank.fingerprint():
            raise RuntimeError('Prototype bank đã thay đổi; cần calibrate lại trước khi nhận diện.')

    # -- lõi ---------------------------------------------------------------
    def recognize_embedding(self, embedding: np.ndarray, image_hash: str = "") -> RecognitionResult:
        with self.bank._lock:
            return self.recognize_distances(self.bank.distances(embedding), image_hash)

    def recognize_distances(self, dists: dict[str, float], image_hash: str = "") -> RecognitionResult:
        with self.bank._lock:
            validate_distances(self.bank, self.calibration, dists)
            return self._recognize_distances(dists, image_hash)

    def _recognize_distances(self, dists: dict[str, float], image_hash: str = "") -> RecognitionResult:
        if not dists:
            return RecognitionResult(
                status=STATUS_EMPTY_CATALOG, candidates=[], prediction_set=[],
                image_sha256=image_hash, encoder_tag=self.encoder.tag,
                note="Danh mục thị giác đang rỗng: chưa đăng ký SKU nào.",
            )
        alpha = self.calibration.alpha if self.calibration else 0.0
        pred_set: list[str] = []
        cands: list[Candidate] = []
        scores = self.calibration.nonconformity(dists) if self.calibration else dists
        for sku, dist in sorted(dists.items(), key=lambda kv: (kv[1], kv[0])):
            if self.calibration is None:
                inside, thr, pval = True, float("inf"), float("nan")
            else:
                thr = self.calibration.quantile_for(sku)
                inside = scores[sku] <= thr
                pval = self.calibration.p_value(sku, scores[sku])
            if inside:
                pred_set.append(sku)
            cands.append(Candidate(
                sku_id=sku,
                label=self.bank.entries[sku].label or sku,
                distance=float(dist),
                normalized_score=float(scores[sku] / thr) if math.isfinite(thr) and thr > 0 else float("nan"),
                in_prediction_set=inside,
                p_value=float(pval),
            ))
        conformal_set = list(pred_set)
        selective = bool(self.calibration and self.calibration.decision_rule == 'prototype_selective')
        tied = False
        if selective:
            ordered = sorted(dists, key=lambda s: (dists[s], s))
            tied = len(ordered) > 1 and abs(dists[ordered[0]] - dists[ordered[1]]) <= 1e-8
            pred_set = [] if tied else ordered[:1]
            for candidate in cands:
                candidate.in_prediction_set = candidate.sku_id in pred_set
        rejected_ood = bool(self.calibration and self.calibration.is_ood(dists))
        rejected_risk = bool(self.calibration and (not self.calibration.risk_accepts(dists)
                             or (selective and not self.calibration.risk_control.get('enabled'))))
        if rejected_ood:
            pred_set = []
            for candidate in cands:
                candidate.in_prediction_set = False
        # Keep the nearest top_k plus every accepted SKU, in distance order.
        if self.top_k:
            cands = [c for i, c in enumerate(cands) if i < self.top_k or c.in_prediction_set]
        if self.calibration is None:
            status, note = STATUS_AMBIGUOUS, 'Chưa hiệu chỉnh ngưỡng; cần calibrate trước khi tự động nhận diện.'
        elif tied:
            status, note = STATUS_AMBIGUOUS, 'Hai prototype gần nhất đồng hạng; cần nhân viên xác nhận.'
        elif not pred_set:
            status, note = STATUS_UNKNOWN, "Không SKU nào nằm trong bán kính chấp nhận đã hiệu chỉnh."
        elif rejected_risk:
            status, note = STATUS_AMBIGUOUS, 'Chưa đạt ngưỡng rủi ro đã hiệu chỉnh; cần nhân viên xác nhận.'
        elif len(pred_set) == 1:
            if self.bank.entries[pred_set[0]].requires_operator:
                status, note = STATUS_AMBIGUOUS, 'Mặt hàng đăng ký có hướng dẫn; cần nhân viên xác nhận.'
            else:
                status, note = STATUS_MATCH, ""
        else:
            status = STATUS_AMBIGUOUS
            note = f"{len(pred_set)} SKU cùng hợp lệ ở mức rủi ro alpha={alpha:g}; cần người xác nhận."
        return RecognitionResult(
            status=status, candidates=cands, prediction_set=pred_set,
            image_sha256=image_hash, encoder_tag=self.encoder.tag, alpha=alpha,
            top1=cands[0] if cands else None, note=note,
            ood_score=self.calibration.ood_score(dists) if self.calibration and self.calibration.ood else None,
            ood_threshold=self.calibration.ood.get('threshold') if self.calibration else None,
            rejection_reason=('ood' if rejected_ood else 'tie' if tied else
                              'conformal' if self.calibration and not pred_set else
                              'risk' if rejected_risk else 'operator_required' if
                              status == STATUS_AMBIGUOUS and len(pred_set) == 1 and
                              self.bank.entries[pred_set[0]].requires_operator else ''),
            conformal_prediction_set=conformal_set,
            decision_rule=self.calibration.decision_rule if self.calibration else 'conformal',
            score_semantics='euclidean_distance',
        )

    def recognize(self, image_path: str | Path) -> RecognitionResult:
        emb = self.encoder.embed_path(image_path)
        return self.recognize_embedding(emb, image_hash=image_sha256(image_path))

    def recognize_batch(self, image_paths: Sequence[str | Path]) -> list[RecognitionResult]:
        paths = list(image_paths)
        embs = self.encoder.embed_paths(paths)
        return [self.recognize_embedding(e, image_sha256(p)) for e, p in zip(embs, paths)]
