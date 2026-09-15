"""Nhận diện: ảnh -> top-k ứng viên + quyết định chấp nhận/từ chối/mơ hồ.

Đầu ra CÓ CHỦ ĐÍCH không phải một nhãn duy nhất. Ở quầy thanh toán, "tôi nghĩ
là A nhưng cũng có thể B, nhờ nhân viên xác nhận" là hành vi đúng và an toàn
hơn nhiều so với việc đoán chắc chắn một nhãn sai.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
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
        dists = self.bank.distances(embedding)
        return self.recognize_distances(dists, image_hash)

    def recognize_distances(self, dists: dict[str, float], image_hash: str = "") -> RecognitionResult:
        if not dists:
            return RecognitionResult(
                status=STATUS_EMPTY_CATALOG, candidates=[], prediction_set=[],
                image_sha256=image_hash, encoder_tag=self.encoder.tag,
                note="Danh mục thị giác đang rỗng: chưa đăng ký SKU nào.",
            )
        alpha = self.calibration.alpha if self.calibration else 0.0
        pred_set: list[str] = []
        cands: list[Candidate] = []
        for sku, dist in sorted(dists.items(), key=lambda kv: (kv[1], kv[0])):
            if self.calibration is None:
                inside, thr, pval = True, float("inf"), float("nan")
            else:
                thr = self.calibration.quantile_for(sku)
                inside = dist <= thr
                pval = self.calibration.p_value(sku, dist)
            if inside:
                pred_set.append(sku)
            cands.append(Candidate(
                sku_id=sku,
                label=self.bank.entries[sku].label or sku,
                distance=float(dist),
                normalized_score=float(dist / thr) if math.isfinite(thr) and thr > 0 else float("nan"),
                in_prediction_set=inside,
                p_value=float(pval),
            ))
        # Keep the nearest top_k plus every accepted SKU, in distance order.
        if self.top_k:
            cands = [c for i, c in enumerate(cands) if i < self.top_k or c.in_prediction_set]
        if self.calibration is None:
            status, note = STATUS_AMBIGUOUS, 'Chưa hiệu chỉnh ngưỡng; cần calibrate trước khi tự động nhận diện.'
        elif not pred_set:
            status, note = STATUS_UNKNOWN, "Không SKU nào nằm trong bán kính chấp nhận đã hiệu chỉnh."
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
        )

    def recognize(self, image_path: str | Path) -> RecognitionResult:
        emb = self.encoder.embed_path(image_path)
        return self.recognize_embedding(emb, image_hash=image_sha256(image_path))

    def recognize_batch(self, image_paths: Sequence[str | Path]) -> list[RecognitionResult]:
        paths = list(image_paths)
        embs = self.encoder.embed_paths(paths)
        return [self.recognize_embedding(e, image_sha256(p)) for e, p in zip(embs, paths)]
