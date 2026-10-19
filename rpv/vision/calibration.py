"""Hiệu chỉnh rủi ro: khi nào được trả lời, khi nào phải nói "không biết".

Đây là phần mang tính học thuật nặng nhất của hệ thống và cũng là điểm khác biệt
so với một classifier thông thường. Ba cơ chế, dùng cùng lúc:

1. **Ngưỡng open-set theo lớp (Mondrian / class-conditional).** Mỗi SKU có
   "bán kính chấp nhận" riêng, lấy theo phân vị khoảng cách nội-lớp trên tập
   calib. Một ngưỡng toàn cục luôn bất công: SKU bao bì đặc thù thì quá lỏng,
   SKU dễ lẫn (mấy chục giống táo) thì quá chặt. SKU có quá ít mẫu calib sẽ
   dùng ngưỡng toàn cục thay vì "học" từ 2-3 điểm dữ liệu.

2. **Split conformal prediction.** Với mức rủi ro alpha, trả về TẬP dự đoán
   (prediction set) thay vì một nhãn, kèm bảo đảm coverage xấp xỉ 1-alpha trên
   dữ liệu i.i.d. Tập rỗng = hàng lạ; tập nhiều phần tử = mơ hồ, cần người xác
   nhận. Đây là ngôn ngữ đúng để nói về rủi ro ở quầy thanh toán, thay cho
   confidence softmax vốn nổi tiếng overconfident với input lạ.

3. **Ngưỡng toàn cục theo ràng buộc nghiệp vụ.** Chọn ngưỡng nhỏ nhất trên
   known-calib đạt known-recall mục tiêu; đánh giá hàng lạ riêng trên test.

Artifact mới còn lưu Energy OOD và gate selective risk fit trên known-calib.
Chỉ phần tập conformal gốc có diễn giải coverage conformal; các gate kết hợp
phải báo risk/coverage thực tế, không tự kế thừa bảo đảm đó.

Mô hình hiệu chỉnh còn lưu lại PHÂN BỐ điểm nonconformity trên tập calib của
từng lớp. Nhờ vậy lúc kiểm chứng tính được p-value conformal đúng nghĩa
``(1 + #{s_calib >= s_obs}) / (n + 1)`` thay vì một tỉ số "gần giống p-value".
Giá trị này biểu diễn hạng của score so với tập hiệu chỉnh, không phải xác
suất nhãn sản phẩm đúng. Các giới hạn khi dùng score gộp nằm trong giao thức v2.

Quy tắc bất di bất dịch: ngưỡng và phân vị được CHỌN trên tập calib, và số liệu
báo cáo phải đo trên tập test rời hẳn. Chọn ngưỡng rồi báo cáo trên cùng tập
đó là rò rỉ dữ liệu và luôn cho số liệu đẹp hơn thực tế.
"""
from __future__ import annotations

import json
import os
import threading
from bisect import bisect_left
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from rpv import config as app_config
from rpv.util import json_dumps
from rpv.vision.rejection import energy_score

def _digest_scores(scores: Sequence[float]) -> list[float]:
    """Keep all scores at full precision; compression changes conformal ranks."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    if not np.isfinite(s).all():
        raise ValueError('Calibration scores phải hữu hạn.')
    return s.tolist()


@dataclass
class CalibrationModel:
    """Kết quả hiệu chỉnh, đủ để tái lập quyết định ở inference."""
    encoder_tag: str
    alpha: float
    global_threshold: float
    per_class_threshold: dict[str, float] = field(default_factory=dict)
    conformal_quantile: dict[str, float] = field(default_factory=dict)
    global_conformal_quantile: float = 0.0
    n_calib_per_class: dict[str, int] = field(default_factory=dict)
    calib_metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    fallback_classes: list[str] = field(default_factory=list)
    #: Phân bố điểm nonconformity trên calib (đã sort) để tính p-value conformal.
    calib_scores: dict[str, list[float]] = field(default_factory=dict)
    global_calib_scores: list[float] = field(default_factory=list)
    bank_fingerprint: str = ""
    ood: dict = field(default_factory=dict)
    risk_control: dict = field(default_factory=dict)
    conformal_score: str = 'distance'
    development_fingerprint: str = ''
    decision_rule: str = 'conformal'

    def nonconformity(self, distances: dict[str, float]) -> dict[str, float]:
        if self.conformal_score == 'distance':
            return dict(distances)
        if self.conformal_score != 'relative':
            raise ValueError('Unsupported conformal score.')
        if len(distances) < 2:
            raise ValueError('Relative conformal requires at least two catalog classes.')
        return {s: float(np.float32(d) / max(np.float32(min(v for k,v in distances.items() if k != s)), np.float32(1e-9)))
                for s,d in distances.items()}

    def confidence(self, distances: dict[str, float]) -> float:
        method = self.risk_control.get('confidence_method', 'energy')
        if method == 'energy':
            return -self.ood_score(distances)
        if method == 'nearest_ratio':
            values = sorted(distances.values())
            if len(values) < 2:
                raise ValueError('Nearest-ratio requires at least two classes.')
            # Preserve float32 arithmetic used by the frozen research recipe.
            return float(np.float32(1) - np.float32(values[0]) / max(np.float32(values[1]), np.float32(1e-9)))
        raise ValueError('Unsupported selective confidence method.')

    def ood_score(self, distances: dict[str, float]) -> float:
        if self.ood.get('method') == 'energy':
            return energy_score(distances, self.ood['temperature'])
        return float(min(distances.values()))

    def is_ood(self, distances: dict[str, float]) -> bool:
        return bool(self.ood and self.ood_score(distances) > self.ood['threshold'])

    def risk_accepts(self, distances: dict[str, float]) -> bool:
        if not self.risk_control.get('enabled'):
            return True
        threshold = self.risk_control.get('confidence_threshold')
        return threshold is not None and self.confidence(distances) >= threshold

    def threshold_for(self, sku_id: str) -> float:
        """Ngưỡng "bán kính chấp nhận" theo phân vị 1-alpha của lớp.

        Dùng để BÁO CÁO và để chẩn đoán. Quyết định chấp nhận/từ chối ở
        inference dùng ``quantile_for`` (phân vị conformal hữu hạn mẫu).
        """
        return float(self.per_class_threshold.get(sku_id, self.global_threshold))

    def quantile_for(self, sku_id: str) -> float:
        """Bán kính conformal dùng cho quyết định. Lớp thiếu mẫu calib tự động
        rơi về phân vị toàn cục."""
        return float(self.conformal_quantile.get(sku_id, self.global_conformal_quantile))

    def scores_for(self, sku_id: str) -> list[float]:
        got = self.calib_scores.get(sku_id)
        return got if got else self.global_calib_scores

    def p_value(self, sku_id: str, score: float) -> float:
        """p-value conformal cho giả thuyết "ảnh này thuộc ``sku_id``".

        Dùng ``(1 + #{s_calib >= s_obs}) / (n + 1)`` trên score đã sắp xếp.
        Lớp thiếu score riêng dùng tập gộp; không suy ra bảo đảm theo lớp từ
        trường hợp fallback đó. p-value không phải xác suất nhãn đúng.
        """
        ref = self.scores_for(sku_id)
        if not ref or not np.isfinite(score):
            return float("nan")
        return (1.0 + len(ref) - bisect_left(ref, float(score))) / (len(ref) + 1.0)

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: Mapping) -> "CalibrationModel":
        """Bỏ qua khoá lạ để file calibration của phiên bản sau vẫn đọc được."""
        known = {f.name for f in dataclass_fields(CalibrationModel)}
        data = {k: v for k, v in dict(d).items() if k in known}
        for key in ('global_threshold', 'global_conformal_quantile'):
            if key in data and data[key] is None:
                data[key] = float('inf')
        for key in ('per_class_threshold', 'conformal_quantile'):
            if key in data:
                data[key] = {k: float('inf') if v is None else v for k, v in data[key].items()}
        return CalibrationModel(**data)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
        try:
            tmp.write_text(json_dumps(self.to_json()), encoding='utf-8')
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return path

    @staticmethod
    def load(path: str | Path, encoder_tag: str | None = None) -> "CalibrationModel":
        model = CalibrationModel.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
        if model.conformal_score not in {'distance', 'relative'}:
            raise ValueError('Unsupported conformal score in calibration artifact.')
        if model.risk_control.get('confidence_method', 'energy') not in {'energy', 'nearest_ratio'}:
            raise ValueError('Unsupported selective confidence in calibration artifact.')
        if model.decision_rule not in {'conformal', 'prototype_selective'}:
            raise ValueError('Unsupported decision rule.')
        if not isinstance(model.encoder_tag, str) or not model.encoder_tag or not isinstance(model.bank_fingerprint, str):
            raise ValueError('Calibration có định danh encoder/bank không hợp lệ.')
        if not isinstance(model.alpha, (int, float)) or not np.isfinite(model.alpha) or not 0 < model.alpha < 1:
            raise ValueError('Calibration alpha phải hữu hạn, trong (0,1).')
        radii = [model.global_threshold, model.global_conformal_quantile,
                 *model.per_class_threshold.values(), *model.conformal_quantile.values()]
        if any(not isinstance(v, (int, float)) or np.isnan(v) or v < 0 for v in radii):
            raise ValueError('Calibration có bán kính không hợp lệ; không được dùng ngưỡng NaN/âm.')
        for scores in [model.global_calib_scores, *model.calib_scores.values()]:
            values = np.asarray(scores, dtype=float)
            if (values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0)
                    or np.any(values[1:] < values[:-1])):
                raise ValueError('Phân bố score calibration phải hữu hạn, không âm và đã sắp xếp.')
        if model.ood:
            if model.ood.get('method') not in {'energy', 'distance'}:
                raise ValueError('Calibration chứa phương pháp OOD không được hỗ trợ.')
            threshold = model.ood.get('threshold')
            if not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
                raise ValueError('Ngưỡng OOD phải hữu hạn.')
            temperature = model.ood.get('temperature', .05)
            if not isinstance(temperature, (int, float)) or not np.isfinite(temperature) or temperature <= 0:
                raise ValueError('Energy temperature phải hữu hạn và > 0.')
        if model.risk_control:
            enabled = model.risk_control.get('enabled')
            threshold = model.risk_control.get('confidence_threshold')
            if type(enabled) is not bool or (threshold is not None and
                    (not isinstance(threshold, (int, float)) or not np.isfinite(threshold))):
                raise ValueError('Risk gate có trạng thái hoặc ngưỡng không hợp lệ.')
        if encoder_tag is not None and model.encoder_tag != encoder_tag:
            raise RuntimeError(
                f"Calibration tại '{path}' thuộc encoder {model.encoder_tag!r}, "
                f"không dùng được với encoder {encoder_tag!r}. Hãy chạy lại bước calibrate."
            )
        return model


def _conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Phân vị conformal hữu hạn mẫu: ceil((n+1)(1-alpha))/n.

    Hệ số (n+1) không phải chi tiết trang trí: đây chính là phần bảo đảm
    coverage hữu hạn mẫu của split conformal. Bỏ nó đi là mất tính bảo đảm.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    if not 0 < alpha < 1 or not np.isfinite(s).all():
        raise ValueError('Cần 0 < alpha < 1 và score hữu hạn.')
    n = s.size
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[rank - 1]) if rank <= n else float('inf')


def calibrate_global_threshold(known: Sequence[float], unknown: Sequence[float],
                               target_known_recall: float,
                               min_useful_unknown_reject: float = 0.5) -> dict:
    """Choose the smallest threshold reaching the target known recall.

    Optional unknown scores provide rejection diagnostics, not threshold tuning.
    The thesis pipeline fits with known-calib only and measures unknown on test.
    """
    k = np.asarray(known, dtype=np.float64)
    u = np.asarray(unknown, dtype=np.float64)
    if k.size == 0:
        raise ValueError("Không có mẫu known nào để hiệu chỉnh ngưỡng.")
    if not 0 < target_known_recall <= 1 or not np.isfinite(k).all() or not np.isfinite(u).all():
        raise ValueError('known_recall phải trong (0,1], mọi khoảng cách phải hữu hạn.')

    warnings: list[str] = []
    cands = np.unique(np.concatenate([k, u])) if u.size else np.unique(k)
    recalls = np.searchsorted(np.sort(k), cands, side='right') / k.size
    eligible = np.flatnonzero(recalls >= target_known_recall)
    chosen = int(eligible[0])
    t = float(cands[chosen])
    best = {'threshold': t, 'known_recall': float(recalls[chosen]),
            'unknown_reject_rate': float(np.mean(u > t)) if u.size else 0.0}

    if u.size == 0:
        warnings.append(
            "Ngưỡng chỉ được chọn từ known-calib. Khả năng từ chối hàng ngoài danh mục "
            "phải được đánh giá trên unknown-test độc lập; không đưa unknown-test vào hiệu chỉnh."
        )
    elif best["unknown_reject_rate"] < min_useful_unknown_reject:
        warnings.append(
            f"Ngưỡng đạt known_recall {best['known_recall']:.2%} nhưng chỉ chặn được "
            f"{best['unknown_reject_rate']:.2%} hàng lạ (dưới mức hữu dụng "
            f"{min_useful_unknown_reject:.0%}). Đây là dấu hiệu embedding chưa đủ tách biệt — "
            "cần encoder mạnh hơn hoặc thêm dữ liệu, KHÔNG nên hạ ràng buộc nghiệp vụ."
        )
    if warnings:
        best["warnings"] = warnings
    return best


class Calibrator:
    """Fit ``CalibrationModel`` từ khoảng cách trên tập calib."""

    def __init__(self, encoder_tag: str, alpha: float | None = None,
                 target_known_recall: float | None = None,
                 min_calib_per_class: int | None = None) -> None:
        cfg = app_config.settings.calibration
        self.encoder_tag = encoder_tag
        self.alpha = cfg.alpha if alpha is None else alpha
        self.target_known_recall = cfg.target_known_recall if target_known_recall is None else target_known_recall
        self.min_calib_per_class = cfg.min_calib_per_class if min_calib_per_class is None else min_calib_per_class

    def fit(self,
            true_class_distances: Mapping[str, Sequence[float]],
            unknown_distances: Sequence[float] = (),
            known_detection_distances: Sequence[float] | None = None,
            ) -> CalibrationModel:
        """``true_class_distances``: sku_id -> khoảng cách của các ảnh calib THUỘC
        sku đó tới prototype của CHÍNH nó (nonconformity score).
        ``unknown_distances``: khoảng cách tới prototype gần nhất của các ảnh
        hàng lạ (không thuộc danh mục)."""
        known_flat = [d for ds in true_class_distances.values() for d in ds]
        glob = calibrate_global_threshold(known_flat if known_detection_distances is None else known_detection_distances,
                                          unknown_distances, self.target_known_recall)
        warnings: list[str] = list(glob.get("warnings", []))

        per_class: dict[str, float] = {}
        quantiles: dict[str, float] = {}
        counts: dict[str, int] = {}
        scores: dict[str, list[float]] = {}
        fallback: list[str] = []
        global_q = _conformal_quantile(known_flat, self.alpha)

        for sku, ds in sorted(true_class_distances.items()):
            ds = [float(x) for x in ds]
            counts[sku] = len(ds)
            if len(ds) >= self.min_calib_per_class:
                per_class[sku] = float(np.quantile(ds, 1 - self.alpha, method="higher"))
                quantiles[sku] = _conformal_quantile(ds, self.alpha)
                scores[sku] = _digest_scores(ds)
            else:
                fallback.append(sku)
        if fallback:
            warnings.append(
                f"{len(fallback)} SKU có dưới {self.min_calib_per_class} mẫu calib nên dùng ngưỡng "
                "toàn cục thay vì ngưỡng riêng (tránh ngưỡng học từ quá ít dữ liệu)."
            )

        return CalibrationModel(
            encoder_tag=self.encoder_tag,
            alpha=self.alpha,
            global_threshold=float(glob["threshold"]),
            per_class_threshold=per_class,
            conformal_quantile=quantiles,
            global_conformal_quantile=float(global_q),
            n_calib_per_class=counts,
            calib_metrics={
                "known_recall_on_calib": glob["known_recall"],
                "unknown_reject_rate_on_calib": glob["unknown_reject_rate"],
                "target_known_recall": self.target_known_recall,
                "n_known_calib": len(known_flat),
                "n_unknown_calib": len(unknown_distances),
            },
            warnings=warnings,
            fallback_classes=fallback,
            calib_scores=scores,
            global_calib_scores=_digest_scores(known_flat),
        )
