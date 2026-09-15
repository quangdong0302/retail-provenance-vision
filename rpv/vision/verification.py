"""Kiểm chứng: ảnh + SKU được KHAI BÁO -> khớp hay không khớp.

Đây là bài toán khác với nhận diện, và là bài toán đúng của quầy tự thanh toán:
người mua (hoặc POS) đã khai "đây là xoài cát Hòa Lộc, 45.000đ/kg"; việc của hệ
thống là xác nhận hoặc phản đối lời khai đó. Nó bắt được cả nhầm lẫn thật thà
(chọn sai trong menu) và gian lận (khai hàng rẻ, lấy hàng đắt).

Ngưỡng dùng ở đây là ngưỡng của CHÍNH SKU được khai (class-conditional), không
phải ngưỡng toàn cục — nhờ vậy SKU dễ lẫn không bị "ăn theo" độ dễ của SKU khác.

``p_value`` trả về là p-value conformal đúng nghĩa, tính từ phân bố điểm calib
của lớp được khai. Bản trước trả về ``1 - d/threshold`` và vẫn gọi nó là
p-value: con số đó đơn điệu theo khoảng cách nên xếp hạng đúng, nhưng không có
phân phối đều dưới giả thuyết không, nên mọi câu như "chấp nhận ở mức ý nghĩa
5%" đều sai. Đây là loại sai dễ bị hội đồng bắt nhất vì nó nằm ở tên biến.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from rpv import config as app_config
from rpv.vision.calibration import CalibrationModel
from rpv.vision.encoders import BaseEncoder, image_sha256
from rpv.vision.prototypes import PrototypeBank

VERIFY_CONFIRMED = "confirmed"          # ảnh khớp SKU khai báo
VERIFY_MISMATCH = "mismatch"            # ảnh khớp SKU KHÁC -> nghi ngờ đổi mã
VERIFY_NOT_IN_CATALOG = "sku_unknown"   # SKU khai báo chưa đăng ký thị giác
VERIFY_OUT_OF_DISTRIBUTION = "out_of_distribution"  # không khớp gì cả


@dataclass
class VerificationResult:
    status: str
    claimed_sku: str
    claimed_distance: float
    best_sku: str
    best_distance: float
    threshold: float
    p_value: float          # p-value conformal cho giả thuyết "ảnh thuộc SKU khai báo"
    margin: float           # claimed_distance - best_distance; >0 nghĩa là có SKU khác gần hơn
    image_sha256: str = ""
    note: str = ""
    # Số mẫu calib của lớp được khai: p-value dựa trên ít mẫu thì thô, và người
    # đọc báo cáo có quyền biết điều đó.
    n_calib: int = 0
    best_p_value: float = float("nan")
    # True khi lời khai vẫn trong bán kính nhưng SKU khác gần hơn quá ngưỡng
    # nghiệp vụ -> policy phải gọi người, không được cho qua im lặng.
    suspicious_margin: bool = False

    @property
    def accepted(self) -> bool:
        return self.status == VERIFY_CONFIRMED

    def to_json(self) -> dict:
        return asdict(self)


class Verifier:
    def __init__(self, encoder: BaseEncoder, bank: PrototypeBank, calibration: CalibrationModel) -> None:
        if calibration.encoder_tag != encoder.tag:
            raise RuntimeError("Calibration và encoder không cùng phiên bản; từ chối kiểm chứng.")
        if calibration.bank_fingerprint and calibration.bank_fingerprint != bank.fingerprint():
            raise RuntimeError('Prototype bank đã thay đổi; cần calibrate lại trước khi kiểm chứng.')
        self.encoder = encoder
        self.bank = bank
        self.calibration = calibration
        self.margin_assist = float(app_config.settings.policy.verify_margin_assist)

    def verify_embedding(self, embedding: np.ndarray, claimed_sku: str, image_hash: str = "") -> VerificationResult:
        dists = self.bank.distances(embedding)
        return self.verify_distances(dists, claimed_sku, image_hash)

    def verify_distances(self, dists: dict[str, float], claimed_sku: str, image_hash: str = "") -> VerificationResult:
        if not dists:
            return VerificationResult(
                status=VERIFY_NOT_IN_CATALOG, claimed_sku=claimed_sku, claimed_distance=float("inf"),
                best_sku="", best_distance=float("inf"), threshold=float("inf"), p_value=0.0, margin=0.0,
                image_sha256=image_hash, note="Danh mục thị giác rỗng.",
            )
        best_sku = min(dists, key=dists.get)
        best_dist = float(dists[best_sku])
        if claimed_sku not in dists:
            return VerificationResult(
                status=VERIFY_NOT_IN_CATALOG, claimed_sku=claimed_sku, claimed_distance=float("inf"),
                best_sku=best_sku, best_distance=best_dist,
                threshold=self.calibration.threshold_for(claimed_sku), p_value=0.0, margin=0.0,
                image_sha256=image_hash,
                note=f"SKU khai báo {claimed_sku!r} chưa có prototype thị giác; không thể kiểm chứng bằng ảnh.",
            )
        claimed_dist = float(dists[claimed_sku])
        thr = self.calibration.quantile_for(claimed_sku)
        p_value = self.calibration.p_value(claimed_sku, claimed_dist)
        best_p = self.calibration.p_value(best_sku, best_dist)
        n_calib = int(self.calibration.n_calib_per_class.get(claimed_sku, 0))
        margin = claimed_dist - best_dist
        suspicious = False

        if claimed_dist <= thr and best_sku == claimed_sku:
            status, note = VERIFY_CONFIRMED, ""
        elif claimed_dist <= thr:
            status = VERIFY_CONFIRMED
            suspicious = margin > self.margin_assist
            note = (f"Khớp SKU khai báo nhưng {best_sku!r} còn gần hơn (margin={margin:.4f}"
                    + (f" > ngưỡng {self.margin_assist:.2f}, cần người soát"
                       if suspicious else ", trong mức chấp nhận") + ").")
        elif best_dist <= self.calibration.quantile_for(best_sku):
            status = VERIFY_MISMATCH
            note = (f"Ảnh không khớp {claimed_sku!r} nhưng khớp {best_sku!r}. "
                    "Đây là dạng nghi vấn đổi mã/chọn sai mặt hàng.")
        else:
            status = VERIFY_OUT_OF_DISTRIBUTION
            note = "Ảnh không khớp SKU khai báo và cũng không khớp SKU nào trong danh mục."
        return VerificationResult(
            status=status, claimed_sku=claimed_sku, claimed_distance=claimed_dist,
            best_sku=best_sku, best_distance=best_dist, threshold=float(thr),
            p_value=float(p_value), margin=float(margin), image_sha256=image_hash, note=note,
            n_calib=n_calib, best_p_value=float(best_p), suspicious_margin=suspicious,
        )

    def verify(self, image_path: str | Path, claimed_sku: str) -> VerificationResult:
        emb = self.encoder.embed_path(image_path)
        return self.verify_embedding(emb, claimed_sku, image_hash=image_sha256(image_path))
