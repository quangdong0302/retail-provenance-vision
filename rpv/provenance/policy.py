"""Policy engine — nơi thị giác và nguồn gốc được đối chiếu với nhau.

Đây là "bộ não nghiệp vụ" và là đóng góp trung tâm của đề tài: mô hình thị giác
không được phép tự quyết định bán hàng. Nó đưa ra bằng chứng thị giác; repository
đưa ra bằng chứng hồ sơ; policy engine hợp nhất hai nguồn thành một trong bốn
kết cục (allow / allow_with_warning / assist / block) kèm mã lý do.

Nguyên tắc thiết kế:
  * Thiếu bằng chứng KHÔNG bao giờ được suy diễn thành "có bằng chứng". Không có
    hồ sơ lô thì trạng thái là PROVENANCE_MISSING, không phải "xuất xứ Việt Nam".
  * An toàn thực phẩm và pháp lý luôn thắng trải nghiệm mua hàng: hàng thu hồi
    hoặc hết hạn thì BLOCK, kể cả khi thị giác rất tự tin.
  * Mọi kết cục đều giải thích được bằng danh sách mã lý do; không có "điểm tin
    cậy" mờ mịt nào được dùng làm lý do duy nhất.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Optional

from rpv import config as app_config
from rpv.domain.models import (
    DecisionOutcome, EvidenceStatus, LotStatus, ReasonCode,
)
from rpv.provenance.repository import ProvenanceRecord
from rpv.vision.recognizer import (
    RecognitionResult, STATUS_AMBIGUOUS, STATUS_MATCH, STATUS_UNKNOWN,
)
from rpv.vision.verification import (
    VerificationResult, VERIFY_CONFIRMED, VERIFY_MISMATCH, VERIFY_NOT_IN_CATALOG,
)

# Mức nghiêm trọng: dùng để hợp nhất nhiều tín hiệu — kết cục cuối là mức nặng nhất.
_SEVERITY = {
    DecisionOutcome.ALLOW: 0,
    DecisionOutcome.ALLOW_WITH_WARNING: 1,
    DecisionOutcome.ASSIST: 2,
    DecisionOutcome.BLOCK: 3,
}

PROVENANCE_OK = "provenance_ok"
PROVENANCE_MISSING = "provenance_missing"
PROVENANCE_INCOMPLETE = "provenance_incomplete"
PROVENANCE_BLOCKED = "provenance_blocked"


@dataclass
class PolicyDecision:
    outcome: str
    reasons: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    provenance_status: str = PROVENANCE_MISSING
    origin_summary: str = "Chưa xác minh"
    requires_operator: bool = False
    audit_notes: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


class PolicyEngine:
    def __init__(self, config=None) -> None:
        self.cfg = config or app_config.settings.policy

    # -- helper ------------------------------------------------------------
    def _required_evidence(self, record: ProvenanceRecord) -> tuple[str, ...]:
        """Chọn bộ chứng từ bắt buộc: hàng nhập khẩu chặt hơn hàng nội địa."""
        category = "default"
        if record.product and record.product.category:
            category = record.product.category
        if record.lot and record.lot.origin_country not in ("VN", "UNKNOWN", ""):
            category = "imported"
        req = self.cfg.required_evidence
        return req.get(category, req.get("default", ()))

    @staticmethod
    def _origin_text(record: ProvenanceRecord) -> str:
        lot = record.lot
        if lot is None:
            return "Chưa xác minh (không tìm thấy hồ sơ lô)"
        bits = [b for b in (lot.farm_or_facility, lot.origin_region, lot.origin_country) if b and b != "UNKNOWN"]
        if not bits:
            return "Chưa xác minh (hồ sơ lô thiếu thông tin xuất xứ)"
        sup = f" — NCC: {record.supplier.name}" if record.supplier else ""
        return " / ".join(bits) + sup

    # -- lõi ---------------------------------------------------------------
    def decide(self,
               recognition: Optional[RecognitionResult] = None,
               verification: Optional[VerificationResult] = None,
               record: Optional[ProvenanceRecord] = None,
               claimed_price: float | None = None,
               today: Optional[date] = None,
               ) -> PolicyDecision:
        today = today or date.today()
        reasons: list[str] = []
        messages: list[str] = []
        outcome = DecisionOutcome.ALLOW

        def escalate(new: DecisionOutcome) -> None:
            nonlocal outcome
            if _SEVERITY[new] > _SEVERITY[outcome]:
                outcome = new

        # ---------- 1. Tín hiệu thị giác ----------
        if verification is not None:
            if verification.status == VERIFY_CONFIRMED:
                reasons.append(ReasonCode.VISION_OK.value)
                if verification.note:
                    messages.append(verification.note)
                    # Lời khai nằm trong bán kính nhưng SKU khác gần hơn: cảnh
                    # báo nhẹ là đủ khi chênh lệch nhỏ, nhưng khi chênh lệch
                    # vượt ngưỡng nghiệp vụ thì bắt buộc gọi người — đó chính là
                    # hình dạng của gian lận "hợp lệ về ngưỡng".
                    if getattr(verification, "suspicious_margin", False):
                        reasons.append(ReasonCode.VISION_AMBIGUOUS.value)
                        escalate(DecisionOutcome.ASSIST)
                    elif verification.margin > 0:
                        escalate(DecisionOutcome.ALLOW_WITH_WARNING)
            elif verification.status == VERIFY_MISMATCH:
                reasons.append(ReasonCode.VISION_MISMATCH.value)
                messages.append(
                    f"Ảnh không khớp mặt hàng khai báo. Hệ thống thấy giống {verification.best_sku!r} hơn."
                )
                escalate(DecisionOutcome.BLOCK)
            elif verification.status == VERIFY_NOT_IN_CATALOG:
                reasons.append(ReasonCode.VISION_SKU_NOT_REGISTERED.value)
                messages.append("Mặt hàng khai báo chưa được đăng ký ảnh mẫu; chưa thể kiểm chứng bằng thị giác.")
                escalate(DecisionOutcome.ASSIST)
            else:  # VERIFY_OUT_OF_DISTRIBUTION
                reasons.append(ReasonCode.VISION_UNKNOWN.value)
                messages.append("Ảnh không khớp mặt hàng nào trong danh mục.")
                escalate(DecisionOutcome.ASSIST)
        elif recognition is not None:
            if recognition.status == STATUS_MATCH:
                reasons.append(ReasonCode.VISION_OK.value)
            elif recognition.status == STATUS_AMBIGUOUS:
                reasons.append(ReasonCode.VISION_AMBIGUOUS.value)
                names = ", ".join(recognition.prediction_set[:4])
                messages.append(f"Nhiều mặt hàng cùng khả năng: {names}. Cần người chọn đúng mặt hàng.")
                escalate(DecisionOutcome.ASSIST)
            elif recognition.status == STATUS_UNKNOWN:
                reasons.append(ReasonCode.VISION_UNKNOWN.value)
                messages.append("Không nhận diện được: mặt hàng có thể chưa đăng ký trong danh mục.")
                escalate(DecisionOutcome.ASSIST)
            else:
                reasons.append(ReasonCode.VISION_CATALOG_EMPTY.value)
                messages.append("Danh mục thị giác rỗng; chưa đăng ký mặt hàng nào.")
                escalate(DecisionOutcome.ASSIST)

        # ---------- 2. Tín hiệu nguồn gốc ----------
        provenance_status = PROVENANCE_MISSING
        if record is None or record.sku is None:
            reasons.append(ReasonCode.SKU_NOT_FOUND.value)
            messages.append("Không tìm thấy mặt hàng trong danh mục nghiệp vụ.")
            escalate(DecisionOutcome.ASSIST)
        elif record.lot is None:
            reasons.append(ReasonCode.PROVENANCE_MISSING.value)
            messages.append(
                "Không có hồ sơ lô cho mặt hàng này: không thể chứng minh nguồn gốc. "
                "Hệ thống KHÔNG suy đoán xuất xứ từ hình ảnh."
            )
            escalate(DecisionOutcome.ASSIST)
        else:
            lot = record.lot
            if record.sku and lot.sku_id != record.sku.sku_id:
                reasons.append(ReasonCode.LOT_SKU_CONFLICT.value)
                messages.append(f"Lô {lot.lot_id} không thuộc mặt hàng {record.sku.sku_id}.")
                escalate(DecisionOutcome.BLOCK)

            if lot.status == LotStatus.RECALLED.value:
                reasons.append(ReasonCode.LOT_RECALLED.value)
                messages.append(f"Lô {lot.lot_id} đang bị THU HỒI. Không được bán.")
                escalate(DecisionOutcome.BLOCK)
            elif lot.status == LotStatus.QUARANTINED.value:
                reasons.append(ReasonCode.LOT_QUARANTINED.value)
                messages.append(f"Lô {lot.lot_id} đang bị tạm giữ chờ kiểm tra.")
                escalate(DecisionOutcome.BLOCK)
            elif lot.status == LotStatus.SOLD_OUT.value:
                reasons.append(ReasonCode.LOT_SOLD_OUT.value)
                messages.append(f"Lô {lot.lot_id} đã bán hết. Không được bán tiếp từ lô này.")
                escalate(DecisionOutcome.BLOCK)
            elif lot.status not in {s.value for s in LotStatus}:
                reasons.append(ReasonCode.LOT_STATUS_INVALID.value)
                messages.append(f"Lô {lot.lot_id} có trạng thái không hợp lệ; cần kiểm tra hồ sơ.")
                escalate(DecisionOutcome.BLOCK)

            days = lot.days_to_expiry(today)
            if lot.status == LotStatus.EXPIRED.value or (days is not None and days < 0):
                reasons.append(ReasonCode.LOT_EXPIRED.value)
                messages.append(f"Lô {lot.lot_id} đã hết hạn sử dụng ({lot.expiry_date}).")
                escalate(DecisionOutcome.BLOCK)
            elif days is not None and days <= self.cfg.expiry_warning_days:
                reasons.append(ReasonCode.LOT_NEAR_EXPIRY.value)
                messages.append(f"Lô {lot.lot_id} còn {days} ngày tới hạn sử dụng.")
                escalate(DecisionOutcome.ALLOW_WITH_WARNING)

            if lot.origin_country in ("", "UNKNOWN"):
                reasons.append(ReasonCode.ORIGIN_UNKNOWN.value)
                messages.append("Hồ sơ lô không ghi nước xuất xứ.")
                escalate(DecisionOutcome.ASSIST)

            # Chứng từ
            have = record.verified_evidence_types()
            required = self._required_evidence(record)
            missing = [t for t in required if t not in have]
            pending = [e for e in record.evidence if e.status == EvidenceStatus.PENDING.value]
            rejected = [e for e in record.evidence if e.status == EvidenceStatus.REJECTED.value]
            if rejected:
                reasons.append(ReasonCode.PROVENANCE_REJECTED.value)
                messages.append(f"{len(rejected)} chứng từ đã bị từ chối hiệu lực.")
                escalate(DecisionOutcome.BLOCK)
            if missing:
                reasons.append(ReasonCode.PROVENANCE_INCOMPLETE.value)
                messages.append("Thiếu chứng từ bắt buộc: " + ", ".join(missing))
                escalate(DecisionOutcome.ASSIST)
            elif pending:
                reasons.append(ReasonCode.PROVENANCE_PENDING.value)
                messages.append(f"{len(pending)} chứng từ chưa được đối chiếu bản gốc.")
                escalate(DecisionOutcome.ALLOW_WITH_WARNING)

            missing_fields = record.missing_traceability_fields()
            if missing_fields:
                reasons.append(ReasonCode.TRACE_CHAIN_INCOMPLETE.value)
                messages.append(
                    "Hồ sơ chưa đủ trường truy xuất tối thiểu (Thông tư 02/2024/TT-BKHCN): "
                    + ", ".join(missing_fields)
                )
                escalate(DecisionOutcome.ALLOW_WITH_WARNING)

            if not record.events:
                reasons.append(ReasonCode.TRACE_CHAIN_INCOMPLETE.value)
                messages.append("Lô chưa có sự kiện chuỗi cung ứng nào được ghi nhận.")
                escalate(DecisionOutcome.ALLOW_WITH_WARNING)

            if ReasonCode.LOT_RECALLED.value in reasons or ReasonCode.LOT_EXPIRED.value in reasons \
                    or ReasonCode.LOT_QUARANTINED.value in reasons or ReasonCode.PROVENANCE_REJECTED.value in reasons \
                    or ReasonCode.LOT_SKU_CONFLICT.value in reasons \
                    or ReasonCode.LOT_SOLD_OUT.value in reasons or ReasonCode.LOT_STATUS_INVALID.value in reasons:
                provenance_status = PROVENANCE_BLOCKED
            elif missing or ReasonCode.ORIGIN_UNKNOWN.value in reasons:
                provenance_status = PROVENANCE_INCOMPLETE
            else:
                provenance_status = PROVENANCE_OK

        # ---------- 3. Rủi ro giá trị giao dịch ----------
        price = claimed_price
        if price is None and record and record.sku:
            price = record.sku.price
        prices = [claimed_price] if claimed_price is not None else []
        if record and record.sku:
            prices.append(record.sku.price)
        if any(not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 for p in prices):
            reasons.append(ReasonCode.PRICE_INVALID.value)
            messages.append("Giá mặt hàng không hợp lệ; cần nhân viên kiểm tra và cập nhật.")
            escalate(DecisionOutcome.ASSIST)
        elif price is not None and price >= self.cfg.high_value_threshold and outcome != DecisionOutcome.ALLOW:
            reasons.append(ReasonCode.HIGH_VALUE_REVIEW.value)
            messages.append(
                f"Giao dịch giá trị cao ({price:,.0f} VND) kèm cảnh báo: bắt buộc nhân viên kiểm tra."
            )
            escalate(DecisionOutcome.ASSIST)

        # Nghi vấn đổi mã có động cơ kinh tế: khai hàng rẻ nhưng ảnh giống hàng đắt.
        if verification is not None and verification.status == VERIFY_MISMATCH:
            reasons.append(ReasonCode.PRICE_RISK.value)

        return PolicyDecision(
            outcome=outcome.value,
            reasons=list(dict.fromkeys(reasons)),   # loại trùng, giữ thứ tự
            messages=messages,
            provenance_status=provenance_status,
            origin_summary=self._origin_text(record) if record else "Chưa xác minh",
            requires_operator=outcome in (DecisionOutcome.ASSIST, DecisionOutcome.BLOCK),
            audit_notes={
                "required_evidence": list(self._required_evidence(record)) if record else [],
                "verified_evidence": sorted(record.verified_evidence_types()) if record else [],
                "severity": _SEVERITY[outcome],
            },
        )
