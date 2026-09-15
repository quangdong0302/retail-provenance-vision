"""Mô hình nghiệp vụ. Đây là nơi "nguồn gốc" trở thành dữ liệu có thể kiểm chứng.

Thiết kế theo tinh thần TCVN 12850:2019 và Thông tư 02/2024/TT-BKHCN: sản phẩm
được định danh (mã truy vết sản phẩm), địa điểm được định danh (mã truy vết địa
điểm), và mỗi công đoạn trong chuỗi là một sự kiện có thời gian. Vì vậy
``TraceEvent`` mượn cấu trúc EPCIS 2.0 (what/when/where/why) — không phải để
"cho có chuẩn quốc tế", mà vì nó là cấu trúc tối thiểu đủ trả lời câu hỏi
"lô hàng này đi qua những đâu".
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Optional


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


class LotStatus(str, Enum):
    ACTIVE = "active"            # đang bán bình thường
    QUARANTINED = "quarantined"  # tạm giữ, chờ kiểm tra
    RECALLED = "recalled"        # bị thu hồi
    EXPIRED = "expired"          # hết hạn (đánh dấu thủ công hoặc do job)
    SOLD_OUT = "sold_out"


class EvidenceStatus(str, Enum):
    VERIFIED = "verified"        # đã đối chiếu với bản gốc
    PENDING = "pending"          # đã nộp, chưa đối chiếu
    REJECTED = "rejected"        # không hợp lệ
    EXPIRED = "expired"          # chứng từ đã quá hiệu lực


class EvidenceType(str, Enum):
    DELIVERY_NOTE = "delivery_note"                  # phiếu giao hàng / hoá đơn nhập
    SUPPLIER_DECLARATION = "supplier_declaration"    # bản tự công bố của nhà cung cấp
    IMPORT_CUSTOMS_DOC = "import_customs_doc"        # tờ khai hải quan
    PHYTOSANITARY_CERT = "phytosanitary_cert"        # kiểm dịch thực vật
    VIETGAP_CERT = "vietgap_cert"
    GLOBALGAP_CERT = "globalgap_cert"
    ORGANIC_CERT = "organic_cert"
    LAB_TEST_REPORT = "lab_test_report"              # kết quả kiểm nghiệm
    PHOTO = "photo"
    QR_PAYLOAD = "qr_payload"
    OTHER = "other"


class OriginCountry(str, Enum):
    VN = "VN"
    US = "US"
    CN = "CN"
    TH = "TH"
    AU = "AU"
    NZ = "NZ"
    JP = "JP"
    KR = "KR"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class DecisionOutcome(str, Enum):
    ALLOW = "allow"                    # cho qua, tự động
    ALLOW_WITH_WARNING = "allow_warning"  # cho qua nhưng ghi cảnh báo
    ASSIST = "assist"                  # cần nhân viên xác nhận
    BLOCK = "block"                    # chặn giao dịch


class ReasonCode(str, Enum):
    """Mã lý do — hiển thị cho nhân viên VÀ dùng để thống kê trong đánh giá.

    Một hệ thống chỉ trả confidence là hệ thống không giải thích được. Mã lý do
    là điều kiện cần để nhân viên tin và để nghiên cứu đo được lỗi ở đâu.
    """
    VISION_OK = "VISION_OK"
    VISION_UNKNOWN = "VISION_UNKNOWN"
    VISION_AMBIGUOUS = "VISION_AMBIGUOUS"
    VISION_MISMATCH = "VISION_MISMATCH"
    VISION_SKU_NOT_REGISTERED = "VISION_SKU_NOT_REGISTERED"
    VISION_CATALOG_EMPTY = "VISION_CATALOG_EMPTY"
    SKU_NOT_FOUND = "SKU_NOT_FOUND"
    LOT_NOT_FOUND = "LOT_NOT_FOUND"
    LOT_SKU_CONFLICT = "LOT_SKU_CONFLICT"
    LOT_RECALLED = "LOT_RECALLED"
    LOT_QUARANTINED = "LOT_QUARANTINED"
    LOT_EXPIRED = "LOT_EXPIRED"
    LOT_SOLD_OUT = "LOT_SOLD_OUT"
    LOT_STATUS_INVALID = "LOT_STATUS_INVALID"
    LOT_NEAR_EXPIRY = "LOT_NEAR_EXPIRY"
    PROVENANCE_MISSING = "PROVENANCE_MISSING"
    PROVENANCE_INCOMPLETE = "PROVENANCE_INCOMPLETE"
    PROVENANCE_PENDING = "PROVENANCE_PENDING"
    PROVENANCE_REJECTED = "PROVENANCE_REJECTED"
    ORIGIN_CLAIM_CONFLICT = "ORIGIN_CLAIM_CONFLICT"
    ORIGIN_UNKNOWN = "ORIGIN_UNKNOWN"
    TRACE_CHAIN_INCOMPLETE = "TRACE_CHAIN_INCOMPLETE"
    HIGH_VALUE_REVIEW = "HIGH_VALUE_REVIEW"
    PRICE_RISK = "PRICE_RISK"
    PRICE_INVALID = "PRICE_INVALID"


# Trường dữ liệu truy xuất tối thiểu để một lô được coi là "công bố được"
# (đối chiếu Thông tư 02/2024/TT-BKHCN, khoản về dữ liệu phục vụ tra cứu).
TRACEABILITY_REQUIRED_FIELDS: tuple[str, ...] = (
    "product_name",
    "product_image",
    "producer_name",
    "producer_address",
    "production_time",
    "trace_code",
    "brand_or_code",
)

_GTIN_RE = re.compile(r"^\d{8}$|^\d{12,14}$")


@dataclass
class Supplier:
    supplier_id: str
    name: str
    address: str = ""
    gln: str = ""                 # Global Location Number, nếu có
    country: str = OriginCountry.VN.value
    contact: str = ""
    certifications: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Product:
    product_id: str
    name: str
    category: str = "default"           # khớp khoá trong PolicyConfig.required_evidence
    brand: str = ""
    parent_category: str = ""           # dùng cho nhận diện phân cấp (Táo -> Táo Fuji)
    description: str = ""
    image_uri: str = ""

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class SKU:
    """Đơn vị bán hàng. Đây là khoá nối giữa thị giác và nghiệp vụ."""
    sku_id: str
    product_id: str
    name: str
    unit: str = "kg"                    # kg | piece | pack
    price: float = 0.0                  # VND
    gtin: str = ""                       # có thể rỗng với hàng rời không bao bì
    requires_weighing: bool = True
    is_barcodeless: bool = True          # True = nhóm hàng KHÔNG dán mã (rau củ, trái cây rời)
    active: bool = True

    def validate(self) -> list[str]:
        problems = []
        if self.gtin and not _GTIN_RE.match(self.gtin):
            problems.append(f"GTIN không đúng độ dài chuẩn (8/12/13/14 chữ số): {self.gtin!r}")
        if not isinstance(self.price, (int, float)) or not math.isfinite(self.price) or self.price < 0:
            problems.append("Giá phải là số hữu hạn không âm.")
        return problems

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Lot:
    """Lô hàng — nơi câu hỏi 'từ đâu' thực sự được trả lời."""
    lot_id: str
    sku_id: str
    supplier_id: str
    origin_country: str = OriginCountry.UNKNOWN.value
    origin_region: str = ""              # tỉnh/huyện/vùng trồng
    farm_or_facility: str = ""           # tên nông trại/cơ sở đóng gói
    gln: str = ""                        # mã truy vết địa điểm
    trace_code: str = ""                 # mã truy xuất nguồn gốc của lô
    production_date: Optional[str] = None
    received_date: Optional[str] = None
    expiry_date: Optional[str] = None
    quantity: float = 0.0
    unit: str = "kg"
    status: str = LotStatus.ACTIVE.value
    notes: str = ""

    def days_to_expiry(self, today: Optional[date] = None) -> Optional[int]:
        if not self.expiry_date:
            return None
        today = today or utcnow().date()
        try:
            exp = date.fromisoformat(str(self.expiry_date)[:10])
        except ValueError:
            return None
        return (exp - today).days

    def is_expired(self, today: Optional[date] = None) -> bool:
        d = self.days_to_expiry(today)
        return d is not None and d < 0

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.quantity < 0:
            problems.append("Số lượng lô không thể âm.")
        if self.status not in {s.value for s in LotStatus}:
            problems.append(f"Trạng thái lô không hợp lệ: {self.status!r}.")
        parsed: dict[str, date] = {}
        for name in ("production_date", "received_date", "expiry_date"):
            value = getattr(self, name)
            if value:
                try:
                    parsed[name] = date.fromisoformat(str(value)[:10])
                except (TypeError, ValueError):
                    problems.append(f"{name} phải theo YYYY-MM-DD, nhận {value!r}.")
        return problems

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Evidence:
    """Chứng từ/bằng chứng. ``sha256`` khiến bằng chứng không thể bị thay êm."""
    evidence_id: str
    lot_id: str
    evidence_type: str = EvidenceType.OTHER.value
    uri: str = ""
    sha256: str = ""
    issuer: str = ""
    issued_at: Optional[str] = None
    valid_until: Optional[str] = None
    status: str = EvidenceStatus.PENDING.value
    verified_by: str = ""
    verified_at: Optional[str] = None
    note: str = ""

    def is_currently_valid(self, today: Optional[date] = None) -> bool:
        if self.status != EvidenceStatus.VERIFIED.value:
            return False
        if not self.valid_until:
            return True
        today = today or utcnow().date()
        try:
            return date.fromisoformat(str(self.valid_until)[:10]) >= today
        except (TypeError, ValueError):
            # Ngày hiệu lực hỏng không thể được coi là còn hiệu lực. Fail closed.
            return False

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class TraceEvent:
    """Sự kiện chuỗi cung ứng, cấu trúc theo EPCIS 2.0 (what/when/where/why)."""
    event_id: str
    lot_id: str
    biz_step: str = "receiving"          # harvesting|packing|shipping|receiving|inspecting|retail_selling
    disposition: str = "in_progress"
    event_time: Optional[str] = None
    location_gln: str = ""
    location_name: str = ""
    actor: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    def to_epcis(self, trace_code: str = "") -> dict:
        """Xuất sang JSON-LD kiểu EPCIS 2.0 ObjectEvent. Không tự nhận là đã
        'tích hợp GS1' — đây là biểu diễn tương thích để chứng minh dữ liệu nội
        bộ có thể xuất ra chuẩn khi cần kết nối."""
        return {
            "@context": ["https://ref.gs1.org/standards/epcis/2.0.0/epcis-context.jsonld"],
            "type": "ObjectEvent",
            "eventID": f"urn:uuid:{self.event_id}",
            "eventTime": _iso(self.event_time) or _iso(utcnow()),
            "eventTimeZoneOffset": "+07:00",
            "action": "OBSERVE",
            "bizStep": f"urn:epcglobal:cbv:bizstep:{self.biz_step}",
            "disposition": f"urn:epcglobal:cbv:disp:{self.disposition}",
            "epcList": [trace_code or f"urn:rpv:lot:{self.lot_id}"],
            "readPoint": {"id": f"urn:epc:id:sgln:{self.location_gln}" if self.location_gln else self.location_name},
            "rpv:actor": self.actor,
            "rpv:payload": self.payload,
        }


@dataclass
class DecisionRecord:
    """Một lượt quyết định tại quầy — đơn vị dữ liệu của audit và của đánh giá."""
    request_id: str
    created_at: str
    outcome: str
    reasons: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    image_sha256: str = ""
    claimed_sku: str = ""
    claimed_lot: str = ""
    resolved_sku: str = ""
    resolved_lot: str = ""
    vision_status: str = ""
    vision_top1: str = ""
    vision_distance: Optional[float] = None
    verification_status: str = ""
    provenance_status: str = ""
    encoder_tag: str = ""
    calibration_alpha: Optional[float] = None
    operator: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)
