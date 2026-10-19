"""Kết nối dữ liệu ngoài và chuyển đổi định dạng.

Open Food Facts cung cấp thông tin cộng đồng, không thay thế chứng từ.
Dữ liệu được đánh dấu ``authoritative=False`` và ``evidence_grade="none"``.

GS1 Digital Link hỗ trợ phân tích và sinh URI offline, không cần tài khoản
resolver. EPCIS 2.0 hỗ trợ xuất dữ liệu nội bộ theo định dạng trao đổi;
chức năng này chưa đồng nghĩa với kết nối một hệ thống EPCIS bên ngoài.

Gọi mạng tắt mặc định qua ``app_config.settings.enable_external_adapters``
để các lượt chạy không phụ thuộc dịch vụ ngoài.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from rpv import config as app_config
from rpv.domain.models import TraceEvent

# Application Identifier của GS1 dùng trong Digital Link.
GS1_AI = {
    "01": "gtin", "8006": "itip", "8013": "gmn", "8010": "cpid",
    "410": "ship_to_gln", "414": "gln", "417": "party_gln",
    "10": "batch_lot", "21": "serial", "7003": "expiry_datetime",
    "17": "expiry_date", "11": "production_date", "15": "best_before",
}


@dataclass
class EnrichmentResult:
    """Kết quả làm giàu từ nguồn ngoài — luôn kèm mức độ tin cậy pháp lý."""
    source: str
    found: bool = False
    authoritative: bool = False
    evidence_grade: str = "none"    # none | informational | authoritative
    fields: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    disclaimer: str = ""

    def to_json(self) -> dict:
        return {
            "source": self.source, "found": self.found, "authoritative": self.authoritative,
            "evidence_grade": self.evidence_grade, "fields": self.fields,
            "error": self.error, "disclaimer": self.disclaimer,
        }


def parse_gs1_digital_link(url: str) -> dict[str, str]:
    """Phân tích URI GS1 Digital Link -> dict khoá đọc được.

    Ví dụ: ``https://id.example.com/01/09506000134352/10/LOT-A/21/9988``
    -> ``{"gtin": "09506000134352", "batch_lot": "LOT-A", "serial": "9988"}``

    Hỗ trợ cả trường hợp AI nằm ở query string (``?10=LOT-A``), vốn là dạng hợp
    lệ cho các AI không thuộc nhóm định danh chính.
    """
    parsed = urllib.parse.urlparse(url)
    segments = [s for s in parsed.path.split("/") if s]
    out: dict[str, str] = {}
    i = 0
    while i < len(segments) - 1:
        ai = segments[i]
        if ai in GS1_AI:
            out[GS1_AI[ai]] = urllib.parse.unquote(segments[i + 1])
            i += 2
        else:
            i += 1
    for key, values in urllib.parse.parse_qs(parsed.query).items():
        if key in GS1_AI and values:
            out.setdefault(GS1_AI[key], values[0])
    return out


def build_gs1_digital_link(gtin: str, *, batch_lot: str = "", serial: str = "",
                           domain: str = "https://id.rpv.local") -> str:
    """Sinh URI GS1 Digital Link cho một lô. Dùng làm ``trace_code`` dạng URI để
    lô hàng có định danh phân giải được về sau."""
    if not re.fullmatch(r"\d{8}|\d{12,14}", gtin or ""):
        raise ValueError(f"GTIN không hợp lệ cho Digital Link: {gtin!r}")
    url = f"{domain.rstrip('/')}/01/{gtin}"
    if batch_lot:
        url += f"/10/{urllib.parse.quote(batch_lot, safe='')}"
    if serial:
        url += f"/21/{urllib.parse.quote(serial, safe='')}"
    return url


class NullExternalAdapter:
    """Adapter mặc định khi tắt kết nối ngoài: trả lời "không có dữ liệu" một
    cách tường minh. Quan trọng hơn nó tưởng: hệ thống phải hành xử giống nhau
    dù nguồn ngoài có sẵn hay không."""

    name = "disabled"

    def enrich_by_gtin(self, gtin: str) -> EnrichmentResult:
        return EnrichmentResult(
            source=self.name, found=False, evidence_grade="none",
            disclaimer="Adapter ngoài đang tắt (RPV_EXTERNAL_ADAPTERS=0); quyết định chỉ dựa trên dữ liệu nội bộ.",
        )


class OpenFoodFactsAdapter:
    """Làm giàu thông tin sản phẩm đóng gói từ Open Food Facts (API v3).

    Dùng để: hiển thị tên/thương hiệu/nhóm hàng, và đối chiếu chéo trường
    ``origins``/``manufacturing_places`` do cộng đồng khai với hồ sơ lô nội bộ —
    khi hai bên lệch nhau thì đó là tín hiệu đáng soát lại, KHÔNG phải bằng
    chứng kết luận.
    """

    name = "open_food_facts"
    BASE = "https://world.openfoodfacts.org/api/v3/product"
    FIELDS = ("product_name,brands,categories_tags,origins,origins_tags,"
              "manufacturing_places,countries_tags,labels_tags,quantity,image_url")

    def __init__(self, timeout: float = 6.0, user_agent: str = "RPV-Thesis/1.0 (research)") -> None:
        self.timeout = timeout
        self.user_agent = user_agent

    def enrich_by_gtin(self, gtin: str) -> EnrichmentResult:
        disclaimer = (
            "Open Food Facts là dữ liệu mở do cộng đồng đóng góp. Chỉ dùng để tham khảo/hiển thị; "
            "KHÔNG có giá trị chứng từ và không được dùng để kết luận nguồn gốc."
        )
        if not app_config.settings.enable_external_adapters:
            return EnrichmentResult(source=self.name, found=False, error="external_adapters_disabled",
                                    disclaimer=disclaimer)
        if not gtin:
            return EnrichmentResult(source=self.name, found=False, error="empty_gtin", disclaimer=disclaimer)
        url = f"{self.BASE}/{urllib.parse.quote(gtin)}?fields={self.FIELDS}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            return EnrichmentResult(source=self.name, found=False, error=f"{type(exc).__name__}: {exc}",
                                    disclaimer=disclaimer)
        product = payload.get("product") or {}
        if not product:
            return EnrichmentResult(source=self.name, found=False, error="product_not_found",
                                    disclaimer=disclaimer)
        return EnrichmentResult(
            source=self.name, found=True, authoritative=False, evidence_grade="informational",
            fields={
                "product_name": product.get("product_name", ""),
                "brands": product.get("brands", ""),
                "origins": product.get("origins", ""),
                "origins_tags": product.get("origins_tags", []),
                "manufacturing_places": product.get("manufacturing_places", ""),
                "countries_tags": product.get("countries_tags", []),
                "labels_tags": product.get("labels_tags", []),
                "categories_tags": product.get("categories_tags", []),
                "image_url": product.get("image_url", ""),
            },
            disclaimer=disclaimer,
        )

    @staticmethod
    def cross_check_origin(internal_country: str, enrichment: EnrichmentResult) -> dict:
        """So sánh xuất xứ nội bộ với dữ liệu cộng đồng. Trả về tín hiệu để
        policy/nhân viên xem xét, không tự động thay đổi kết luận."""
        if not enrichment.found:
            return {"comparable": False, "conflict": False, "note": "Không có dữ liệu ngoài để đối chiếu."}
        tags = [t.lower() for t in enrichment.fields.get("origins_tags", [])]
        if not tags:
            return {"comparable": False, "conflict": False, "note": "Nguồn ngoài không khai xuất xứ."}
        code = (internal_country or "").lower()
        mentions = any(code and (code in t) for t in tags)
        return {
            "comparable": True,
            "conflict": not mentions,
            "external_origins": tags,
            "internal_country": internal_country,
            "note": ("Xuất xứ nội bộ không xuất hiện trong dữ liệu cộng đồng — cần soát lại chứng từ."
                     if not mentions else "Xuất xứ nội bộ trùng khớp dữ liệu cộng đồng (chỉ tham khảo)."),
        }


class EpcisExporter:
    """Xuất sự kiện nội bộ ra EPCIS 2.0 JSON-LD (EPCISDocument)."""

    name = "epcis_2_0"

    @staticmethod
    def export(events: list[TraceEvent], trace_code: str = "",
               creation_date: str | None = None) -> dict:
        # creationDate là trường BẮT BUỘC của EPCISDocument. Bản trước để None,
        # nên tài liệu xuất ra không hợp lệ với schema GS1 — đúng loại lỗi làm
        # một tuyên bố "tương thích chuẩn" trở thành không đúng.
        return {
            "@context": ["https://ref.gs1.org/standards/epcis/2.0.0/epcis-context.jsonld"],
            "type": "EPCISDocument",
            "schemaVersion": "2.0",
            "creationDate": creation_date or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "epcisBody": {"eventList": [e.to_epcis(trace_code) for e in events]},
        }
