"""Các tình huống kiểm tra policy bằng dữ liệu có kiểm soát.

Mỗi tình huống có kết quả mong đợi để đối chiếu với quyết định của hệ thống.
Các trường hợp gồm thiếu hồ sơ, chứng từ hết hiệu lực, lô thu hồi và SKU
khai báo không khớp ảnh. Kết quả đo xử lý nghiệp vụ, tách khỏi accuracy
nhận diện; dữ liệu tình huống là mô phỏng.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Sequence

from rpv.domain.models import (
    Evidence, EvidenceStatus, EvidenceType, Lot, LotStatus, Product,
    SKU, Supplier, TraceEvent,
)
from rpv.provenance.repository import ProvenanceRepository
from rpv.util import stable_hash_hex

# Kết cục mong đợi cho từng kịch bản. "assist_or_block" nghĩa là hệ thống bắt
# buộc phải dừng luồng tự động; chấp nhận cả hai mức nghiêm trọng.
SCENARIOS = (
    ("happy_path", "Ảnh khớp lời khai, lô hợp lệ, chứng từ đủ", "allow"),
    ("missing_lot", "Mặt hàng có ảnh nhưng chưa có hồ sơ lô", "assist"),
    ("missing_evidence", "Có lô nhưng thiếu chứng từ bắt buộc", "assist"),
    ("pending_evidence", "Chứng từ đã nộp nhưng chưa đối chiếu bản gốc", "allow_warning"),
    ("expired_lot", "Lô đã hết hạn sử dụng", "block"),
    ("recalled_lot", "Lô đang bị thu hồi", "block"),
    ("ticket_switch", "Khai hàng rẻ nhưng ảnh là hàng đắt", "block"),
    ("unknown_item", "Vật thể ngoài danh mục đưa vào camera", "assist_or_block"),
    ("imported_missing_customs", "Hàng nhập khẩu thiếu tờ khai/kiểm dịch", "assist"),
)


@dataclass
class ScenarioCase:
    name: str
    description: str
    expected: str
    image_path: str
    claimed_sku: str
    claimed_lot: Optional[str] = None
    claimed_price: Optional[float] = None
    notes: str = ""


def _mk_supplier(repo: ProvenanceRepository, idx: int, country: str = "VN") -> Supplier:
    s = Supplier(
        supplier_id=f"sup_{idx:03d}", name=f"HTX Nông sản Số {idx}",
        address=f"Số {idx} đường Vành Đai, Tỉnh Long An", gln=f"893{idx:010d}",
        country=country, contact=f"0900{idx:06d}", certifications=["VietGAP"] if idx % 2 else [],
    )
    return repo.upsert_supplier(s)


def build_provenance_scenarios(repo: ProvenanceRepository,
                               sku_to_images: dict[str, list[str]],
                               unknown_images: Sequence[str] = (),
                               today: Optional[date] = None) -> list[ScenarioCase]:
    """Seed dữ liệu cho từng kịch bản + trả về danh sách case để chạy.

    ``sku_to_images``: sku_id -> ảnh test (KHÔNG dùng ảnh support, tránh đo lại
    chính ảnh đã dùng để tạo prototype).
    """
    today = today or date.today()
    skus = sorted(k for k, v in sku_to_images.items() if v)
    slots = ["happy", "missing_lot", "missing_evidence", "pending", "expired",
             "recalled", "cheap", "import"]
    if len(skus) < len(slots):
        # Bản trước kiểm tra ">= 7" nhưng cấp phát 8 slot, nên đúng ở mốc 7 SKU
        # thì vỡ bằng IndexError thay vì thông báo tử tế. Giờ số kiểm tra được
        # lấy từ chính danh sách slot nên không thể lệch lại lần nữa.
        raise ValueError(
            f"Cần >= {len(slots)} SKU có ảnh test để mỗi kịch bản dùng một SKU riêng "
            f"(đang có {len(skus)}). Dùng chung SKU giữa các kịch bản làm lô hàng lẫn nhau "
            "và kết quả mất ý nghĩa."
        )
    cases: list[ScenarioCase] = []
    sup_vn = _mk_supplier(repo, 1)
    sup_import = _mk_supplier(repo, 2, country="US")

    # Cấp phát SKU riêng cho từng kịch bản. Lần trước dùng chung SKU giữa
    # "missing_lot" và "recalled_lot" khiến lô bị thu hồi của kịch bản sau bị
    # gán cho kịch bản trước — một lỗi chỉ lộ ra khi chạy bộ kịch bản đầy đủ.
    S = {name: skus[i] for i, name in enumerate(slots)}

    def ensure_sku(sku_id: str, price: float, category: str = "fresh_produce") -> SKU:
        product = repo.upsert_product(Product(
            product_id=f"prd_{sku_id}", name=sku_id.replace("_", " ").title(),
            category=category, brand="", parent_category=sku_id.split("_")[0],
            image_uri=(sku_to_images.get(sku_id) or [""])[0],
        ))
        return repo.upsert_sku(SKU(
            sku_id=sku_id, product_id=product.product_id, name=product.name,
            unit="kg", price=price, gtin="", is_barcodeless=True,
        ))

    def add_evidence(lot_id: str, types: Sequence[str], status: str = EvidenceStatus.VERIFIED.value,
                     valid_until: Optional[str] = None) -> None:
        for t in types:
            # ID tiền định theo (lô, loại chứng từ) để seed IDEMPOTENT.
            # Bản đầu dùng ID ngẫu nhiên: mỗi lần chạy lại `evaluate` là chứng từ
            # bị nhân bản trong DB. Lỗi chỉ lộ ra khi chạy pipeline hai lần rồi
            # mở hồ sơ lô ra xem — không có test nào bắt được kiểu lỗi này.
            repo.add_evidence(Evidence(
                evidence_id=f"ev_{lot_id}_{t}", lot_id=lot_id, evidence_type=t,
                # hash() bị random hoá theo tiến trình: dùng nó thì sha256 của
                # chứng từ đổi mỗi lần chạy, và bộ dữ liệu seed không tái lập được.
                uri=f"file://docs/{lot_id}/{t}.pdf", sha256=stable_hash_hex(lot_id, t),
                issuer="Nhà cung cấp", issued_at=str(today - timedelta(days=2)),
                valid_until=valid_until, status=status,
                verified_by="staff_01" if status == EvidenceStatus.VERIFIED.value else "",
                verified_at=str(today) if status == EvidenceStatus.VERIFIED.value else None,
            ))

    def add_event(lot_id: str, step: str = "receiving") -> None:
        repo.add_event(TraceEvent(
            event_id=f"evt_{lot_id}_{step}", lot_id=lot_id, biz_step=step, disposition="in_progress",
            event_time=str(today - timedelta(days=1)), location_name="Kho trung tâm",
            actor="warehouse_01", payload={"seeded": True},
        ))

    # Mọi SKU có prototype thị giác đều tồn tại trong danh mục nghiệp vụ. Nhờ đó
    # trạng thái "chưa có hồ sơ lô" (PROVENANCE_MISSING) là đường đi được kiểm
    # tra, thay vì bị che bởi lỗi "không tìm thấy SKU".
    for i, s_id in enumerate(skus):
        ensure_sku(s_id, 20_000 + (i % 7) * 5_000)

    # 1) happy_path -------------------------------------------------------
    sku = S["happy"]
    ensure_sku(sku, 42_000)
    lot = repo.upsert_lot(Lot(
        lot_id="lot_happy", sku_id=sku, supplier_id=sup_vn.supplier_id, origin_country="VN",
        origin_region="Cao Lãnh, Đồng Tháp", farm_or_facility="Nông trại Bảy Núi",
        gln=sup_vn.gln, production_date=str(today - timedelta(days=3)),
        received_date=str(today - timedelta(days=1)), expiry_date=str(today + timedelta(days=20)),
        quantity=120, status=LotStatus.ACTIVE.value,
    ))
    add_evidence(lot.lot_id, [EvidenceType.DELIVERY_NOTE.value, EvidenceType.SUPPLIER_DECLARATION.value])
    add_event(lot.lot_id)
    cases.append(ScenarioCase("happy_path", SCENARIOS[0][1], "allow",
                              sku_to_images[sku][0], sku, lot.lot_id, 42_000))

    # 2) missing_lot ------------------------------------------------------
    sku2 = S["missing_lot"]
    ensure_sku(sku2, 28_000)
    cases.append(ScenarioCase("missing_lot", SCENARIOS[1][1], "assist",
                              sku_to_images[sku2][0], sku2, None, 28_000,
                              notes="Không seed lô nào cho SKU này."))

    # 3) missing_evidence -------------------------------------------------
    sku3 = S["missing_evidence"]
    ensure_sku(sku3, 35_000)
    lot3 = repo.upsert_lot(Lot(
        lot_id="lot_noevidence", sku_id=sku3, supplier_id=sup_vn.supplier_id, origin_country="VN",
        origin_region="Lâm Đồng", farm_or_facility="Trang trại Đơn Dương", gln=sup_vn.gln,
        production_date=str(today - timedelta(days=4)), received_date=str(today - timedelta(days=1)),
        expiry_date=str(today + timedelta(days=15)), quantity=80, status=LotStatus.ACTIVE.value,
    ))
    add_event(lot3.lot_id)
    cases.append(ScenarioCase("missing_evidence", SCENARIOS[2][1], "assist",
                              sku_to_images[sku3][0], sku3, lot3.lot_id, 35_000))

    # 4) pending_evidence -------------------------------------------------
    sku4 = S["pending"]
    ensure_sku(sku4, 31_000)
    lot4 = repo.upsert_lot(Lot(
        lot_id="lot_pending", sku_id=sku4, supplier_id=sup_vn.supplier_id, origin_country="VN",
        origin_region="Tiền Giang", farm_or_facility="Vườn Chợ Gạo", gln=sup_vn.gln,
        production_date=str(today - timedelta(days=2)), received_date=str(today),
        expiry_date=str(today + timedelta(days=18)), quantity=60, status=LotStatus.ACTIVE.value,
    ))
    add_evidence(lot4.lot_id, [EvidenceType.DELIVERY_NOTE.value, EvidenceType.SUPPLIER_DECLARATION.value])
    add_evidence(lot4.lot_id, [EvidenceType.LAB_TEST_REPORT.value], status=EvidenceStatus.PENDING.value)
    add_event(lot4.lot_id)
    cases.append(ScenarioCase("pending_evidence", SCENARIOS[3][1], "allow_warning",
                              sku_to_images[sku4][0], sku4, lot4.lot_id, 31_000))

    # 5) expired_lot ------------------------------------------------------
    sku5 = S["expired"]
    ensure_sku(sku5, 26_000)
    lot5 = repo.upsert_lot(Lot(
        lot_id="lot_expired", sku_id=sku5, supplier_id=sup_vn.supplier_id, origin_country="VN",
        origin_region="Bến Tre", farm_or_facility="Vườn Mỏ Cày", gln=sup_vn.gln,
        production_date=str(today - timedelta(days=30)), received_date=str(today - timedelta(days=20)),
        expiry_date=str(today - timedelta(days=2)), quantity=25, status=LotStatus.ACTIVE.value,
    ))
    add_evidence(lot5.lot_id, [EvidenceType.DELIVERY_NOTE.value, EvidenceType.SUPPLIER_DECLARATION.value])
    add_event(lot5.lot_id)
    cases.append(ScenarioCase("expired_lot", SCENARIOS[4][1], "block",
                              sku_to_images[sku5][0], sku5, lot5.lot_id, 26_000))

    # 6) recalled_lot -----------------------------------------------------
    sku6 = S["recalled"]
    ensure_sku(sku6, 33_000)
    lot6 = repo.upsert_lot(Lot(
        lot_id="lot_recalled", sku_id=sku6, supplier_id=sup_vn.supplier_id, origin_country="VN",
        origin_region="Hải Dương", farm_or_facility="HTX Thanh Hà", gln=sup_vn.gln,
        production_date=str(today - timedelta(days=5)), received_date=str(today - timedelta(days=2)),
        expiry_date=str(today + timedelta(days=10)), quantity=40,
        status=LotStatus.RECALLED.value, notes="Thu hồi do phát hiện dư lượng vượt ngưỡng.",
    ))
    add_evidence(lot6.lot_id, [EvidenceType.DELIVERY_NOTE.value, EvidenceType.SUPPLIER_DECLARATION.value])
    add_event(lot6.lot_id, step="inspecting")
    cases.append(ScenarioCase("recalled_lot", SCENARIOS[5][1], "block",
                              sku_to_images[sku6][0], sku6, lot6.lot_id, 33_000))

    # 7) ticket_switch ----------------------------------------------------
    # Ảnh của SKU đắt nhưng lời khai dùng mã SKU rẻ.
    expensive, cheap = S["happy"], S["cheap"]
    ensure_sku(cheap, 12_000)
    lot7 = repo.upsert_lot(Lot(
        lot_id="lot_cheap", sku_id=cheap, supplier_id=sup_vn.supplier_id, origin_country="VN",
        origin_region="Vĩnh Long", farm_or_facility="Vườn Bình Minh", gln=sup_vn.gln,
        production_date=str(today - timedelta(days=2)), received_date=str(today),
        expiry_date=str(today + timedelta(days=12)), quantity=90, status=LotStatus.ACTIVE.value,
    ))
    add_evidence(lot7.lot_id, [EvidenceType.DELIVERY_NOTE.value, EvidenceType.SUPPLIER_DECLARATION.value])
    add_event(lot7.lot_id)
    cases.append(ScenarioCase("ticket_switch", SCENARIOS[6][1], "block",
                              sku_to_images[expensive][0], cheap, lot7.lot_id, 12_000,
                              notes=f"Ảnh thật là {expensive}, khai là {cheap}."))

    # 8) unknown_item -----------------------------------------------------
    if unknown_images:
        cases.append(ScenarioCase("unknown_item", SCENARIOS[7][1], "assist_or_block",
                                  unknown_images[0], sku, lot.lot_id, 42_000,
                                  notes="Vật thể ngoài danh mục nhưng khai là hàng hợp lệ."))

    # 9) imported_missing_customs -----------------------------------------
    sku9 = S["import"]
    ensure_sku(sku9, 89_000, category="fresh_produce")
    lot9 = repo.upsert_lot(Lot(
        lot_id="lot_import", sku_id=sku9, supplier_id=sup_import.supplier_id, origin_country="US",
        origin_region="Washington", farm_or_facility="Yakima Valley Orchard", gln=sup_import.gln,
        production_date=str(today - timedelta(days=25)), received_date=str(today - timedelta(days=3)),
        expiry_date=str(today + timedelta(days=40)), quantity=200, status=LotStatus.ACTIVE.value,
    ))
    add_evidence(lot9.lot_id, [EvidenceType.DELIVERY_NOTE.value])   # thiếu tờ khai + kiểm dịch
    add_event(lot9.lot_id)
    cases.append(ScenarioCase("imported_missing_customs", SCENARIOS[8][1], "assist",
                              sku_to_images[sku9][0], sku9, lot9.lot_id, 89_000))
    # Same SKU, different lot: appearance cannot establish which lot was sold.
    sku = S['happy']
    lot_other = repo.upsert_lot(Lot(
        lot_id='lot_same_sku_expired', sku_id=sku, supplier_id=sup_vn.supplier_id,
        origin_country='VN', origin_region='Vùng khác', expiry_date=str(today - timedelta(days=1)),
        production_date=str(today - timedelta(days=10)), received_date=str(today - timedelta(days=5)),
        status=LotStatus.ACTIVE.value))
    add_evidence(lot_other.lot_id, [EvidenceType.DELIVERY_NOTE.value, EvidenceType.SUPPLIER_DECLARATION.value])
    add_event(lot_other.lot_id)
    cases.append(ScenarioCase('same_sku_different_lot', 'Ảnh cùng SKU nhưng POS chọn lô đã hết hạn',
                              'block', sku_to_images[sku][0], sku, lot_other.lot_id))
    lot_cert = repo.upsert_lot(Lot(
        lot_id='lot_expired_certificate', sku_id=sku, supplier_id=sup_vn.supplier_id,
        origin_country='VN', expiry_date=str(today + timedelta(days=20)),
        production_date=str(today - timedelta(days=2)), received_date=str(today), status=LotStatus.ACTIVE.value))
    add_evidence(lot_cert.lot_id, [EvidenceType.DELIVERY_NOTE.value, EvidenceType.SUPPLIER_DECLARATION.value],
                 valid_until=str(today - timedelta(days=1)))
    add_event(lot_cert.lot_id)
    cases.append(ScenarioCase('expired_certificate', 'Lô còn hạn nhưng chứng từ bắt buộc đã hết hiệu lực',
                              'assist', sku_to_images[sku][0], sku, lot_cert.lot_id))
    cases.append(ScenarioCase('lot_sku_conflict', 'Nhận đúng SKU nhưng POS chọn lô của SKU khác',
                              'block', sku_to_images[sku][0], sku, lot7.lot_id))
    return cases


_VISION_REASONS = {
    "VISION_UNKNOWN", "VISION_AMBIGUOUS", "VISION_MISMATCH",
    "VISION_SKU_NOT_REGISTERED", "VISION_CATALOG_EMPTY",
}


def _divergence_source(resp) -> str:
    """Phân loại nguyên nhân khi kết quả khác với mong đợi.

    Lỗi nhận diện do encoder hoặc chất lượng ảnh được phân biệt với lỗi
    quy tắc nghiệp vụ để đánh giá đúng tỷ lệ đạt của từng phần. Luật nghiệp
    vụ được kiểm tra riêng bằng unit test với tín hiệu thị giác giả định đúng.
    """
    reasons = set(resp.reasons)
    if reasons & _VISION_REASONS:
        return "vision"
    return "policy"


def evaluate_scenarios(service, cases: Sequence[ScenarioCase]) -> dict:
    """Chạy toàn bộ kịch bản qua ``CheckoutService`` và đối chiếu kết cục."""
    rows, n_pass = [], 0
    for case in cases:
        resp = service.process(
            case.image_path, claimed_sku=case.claimed_sku, claimed_lot=case.claimed_lot,
            claimed_price=case.claimed_price, operator="evaluator",
        )
        expected = case.expected
        ok = (resp.outcome in ("assist", "block")) if expected == "assist_or_block" else (resp.outcome == expected)
        n_pass += int(ok)
        rows.append({
            "scenario": case.name, "description": case.description,
            "expected": expected, "actual": resp.outcome, "passed": ok,
            "divergence_source": "" if ok else _divergence_source(resp),
            "reasons": resp.reasons, "provenance_status": resp.provenance_status,
            "origin_summary": resp.origin_summary,
            "vision_status": (resp.recognition or {}).get("status", ""),
            "verification_status": (resp.verification or {}).get("status", ""),
            "messages": resp.messages, "notes": case.notes,
            'decision_state': resp.decision_state,
            'vision_only_accept': bool(resp.verification and resp.verification.get('status') == 'confirmed'),
            'expected_auto_accept': expected in {'allow', 'allow_warning'},
        })
    n_vision = sum(1 for r in rows if r["divergence_source"] == "vision")
    n_policy = sum(1 for r in rows if r["divergence_source"] == "policy")
    return {
        "n_cases": len(rows), "n_passed": n_pass,
        "pass_rate": n_pass / len(rows) if rows else float("nan"),
        "n_failed_due_to_vision": n_vision,
        "n_failed_due_to_policy": n_policy,
        "policy_pass_rate_excluding_vision_errors": (
            (n_pass + n_vision) / len(rows) if rows else float("nan")
        ),
        "cases": rows,
        'image_only_vs_policy': {
            'n_unsafe_accept_image_only': sum(r['vision_only_accept'] and not r['expected_auto_accept'] for r in rows),
            'n_unsafe_accept_full_policy': sum(r['actual'] in {'allow', 'allow_warning'} and not r['expected_auto_accept'] for r in rows),
            'scope': 'Controlled simulated scenarios; count of unsafe automatic acceptance, not a field prevalence estimate.',
        },
    }
