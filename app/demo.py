"""

Các tab của giao diện demo:

  1. Quầy kiểm tra: kiểm chứng lời khai và ra quyết định có mã lý do.
  2. Đăng ký có kiểm tra: đối chiếu SKU dễ nhầm và hướng dẫn bổ sung ảnh.
  3. Nhập hồ sơ lô: liên kết SKU với nhà cung cấp, nguồn gốc và chứng từ.
  4. Truy xuất: từ lô hàng ra chứng từ, sự kiện và bản xuất EPCIS 2.0.
  5. Kiểm toán: phát hiện thay đổi làm đứt chuỗi hash của lịch sử quyết định.
  6. Hướng dẫn demo: các bước chuẩn bị và tình huống kiểm tra.

Service và DINOv2 được nạp ở lần gọi đầu tiên, không nạp lúc import.
Việc import để đọc tài liệu hoặc dùng công cụ như pydoc không phải chờ
backbone hay phụ thuộc vào cấu hình chạy mô hình.

Chạy: python -m app.demo
"""
from __future__ import annotations

import os
import math
import threading
from datetime import date
from pathlib import Path

import gradio as gr

from app.presentation import empty_result, render_checkout

from rpv.audit.hash_chain import AuditLog
from rpv import config as app_config
from rpv.domain.models import (
    Evidence, EvidenceStatus, Lot, LotStatus, Product, SKU, Supplier, TraceEvent,
)
from rpv.provenance.adapters import EpcisExporter
from rpv.provenance.repository import ProvenanceRepository, new_id
from rpv.services.checkout import CheckoutService
from rpv.util import json_dumps, stable_hash_hex
from rpv.vision.calibration import CalibrationModel
from rpv.vision.encoders import load_encoder
from rpv.vision.prototypes import PrototypeBank

_STATE: dict = {}
_BUILD_LOCK = threading.Lock()


def _build() -> dict:
    app_config.settings.paths.ensure()
    encoder = load_encoder()
    bank = PrototypeBank(encoder)
    if Path(app_config.settings.paths.bank_path).exists():
        bank.load(app_config.settings.paths.bank_path)
    calibration = None
    if Path(app_config.settings.paths.calib_path).exists():
        calibration = CalibrationModel.load(app_config.settings.paths.calib_path, encoder.tag)
    repo = ProvenanceRepository(app_config.settings.paths.db_path)
    repo.init_schema()
    service = CheckoutService(encoder, bank, repo, calibration,
                              audit=AuditLog(app_config.settings.paths.audit_path), bank_path=app_config.settings.paths.bank_path,
                              calibration_path=app_config.settings.paths.calib_path)
    return {"encoder": encoder, "bank": bank, "calib": service.calibration,
            "repo": repo, "service": service}


def state() -> dict:
    """Dựng một lần, có khoá: Gradio gọi handler từ nhiều thread."""
    if not _STATE:
        with _BUILD_LOCK:
            if not _STATE:
                _STATE.update(_build())
    return _STATE


def _repo() -> ProvenanceRepository:
    return state()["repo"]


def _service() -> CheckoutService:
    service = state()["service"]
    service.refresh_catalog()
    return service


def _sku_choices() -> list[str]:
    st = state()
    st['service'].refresh_catalog()
    return sorted({s.sku_id for s in st['repo'].list_skus()} | set(st['bank'].sku_ids))


def _lot_choices(sku_id: str | None = None) -> list[tuple[str, str]]:
    labels = {'active': 'Đang sử dụng', 'quarantined': 'Tạm giữ', 'recalled': 'Thu hồi',
              'expired': 'Hết hạn', 'sold_out': 'Đã hết'}
    return [(f"{r['lot_id']} · {r['sku_id']} · {labels.get(r['status'], r['status'])}", r['lot_id'])
            for r in _repo().list_lot_summaries() if not sku_id or r['sku_id'] == sku_id]


def _lot_id(choice: str | None) -> str | None:
    return choice.split(" ")[0] if choice else None


# ---------- Tab 1: nhập hàng ----------
def intake(sku_id, sku_name, price, category, supplier_name, supplier_addr,
           origin_country, origin_region, farm, production_date, expiry_date,
           evidence_types, evidence_verified, images):
    sku_id = (sku_id or '').strip()
    if not sku_id:
        return "Cần mã mặt hàng.", gr.update(), gr.update()
    try:
        price = float(price or 0)
        if not math.isfinite(price) or price < 0:
            raise ValueError('Giá phải là số hữu hạn, không âm.')
        production = date.fromisoformat(production_date) if production_date else None
        expiry = date.fromisoformat(expiry_date) if expiry_date else None
        if production and expiry and expiry < production:
            raise ValueError('Hạn sử dụng phải từ ngày sản xuất trở đi.')
    except (ValueError, TypeError) as exc:
        return f"Chưa thể lưu: kiểm tra giá và ngày theo định dạng YYYY-MM-DD. {exc}", gr.update(), gr.update()
    registered = _service().bank.entries.get(sku_id)
    guided = registered is not None and bool(registered.enrollment_id)
    if not images and not guided:
        return "Cần ảnh mẫu hoặc đăng ký có kiểm tra trước khi nhập hồ sơ lô.", gr.update(), gr.update()
    repo = _repo()
    paths = [img if isinstance(img, str) else img.name for img in (images or [])]
    # ID nhà cung cấp suy ra từ TÊN: nhập hàng hai lần từ cùng một HTX không
    # được sinh ra hai nhà cung cấp trùng tên, vì truy xuất nguồn gốc dựa trên
    # định danh chứ không dựa trên chuỗi hiển thị.
    sup_name = (supplier_name or "Nhà cung cấp chưa rõ").strip()
    sup = repo.upsert_supplier(Supplier(f"sup_{stable_hash_hex(sup_name, length=10)}", sup_name,
                                        supplier_addr or "", country=origin_country or "VN"))
    prod = repo.upsert_product(Product(f"prd_{sku_id}", sku_name or sku_id,
                                       category=category or "fresh_produce", image_uri=paths[0] if paths else ''))
    try:
        repo.upsert_sku(SKU(sku_id, prod.product_id, sku_name or sku_id, price=float(price or 0)))
    except ValueError as exc:
        return f"Lỗi dữ liệu SKU: {exc}", gr.update(), gr.update()

    lot = repo.upsert_lot(Lot(new_id("lot"), sku_id, sup.supplier_id,
                              origin_country=origin_country or "UNKNOWN",
                              origin_region=origin_region or "", farm_or_facility=farm or "",
                              production_date=production_date or None, received_date=str(date.today()),
                              expiry_date=expiry_date or None, status=LotStatus.ACTIVE.value))
    status = EvidenceStatus.VERIFIED.value if evidence_verified else EvidenceStatus.PENDING.value
    for t in (evidence_types or []):
        repo.add_evidence(Evidence(new_id("ev"), lot.lot_id, t, uri=f"file://docs/{lot.lot_id}/{t}.pdf",
                                   issuer=sup.name, issued_at=str(date.today()), status=status,
                                   verified_by="staff_demo" if evidence_verified else "",
                                   verified_at=str(date.today()) if evidence_verified else None))
    repo.add_event(TraceEvent(new_id("evt"), lot.lot_id, "receiving", event_time=str(date.today()),
                              location_name="Kho cửa hàng", actor="staff_demo"))
    if guided:
        res = {'n_shots': registered.n_shots, 'n_centroids': len(registered.centroids),
               'catalog_size': len(_service().bank)}
    else:
        res = _service().register_product(sku_id, paths, label=sku_name or sku_id, operator="staff_demo")

    missing = repo.resolve(sku_id, lot.lot_id).missing_traceability_fields()
    lines = [
        f"Đã đăng ký **{sku_id}** với {res['n_shots']} ảnh mẫu ({res['n_centroids']} tâm). "
        f"Danh mục thị giác: {res['catalog_size']} mặt hàng.",
        f"Lô: `{lot.lot_id}` · mã truy vết: `{lot.trace_code}`",
        f"Chứng từ đã nộp: {', '.join(evidence_types or []) or 'chưa có'} ({status}).",
        "",
        ("Hồ sơ đủ trường truy xuất tối thiểu." if not missing
         else f"⚠️ Còn thiếu trường truy xuất: {', '.join(missing)}"),
        "",
        ('Giữ nguyên ảnh mẫu của hồ sơ đăng ký có kiểm tra; chỉ thêm hồ sơ lô.' if guided else
         'Danh mục thay đổi; cần nhân viên xác nhận cho tới khi hiệu chỉnh phù hợp.'),
        "_Không train lại mô hình. Nhưng muốn bán được thì lô vẫn phải có chứng từ hợp lệ._",
    ]
    return "\n".join(lines), gr.update(choices=_sku_choices()), gr.update(choices=_lot_choices())


# ---------- Tab 2: quầy thanh toán ----------
def checkout(image, claimed_sku, claimed_lot, price, use_claim):
    if image is None:
        return empty_result('Chưa có ảnh sản phẩm', 'Tải ảnh hoặc chụp bằng camera, sau đó bấm Kiểm tra mặt hàng.'), "{}"
    if use_claim and not claimed_sku:
        return empty_result('Chưa chọn mặt hàng', 'Chọn mã mặt hàng cần đối chiếu hoặc tắt chế độ đối chiếu mã khai báo.'), "{}"
    path = image if isinstance(image, str) else image.name
    resp = _service().process(path, claimed_sku=(claimed_sku if use_claim else None),
                           claimed_lot=_lot_id(claimed_lot) if use_claim else None,
                           claimed_price=float(price) if price else None, operator="staff_demo")
    payload = resp.to_json()
    return render_checkout(payload), json_dumps(payload)


# ---------- Tab 3: truy xuất ----------
def trace(lot_choice, want_epcis):
    lot_id = _lot_id(lot_choice)
    if not lot_id:
        return "Chưa chọn lô.", "{}"
    repo = _repo()
    lot = repo.get_lot(lot_id)
    if lot is None:
        return f"Không tìm thấy lô {lot_id}.", "{}"
    rec = repo.resolve(lot.sku_id, lot_id)
    L = [f"## Hồ sơ lô `{lot_id}`", "",
         f"- Mặt hàng: **{rec.sku.name if rec.sku else lot.sku_id}**",
         f"- Mã truy vết: `{lot.trace_code}`",
         f"- Xuất xứ: {lot.farm_or_facility or '—'} / {lot.origin_region or '—'} / {lot.origin_country}",
         f"- Nhà cung cấp: {rec.supplier.name if rec.supplier else '—'} "
         f"({rec.supplier.address if rec.supplier else '—'})",
         f"- Sản xuất: {lot.production_date or '—'} · Nhập: {lot.received_date or '—'} "
         f"· Hết hạn: {lot.expiry_date or '—'}",
         f"- Trạng thái: `{lot.status}`", ""]
    L += ["### Chứng từ", "", "| Loại | Trạng thái | Hiệu lực đến | Nơi cấp |", "|---|---|---|---|"]
    for e in rec.evidence:
        L.append(f"| {e.evidence_type} | {e.status} | {e.valid_until or '—'} | {e.issuer or '—'} |")
    if not rec.evidence:
        L.append("| _chưa có chứng từ_ | | | |")
    L += ["", "### Chuỗi sự kiện", ""]
    for ev in rec.events:
        L.append(f"- `{ev.event_time}` · **{ev.biz_step}** tại {ev.location_name or '—'} ({ev.actor or '—'})")
    if not rec.events:
        L.append("- _chưa có sự kiện nào_")
    missing = rec.missing_traceability_fields()
    L += ["", ("✅ Đủ trường truy xuất tối thiểu trong hệ thống."
               if not missing else f"⚠️ Thiếu trường: {', '.join(missing)}")]
    payload = rec.to_json()
    if want_epcis:
        payload["epcis_document"] = EpcisExporter.export(rec.events, lot.trace_code)
    return "\n".join(L), json_dumps(payload)


def set_status(lot_choice, status, note):
    lot_id = _lot_id(lot_choice)
    if not lot_id:
        return "Chưa chọn lô.", gr.update()
    _repo().set_lot_status(lot_id, status, note or "")
    AuditLog(app_config.settings.paths.audit_path).append("lot_status_changed",
                                               {"lot_id": lot_id, "status": status, "note": note or ""})
    return (f"Đã đổi trạng thái lô `{lot_id}` thành **{status}**. "
            "Thử thanh toán lại mặt hàng này để thấy policy chặn."), gr.update(choices=_lot_choices())


# ---------- Tab 4: kiểm toán ----------
def audit_view():
    log = AuditLog(app_config.settings.paths.audit_path)
    res = log.verify()
    L = [f"## Chuỗi audit: {'✅ toàn vẹn' if res['valid'] else '❌ ĐÃ BỊ CAN THIỆP'}", "",
         f"- Số bản ghi: {res['n_entries']}", f"- Hash đầu chuỗi: `{res['head'][:32]}…`", ""]
    for p in res["problems"]:
        L.append(f"- ⚠️ {p}")
    entries = list(log.entries())[-12:]
    L += ["", "### 12 bản ghi gần nhất", "", "| # | Thời điểm | Loại | Kết cục | Hash |", "|---|---|---|---|---|"]
    for e in entries:
        L.append(f"| {e.index} | {e.timestamp[:19]} | {e.event_type} | "
                 f"{e.payload.get('outcome', '—')} | `{e.entry_hash[:12]}…` |")
    stats = _repo().decision_breakdown()
    L += ["", f"**Thống kê quyết định:** `{stats}`"]
    return "\n".join(L)


def build_ui() -> gr.Blocks:
    from app.ui import build_demo
    return build_demo(_service, _sku_choices, _lot_choices, intake, checkout, trace, set_status, audit_view)


if __name__ == "__main__":
    from app.ui import launch_options
    build_ui().launch(server_name=os.getenv('GRADIO_SERVER_NAME', '127.0.0.1'),
                      server_port=int(os.getenv('GRADIO_SERVER_PORT', '7860')), **launch_options())
