"""Demo Gradio — giao diện để bảo vệ khóa luận.

Các tab trình bày luồng vận hành trước hội đồng:

  1. Nhập hàng: thêm mặt hàng mới bằng 2-3 ảnh, KÈM hồ sơ lô. Đây là chỗ cho
     thấy nguồn gốc là dữ liệu được khai lúc nhập hàng, không phải thứ đoán ra
     từ ảnh.
  2. Quầy thanh toán: kiểm chứng lời khai và ra quyết định có mã lý do.
  3. Truy xuất: từ lô hàng ra chứng từ, sự kiện, và bản xuất EPCIS 2.0.
  4. Kiểm toán: phát hiện thay đổi làm đứt chuỗi hash của lịch sử quyết định.
  5. Đăng ký có kiểm tra: đối chiếu SKU dễ nhầm và hướng dẫn bổ sung ảnh.

Hệ thống được dựng LAZY (lần gọi đầu tiên), không phải lúc import: nạp DINOv2
mất vài giây và có thể thất bại, nên nó không được xảy ra như một tác dụng phụ
của câu ``import``. Trước đây chỉ cần import module là backbone đã bị nạp — làm
mọi công cụ đọc file này (kể cả pydoc) phải chờ, và làm lỗi cấu hình hiện ra ở
đúng chỗ khó hiểu nhất.

Chạy: python -m app.demo
"""
from __future__ import annotations

import os
import threading
from datetime import date, timedelta
from pathlib import Path

import gradio as gr

from rpv.audit.hash_chain import AuditLog
from rpv import config as app_config
from rpv.domain.models import (
    Evidence, EvidenceStatus, EvidenceType, Lot, LotStatus, Product, SKU, Supplier, TraceEvent,
)
from rpv.provenance.adapters import EpcisExporter
from rpv.provenance.repository import ProvenanceRepository, new_id
from rpv.services.checkout import CheckoutService
from rpv.util import json_dumps, stable_hash_hex
from rpv.vision.calibration import CalibrationModel
from rpv.vision.encoders import load_encoder
from rpv.vision.prototypes import PrototypeBank

OUTCOME_STYLE = {
    "allow": ("✅", "CHO QUA"),
    "allow_warning": ("⚠️", "CHO QUA — CÓ CẢNH BÁO"),
    "assist": ("🙋", "CẦN NHÂN VIÊN XÁC NHẬN"),
    "block": ("⛔", "CHẶN GIAO DỊCH"),
}


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
                              audit=AuditLog(app_config.settings.paths.audit_path), bank_path=app_config.settings.paths.bank_path)
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


def _lot_choices() -> list[str]:
    return [f"{r['lot_id']} ({r['sku_id']}, {r['status']})"
            for r in _repo().list_lot_summaries()]


def _lot_id(choice: str | None) -> str | None:
    return choice.split(" ")[0] if choice else None


# ---------- Tab 1: nhập hàng ----------
def intake(sku_id, sku_name, price, category, supplier_name, supplier_addr,
           origin_country, origin_region, farm, production_date, expiry_date,
           evidence_types, evidence_verified, images):
    if not sku_id:
        return "Cần mã mặt hàng.", gr.update(), gr.update()
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
        return "Chưa có ảnh.", "{}"
    path = image if isinstance(image, str) else image.name
    resp = _service().process(path, claimed_sku=(claimed_sku if use_claim else None),
                           claimed_lot=_lot_id(claimed_lot) if use_claim else None,
                           claimed_price=float(price) if price else None, operator="staff_demo")
    icon, title = OUTCOME_STYLE.get(resp.outcome, ("•", resp.outcome))
    L = [f"## {icon} {title}", "", f"**Xuất xứ:** {resp.origin_summary}",
         f"**Trạng thái hồ sơ:** `{resp.provenance_status}`", ""]
    if resp.messages:
        L.append("**Lý do:**")
        L += [f"- {m}" for m in resp.messages]
    else:
        L.append("Không có cảnh báo nào.")
    L += ["", f"**Mã lý do:** `{', '.join(resp.reasons)}`"]
    rec = resp.recognition or {}
    if rec.get("candidates"):
        L += ["", "**Ứng viên thị giác:**", "", "| Mặt hàng | Khoảng cách | Trong tập hợp lệ |", "|---|---|---|"]
        for c in rec["candidates"]:
            L.append(f"| {c['label']} | {c['distance']:.4f} | {'có' if c['in_prediction_set'] else 'không'} |")
    ver = resp.verification or {}
    if ver:
        L += ["", f"**Kiểm chứng lời khai:** `{ver.get('status')}` "
                  f"(khoảng cách {ver.get('claimed_distance', 0):.4f} / ngưỡng {ver.get('threshold', 0):.4f})"]
    L += ["", f"_Audit #{resp.audit.get('index')} · hash `{str(resp.audit.get('entry_hash'))[:16]}…`_"]
    return "\n".join(L), json_dumps(resp.to_json())


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
    L += ["", ("✅ Đủ trường truy xuất tối thiểu (Thông tư 02/2024/TT-BKHCN)."
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
    st = state()
    encoder, bank, calib = st["encoder"], st["bank"], st["calib"]
    cal_note = ("⚠️ **Chưa hiệu chỉnh.** Chạy `python -m rpv.cli calibrate` để hệ thống biết cách "
                "từ chối hàng lạ. Khi chưa hiệu chỉnh, cần nhân viên xác nhận."
                if calib is None else
                f"Đã hiệu chỉnh · alpha={calib.alpha:g} · ngưỡng toàn cục={calib.global_threshold:.4f} "
                f"· ngưỡng riêng cho {len(calib.per_class_threshold)} mặt hàng")

    with gr.Blocks(title="RPV — Nhận diện & kiểm chứng nguồn gốc bán lẻ",
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# RPV — Nhận diện sản phẩm không mã vạch + Kiểm chứng nguồn gốc\n"
            "**Ảnh trả lời _đây là gì_. Hồ sơ lô trả lời _nó từ đâu_. "
            "Hệ thống chỉ cho bán khi hai câu trả lời khớp nhau.**\n\n"
            f"Encoder: `{encoder.tag}` · Danh mục: {len(bank)} mặt hàng · {cal_note}"
        )
        from app.enrollment import build_enrollment_tab
        build_enrollment_tab(_service)
        with gr.Tab("1. Nhập hàng (few-shot + hồ sơ lô)"):
            gr.Markdown("Thêm mặt hàng mới bằng vài ảnh mẫu **và** khai hồ sơ nguồn gốc của lô. "
                        "Đây là nơi dữ liệu xuất xứ thực sự được sinh ra.")
            with gr.Row():
                with gr.Column():
                    sku_id = gr.Textbox(label="Mã mặt hàng (SKU)", placeholder="xoai_cat_hoa_loc")
                    sku_name = gr.Textbox(label="Tên hiển thị", placeholder="Xoài cát Hòa Lộc")
                    price = gr.Number(label="Giá (VND/kg)", value=42000)
                    category = gr.Dropdown(["fresh_produce", "packaged", "default"],
                                           value="fresh_produce", label="Nhóm hàng")
                    images = gr.File(label="Ảnh mẫu (bỏ trống nếu SKU đã đăng ký có kiểm tra)", file_count="multiple",
                                     file_types=["image"])
                with gr.Column():
                    supplier_name = gr.Textbox(label="Nhà cung cấp", placeholder="HTX Cao Lãnh")
                    supplier_addr = gr.Textbox(label="Địa chỉ NCC", placeholder="Đồng Tháp")
                    origin_country = gr.Dropdown(["VN", "US", "CN", "TH", "AU", "NZ", "JP", "KR", "UNKNOWN"],
                                                 value="VN", label="Nước xuất xứ")
                    origin_region = gr.Textbox(label="Vùng trồng/sản xuất", placeholder="Cao Lãnh, Đồng Tháp")
                    farm = gr.Textbox(label="Nông trại/cơ sở đóng gói", placeholder="Nông trại Bảy Núi")
                    production_date = gr.Textbox(label="Ngày sản xuất/thu hoạch (YYYY-MM-DD)",
                                                 value=str(date.today() - timedelta(days=2)))
                    expiry_date = gr.Textbox(label="Hạn sử dụng (YYYY-MM-DD)",
                                             value=str(date.today() + timedelta(days=20)))
                    evidence_types = gr.CheckboxGroup(
                        [e.value for e in EvidenceType], label="Chứng từ kèm theo",
                        value=[EvidenceType.DELIVERY_NOTE.value, EvidenceType.SUPPLIER_DECLARATION.value])
                    evidence_verified = gr.Checkbox(label="Đã đối chiếu bản gốc (verified)", value=True)
            btn = gr.Button("Đăng ký mặt hàng + lô hàng", variant="primary")
            intake_out = gr.Markdown()

        with gr.Tab("2. Quầy thanh toán"):
            gr.Markdown("Chế độ kiểm chứng: khai mặt hàng rồi để hệ thống xác nhận hoặc phản đối. "
                        "Thử khai sai mặt hàng để thấy hệ thống bắt được hành vi đổi mã.")
            with gr.Row():
                with gr.Column():
                    co_image = gr.Image(label="Ảnh tại quầy", type="filepath")
                    use_claim = gr.Checkbox(label="Có khai báo mặt hàng (chế độ kiểm chứng)", value=True)
                    co_sku = gr.Dropdown(_sku_choices(), label="Mặt hàng khai báo", allow_custom_value=True)
                    co_lot = gr.Dropdown(_lot_choices(), label="Lô hàng (bỏ trống = tự chọn lô active)",
                                         allow_custom_value=True)
                    co_price = gr.Number(label="Giá khai báo (VND)", value=None)
                    co_btn = gr.Button("Kiểm tra và quyết định", variant="primary")
                    refresh_catalog_btn = gr.Button('Tải lại mặt hàng và lô')
                with gr.Column():
                    co_out = gr.Markdown()
                    co_json = gr.Code(label="Phản hồi đầy đủ (JSON)", language="json")

        with gr.Tab("3. Truy xuất nguồn gốc"):
            with gr.Row():
                with gr.Column():
                    tr_lot = gr.Dropdown(_lot_choices(), label="Chọn lô")
                    tr_epcis = gr.Checkbox(label="Kèm bản xuất EPCIS 2.0 (JSON-LD)", value=False)
                    tr_btn = gr.Button("Xem hồ sơ", variant="primary")
                    gr.Markdown("---\n**Mô phỏng thu hồi/tạm giữ** để thấy policy phản ứng:")
                    st_status = gr.Dropdown([s.value for s in LotStatus],
                                            value=LotStatus.RECALLED.value, label="Trạng thái mới")
                    st_note = gr.Textbox(label="Lý do", placeholder="Phát hiện dư lượng vượt ngưỡng")
                    st_btn = gr.Button("Cập nhật trạng thái lô")
                    st_out = gr.Markdown()
                with gr.Column():
                    tr_out = gr.Markdown()
                    tr_json = gr.Code(label="Hồ sơ (JSON)", language="json")

        with gr.Tab("4. Kiểm toán"):
            gr.Markdown("Mọi quyết định đều được ghi vào chuỗi hash. Sửa hay xoá một dòng ở giữa "
                        "sẽ làm đứt chuỗi và bị phát hiện — đó là điều kiện cần để hệ thống "
                        "kiểm chứng nguồn gốc đáng tin.")
            au_btn = gr.Button("Kiểm tra toàn vẹn chuỗi", variant="primary")
            au_out = gr.Markdown()

        btn.click(intake,
                  [sku_id, sku_name, price, category, supplier_name, supplier_addr, origin_country,
                   origin_region, farm, production_date, expiry_date, evidence_types,
                   evidence_verified, images],
                  [intake_out, co_sku, co_lot])
        co_btn.click(checkout, [co_image, co_sku, co_lot, co_price, use_claim], [co_out, co_json])
        refresh_catalog_btn.click(lambda: (gr.update(choices=_sku_choices()), gr.update(choices=_lot_choices())),
                                  None, [co_sku, co_lot])
        tr_btn.click(trace, [tr_lot, tr_epcis], [tr_out, tr_json])
        st_btn.click(set_status, [tr_lot, st_status, st_note], [st_out, tr_lot])
        au_btn.click(audit_view, None, au_out)
        return demo


if __name__ == "__main__":
    build_ui().launch(server_name=os.getenv('GRADIO_SERVER_NAME', '127.0.0.1'), server_port=7860)
