"""FastAPI service — lớp tích hợp với thiết bị quầy/POS.

Toàn bộ nghiệp vụ nằm ở ``rpv.services.checkout``; file này chỉ làm HTTP. Nhờ
vậy demo Gradio, CLI, test và API dùng CHUNG một logic, và số liệu trong luận
văn đo đúng thứ chạy trong sản phẩm.

Lưu ý về khởi tạo: encoder DINOv2 được nạp MỘT LẦN trong ``lifespan``. Nếu nạp
thất bại (thiếu torch, chưa có trọng số, không có mạng) thì service khởi động ở
trạng thái suy giảm và ``/health`` nói rõ vì sao, thay vì tiến trình chết ngay
lúc boot và người dùng chỉ thấy container restart liên tục.

Chạy: uvicorn api.main:app --reload  (mở http://127.0.0.1:8000/docs)
"""
from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from rpv import __version__
from rpv.audit.hash_chain import AuditLog
from rpv import config as app_config
from rpv.domain.models import (
    Evidence, EvidenceStatus, EvidenceType, Lot, LotStatus, Product, SKU, Supplier, TraceEvent,
)
from rpv.provenance.adapters import EpcisExporter
from rpv.provenance.repository import ProvenanceRepository, new_id
from rpv.services.checkout import CheckoutService
from rpv.util import json_safe, log
from rpv.vision.calibration import CalibrationModel
from rpv.vision.encoders import load_encoder
from rpv.vision.prototypes import PrototypeBank

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_config.settings.paths.ensure()
    repo = ProvenanceRepository(app_config.settings.paths.db_path)
    repo.init_schema()
    _state["repo"] = repo
    try:
        encoder = load_encoder()
        bank = PrototypeBank(encoder)
        if Path(app_config.settings.paths.bank_path).exists():
            bank.load(app_config.settings.paths.bank_path)
        calibration = None
        if Path(app_config.settings.paths.calib_path).exists():
            calibration = CalibrationModel.load(app_config.settings.paths.calib_path, encoder.tag)
        _state["encoder"] = encoder
        _state["bank"] = bank
        _state["calibration"] = calibration
        _state["service"] = CheckoutService(encoder, bank, repo, calibration,
                                            audit=AuditLog(app_config.settings.paths.audit_path), bank_path=app_config.settings.paths.bank_path,
                                            calibration_path=app_config.settings.paths.calib_path)
        _state['calibration'] = _state['service'].calibration
    except Exception as exc:                     # noqa: BLE001 - phải báo, không được chết
        _state["startup_error"] = f"{type(exc).__name__}: {exc}"
        log(f"[api] KHÔNG nạp được lớp thị giác: {_state['startup_error']}\n"
            "      Các endpoint /catalog và /provenance vẫn hoạt động; "
            "/vision và /checkout sẽ trả 503.")
    try:
        yield
    finally:
        repo.close()
        _state.clear()


app = FastAPI(
    title="RPV — Retail Provenance Vision API",
    version=__version__,
    description=(
        "Nhận diện sản phẩm bán lẻ không mã vạch (few-shot, open-set) GẮN với kiểm chứng nguồn gốc "
        "theo lô hàng. Hệ thống không suy đoán xuất xứ từ hình ảnh: mọi kết luận nguồn gốc đều phải "
        "truy về chứng từ kiểm chứng được, nếu không sẽ được báo là 'chưa xác minh'."
    ),
    lifespan=lifespan,
)


def get_service() -> CheckoutService:
    svc = _state.get("service")
    if svc is None:
        raise HTTPException(503, "Lớp thị giác chưa khởi tạo được: "
                                 + str(_state.get("startup_error", "chưa rõ nguyên nhân")))
    try:
        svc.refresh_catalog()
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(503, f'Không nạp được danh mục hiện hành: {exc}') from exc
    return svc


def get_repo() -> ProvenanceRepository:
    repo = _state.get("repo")
    if repo is None:
        raise HTTPException(503, "Repository chưa khởi tạo.")
    return repo


_ALLOWED_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_MAX_UPLOAD_BYTES = 15 * 1024 * 1024


def _save_upload(upload: UploadFile) -> Path:
    """Lưu file tải lên vào temp file, có kiểm tra đuôi.

    Đuôi được ép về danh sách trắng: tên file do client đặt, và ghép trực tiếp
    một suffix tuỳ ý vào đường dẫn hệ thống là thói quen nên bỏ.
    """
    suffix = Path(upload.filename or "upload.jpg").suffix.lower()
    if suffix not in _ALLOWED_SUFFIX:
        suffix = ".jpg"
    fd, name = tempfile.mkstemp(suffix=suffix)
    tmp = Path(name)
    try:
        total = 0
        with os.fdopen(fd, "wb") as fh:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Ảnh vượt giới hạn 15 MB.")
                fh.write(chunk)
        if total == 0:
            raise HTTPException(400, "File ảnh rỗng.")
        try:
            with Image.open(tmp) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(400, "File tải lên không phải ảnh hợp lệ.") from exc
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


# ---------- schema ----------
class SupplierIn(BaseModel):
    supplier_id: Optional[str] = None
    name: str
    address: str = ""
    gln: str = ""
    country: str = "VN"
    contact: str = ""
    certifications: list[str] = Field(default_factory=list)


class ProductIn(BaseModel):
    product_id: Optional[str] = None
    name: str
    category: str = "fresh_produce"
    brand: str = ""
    parent_category: str = ""
    description: str = ""
    image_uri: str = ""


class SkuIn(BaseModel):
    sku_id: str
    product_id: str
    name: str
    unit: str = "kg"
    price: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    gtin: str = ""
    is_barcodeless: bool = True


class LotIn(BaseModel):
    lot_id: Optional[str] = None
    sku_id: str
    supplier_id: str
    origin_country: str = "VN"
    origin_region: str = ""
    farm_or_facility: str = ""
    gln: str = ""
    production_date: Optional[str] = None
    received_date: Optional[str] = None
    expiry_date: Optional[str] = None
    quantity: float = 0.0
    unit: str = "kg"
    notes: str = ""


class EvidenceIn(BaseModel):
    lot_id: str
    evidence_type: EvidenceType = EvidenceType.DELIVERY_NOTE
    uri: str = ""
    sha256: str = ""
    issuer: str = ""
    issued_at: Optional[str] = None
    valid_until: Optional[str] = None
    status: EvidenceStatus = EvidenceStatus.PENDING
    verified_by: str = ""
    note: str = ""


class EventIn(BaseModel):
    lot_id: str
    biz_step: str = "receiving"
    disposition: str = "in_progress"
    event_time: Optional[str] = None
    location_gln: str = ""
    location_name: str = ""
    actor: str = ""
    payload: dict = Field(default_factory=dict)


class LotStatusIn(BaseModel):
    status: LotStatus
    note: str = ""


class EnrollmentIn(BaseModel):
    sku_id: str = Field(min_length=1, max_length=120)
    label: str = ''
    operator: str = ''


def enrollment_call(action):
    try:
        return action()
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post('/enrollment', tags=['enrollment'])
def create_enrollment(data: EnrollmentIn, service: CheckoutService = Depends(get_service)):
    return enrollment_call(lambda: service.enrollment.create(data.sku_id, data.label, data.operator))


@app.get('/enrollment', tags=['enrollment'])
def list_enrollments(service: CheckoutService = Depends(get_service)):
    return service.enrollment.list_sessions()


@app.get('/enrollment/{session_id}', tags=['enrollment'])
def get_enrollment(session_id: str, service: CheckoutService = Depends(get_service)):
    return enrollment_call(lambda: service.enrollment.get(session_id))


@app.post('/enrollment/{session_id}/images', tags=['enrollment'])
def add_enrollment_images(session_id: str, role: str = Form(...), group_id: str = Form(...),
                          files: list[UploadFile] = File(...), negative_sku: str = Form(''),
                          service: CheckoutService = Depends(get_service)):
    if not 1 <= len(files) <= 40:
        raise HTTPException(400, 'Cần 1..40 ảnh mỗi lượt.')
    paths = []
    try:
        for file in files:
            paths.append(_save_upload(file))
        return enrollment_call(lambda: service.enrollment.add_images(
            session_id, role, paths, group_id, negative_sku, [f.filename or 'upload' for f in files]))
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


@app.delete('/enrollment/{session_id}/support/{sample_id}', tags=['enrollment'])
def remove_enrollment_support(session_id: str, sample_id: str, service: CheckoutService = Depends(get_service)):
    return enrollment_call(lambda: service.enrollment.remove_support(session_id, sample_id))


@app.get('/enrollment/{session_id}/images/{sample_id}', tags=['enrollment'])
def enrollment_image(session_id: str, sample_id: str, service: CheckoutService = Depends(get_service)):
    return FileResponse(enrollment_call(lambda: service.enrollment.sample_path(session_id, sample_id)))


@app.post('/enrollment/{session_id}/assess', tags=['enrollment'])
def assess_enrollment(session_id: str, service: CheckoutService = Depends(get_service)):
    return enrollment_call(lambda: service.enrollment.assess(session_id))


@app.post('/enrollment/{session_id}/publish', tags=['enrollment'])
def publish_enrollment(session_id: str, service: CheckoutService = Depends(get_service)):
    return enrollment_call(lambda: service.publish_enrollment(session_id))


# ---------- hệ thống ----------
@app.get("/health", tags=["system"])
def health(response: Response) -> dict:
    service = _state.get('service')
    error = _state.get('startup_error')
    if service is not None:
        try:
            service.refresh_catalog()
        except (OSError, ValueError, RuntimeError) as exc:
            error = f'{type(exc).__name__}: {exc}'
    ready = service is not None and not error
    response.status_code = 200 if ready else 503
    encoder = service.encoder if service is not None else _state.get('encoder')
    bank = service.bank if service is not None else _state.get('bank')
    return {
        "status": "ok" if ready else "degraded",
        "version": __version__,
        "encoder": getattr(encoder, "tag", None),
        "encoder_info": encoder.describe() if encoder is not None else None,
        "catalog_size": len(bank) if bank is not None else 0,
        "calibrated": ready and service.calibration is not None,
        "error": error,
    }


@app.get("/status", tags=["system"])
def status(repo: ProvenanceRepository = Depends(get_repo)) -> dict:
    if _state.get('service') is not None:
        get_service()
    bank = _state.get("bank")
    calib = getattr(_state.get('service'), 'calibration', None)
    return {
        "version": __version__,
        "vision": bank.summary() if bank is not None else {"error": _state.get("startup_error")},
        "calibration": None if calib is None else {
            "alpha": calib.alpha if calib.decision_rule == 'conformal' else None,
            'decision_rule': calib.decision_rule,
            'risk_control': calib.risk_control,
            'ood': calib.ood,
            "global_threshold": calib.global_threshold,
            "n_per_class_thresholds": len(calib.per_class_threshold),
            "warnings": calib.warnings,
        },
        "database": repo.stats(),
        "decisions_by_outcome": repo.decision_breakdown(),
        "audit": AuditLog(app_config.settings.paths.audit_path).verify(),
        "external_adapters_enabled": app_config.settings.enable_external_adapters,
    }


# ---------- thị giác ----------
@app.post("/vision/register", tags=["vision"])
def register_product(
    sku_id: str = Form(...),
    label: str = Form(""),
    operator: str = Form(""),
    files: list[UploadFile] = File(...),
    service: CheckoutService = Depends(get_service),
) -> dict:
    """Đăng ký mặt hàng mới bằng 2-3 ảnh mẫu. Không train lại mô hình."""
    if not 1 <= len(files) <= 40:
        raise HTTPException(400, "Cần 1..40 ảnh mẫu (khuyến nghị 3-10).")
    tmps: list[Path] = []
    try:
        for file in files:
            tmps.append(_save_upload(file))
        return service.register_product(sku_id, [str(p) for p in tmps], label=label, operator=operator)
    finally:
        for p in tmps:
            p.unlink(missing_ok=True)


@app.post("/vision/recognize", tags=["vision"])
def recognize(file: UploadFile = File(...), service: CheckoutService = Depends(get_service)) -> dict:
    """Nhận diện tự do: trả về tập ứng viên, hoặc trạng thái 'unknown'."""
    tmp = _save_upload(file)
    try:
        # json_safe: khoảng cách/ngưỡng có thể là inf khi chưa hiệu chỉnh, và
        # ``Infinity`` không phải JSON hợp lệ nên client JS sẽ lỗi khi parse.
        return json_safe(service.recognizer.recognize(str(tmp)).to_json())
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/checkout", tags=["checkout"])
def checkout(
    file: UploadFile = File(...),
    claimed_sku: Optional[str] = Form(None),
    claimed_lot: Optional[str] = Form(None),
    claimed_price: Optional[float] = Form(None, ge=0, allow_inf_nan=False),
    operator: str = Form(""),
    enrich: bool = Form(False),
    service: CheckoutService = Depends(get_service),
) -> dict:
    """Một lượt quầy đầy đủ: thị giác -> hồ sơ lô -> policy -> audit."""
    tmp = _save_upload(file)
    try:
        resp = service.process(str(tmp), claimed_sku=claimed_sku, claimed_lot=claimed_lot,
                               claimed_price=claimed_price, operator=operator, enrich=enrich)
        payload = resp.to_json()
        payload["human_summary"] = resp.human_summary()
        return json_safe(payload)
    finally:
        tmp.unlink(missing_ok=True)


# ---------- danh mục & nguồn gốc ----------
@app.post("/catalog/suppliers", tags=["catalog"])
def create_supplier(body: SupplierIn, repo: ProvenanceRepository = Depends(get_repo)) -> dict:
    s = Supplier(supplier_id=body.supplier_id or new_id("sup"), **body.model_dump(exclude={"supplier_id"}))
    return repo.upsert_supplier(s).to_json()


@app.post("/catalog/products", tags=["catalog"])
def create_product(body: ProductIn, repo: ProvenanceRepository = Depends(get_repo)) -> dict:
    p = Product(product_id=body.product_id or new_id("prd"), **body.model_dump(exclude={"product_id"}))
    return repo.upsert_product(p).to_json()


@app.post("/catalog/skus", tags=["catalog"])
def create_sku(body: SkuIn, repo: ProvenanceRepository = Depends(get_repo)) -> dict:
    try:
        return repo.upsert_sku(SKU(**body.model_dump())).to_json()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/catalog/skus", tags=["catalog"])
def list_skus(active_only: bool = True, repo: ProvenanceRepository = Depends(get_repo)) -> list[dict]:
    bank = _state.get("bank")
    out = []
    for s in repo.list_skus(active_only):
        d = s.to_json()
        d["has_vision_prototype"] = (s.sku_id in bank) if bank is not None else None
        out.append(d)
    return out


@app.post("/provenance/lots", tags=["provenance"])
def create_lot(body: LotIn, repo: ProvenanceRepository = Depends(get_repo)) -> dict:
    if repo.get_sku(body.sku_id) is None:
        raise HTTPException(404, f"SKU {body.sku_id!r} không tồn tại.")
    lot = Lot(lot_id=body.lot_id or new_id("lot"), **body.model_dump(exclude={"lot_id"}))
    try:
        return repo.upsert_lot(lot).to_json()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/provenance/evidence", tags=["provenance"])
def add_evidence(body: EvidenceIn, repo: ProvenanceRepository = Depends(get_repo)) -> dict:
    if repo.get_lot(body.lot_id) is None:
        raise HTTPException(404, f"Lô {body.lot_id!r} không tồn tại.")
    data = body.model_dump()
    data["evidence_type"] = body.evidence_type.value
    data["status"] = body.status.value
    return repo.add_evidence(Evidence(evidence_id=new_id("ev"), **data)).to_json()


@app.post("/provenance/events", tags=["provenance"])
def add_event(body: EventIn, repo: ProvenanceRepository = Depends(get_repo)) -> dict:
    if repo.get_lot(body.lot_id) is None:
        raise HTTPException(404, f"Lô {body.lot_id!r} không tồn tại.")
    return repo.add_event(TraceEvent(event_id=new_id("evt"), **body.model_dump())).to_json()


@app.patch("/provenance/lots/{lot_id}/status", tags=["provenance"])
def set_lot_status(lot_id: str, body: LotStatusIn, repo: ProvenanceRepository = Depends(get_repo)) -> dict:
    if not repo.set_lot_status(lot_id, body.status.value, body.note):
        raise HTTPException(404, f"Lô {lot_id!r} không tồn tại.")
    AuditLog(app_config.settings.paths.audit_path).append("lot_status_changed",
                                               {"lot_id": lot_id, "status": body.status.value, "note": body.note})
    return {"lot_id": lot_id, "status": body.status.value}


@app.get("/provenance/trace/{lot_id}", tags=["provenance"])
def trace_lot(lot_id: str, epcis: bool = False, repo: ProvenanceRepository = Depends(get_repo)) -> dict:
    lot = repo.get_lot(lot_id)
    if lot is None:
        raise HTTPException(404, f"Lô {lot_id!r} không tồn tại.")
    rec = repo.resolve(lot.sku_id, lot_id)
    payload = rec.to_json()
    if epcis:
        payload["epcis_document"] = EpcisExporter.export(rec.events, lot.trace_code)
    return payload


@app.get("/audit/verify", tags=["system"])
def audit_verify() -> dict:
    return AuditLog(app_config.settings.paths.audit_path).verify()
