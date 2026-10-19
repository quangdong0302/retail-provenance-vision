"""Orchestrator: một lượt quầy thanh toán từ đầu đến cuối.

    ảnh (+ mặt hàng khai báo) -> thị giác -> hồ sơ lô -> policy -> audit -> phản hồi

Toàn bộ nghiệp vụ nằm ở lớp này, KHÔNG nằm trong API hay giao diện. Nhờ vậy cùng
một logic được dùng cho FastAPI, demo Gradio, CLI và test tự động — và số liệu
trong luận văn đo đúng thứ chạy ở demo.
"""
from __future__ import annotations

import uuid
from time import perf_counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rpv.audit.hash_chain import AuditLog
from rpv import config as app_config
from rpv.domain.models import DecisionRecord, TraceEvent
from rpv.provenance.adapters import NullExternalAdapter, OpenFoodFactsAdapter
from rpv.provenance.policy import PolicyDecision, PolicyEngine
from rpv.provenance.repository import ProvenanceRecord, ProvenanceRepository, new_id
from rpv.vision.calibration import CalibrationModel
from rpv.vision.encoders import BaseEncoder, image_sha256
from rpv.vision.prototypes import PrototypeBank
from rpv.vision.recognizer import Recognizer
from rpv.vision.verification import Verifier, VerificationResult


@dataclass
class CheckoutResponse:
    request_id: str
    outcome: str
    reasons: list[str]
    messages: list[str]
    origin_summary: str
    provenance_status: str
    requires_operator: bool
    recognition: Optional[dict] = None
    verification: Optional[dict] = None
    provenance: Optional[dict] = None
    enrichment: Optional[dict] = None
    audit: dict = field(default_factory=dict)
    created_at: str = ""
    decision_state: str = 'abstain'
    timings_ms: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "request_id": self.request_id,
            "outcome": self.outcome,
            "reasons": self.reasons,
            "messages": self.messages,
            "origin_summary": self.origin_summary,
            "provenance_status": self.provenance_status,
            "requires_operator": self.requires_operator,
            "recognition": self.recognition,
            "verification": self.verification,
            "provenance": self.provenance,
            "enrichment": self.enrichment,
            "audit": self.audit,
            "created_at": self.created_at,
            'decision_state': self.decision_state,
            'timings_ms': self.timings_ms,
        }

    def human_summary(self) -> str:
        """Bản tóm tắt cho nhân viên thu ngân: nói lý do, không nói con số mô hình."""
        head = {
            "allow": "CHO QUA",
            "allow_warning": "CHO QUA — CÓ CẢNH BÁO",
            "assist": "CẦN NHÂN VIÊN XÁC NHẬN",
            "block": "CHẶN GIAO DỊCH",
        }.get(self.outcome, self.outcome.upper())
        lines = [f"[{head}]", f"Xuất xứ: {self.origin_summary}"]
        lines += [f"- {m}" for m in self.messages] or ["- Không có cảnh báo nào."]
        return "\n".join(lines)


class CheckoutService:
    def __init__(self,
                 encoder: BaseEncoder,
                 bank: PrototypeBank,
                 repo: ProvenanceRepository,
                 calibration: CalibrationModel | None = None,
                 policy: PolicyEngine | None = None,
                 audit: AuditLog | None = None,
                 external_adapter: Any | None = None,
                 bank_path: str | Path | None = None,
                 calibration_path: str | Path | None = None) -> None:
        self.encoder = encoder
        self.bank = bank
        self.repo = repo
        self.bank_path = Path(bank_path) if bank_path else None
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self._calibration_stamp = -1
        # The caller may have loaded a bank just before another process saved it.
        # Force a refresh on first use instead of stamping possibly stale memory.
        self._bank_stamp = -1 if self.bank_path and self.bank_path.exists() else None
        if calibration and calibration.bank_fingerprint and calibration.bank_fingerprint != bank.fingerprint():
            calibration = None
        self.calibration = calibration
        self.recognizer = Recognizer(encoder, bank, calibration)
        self.verifier = Verifier(encoder, bank, calibration) if calibration else None
        self.policy = policy or PolicyEngine()
        # An empty AuditLog is falsey (__len__ == 0), but is still an explicitly
        # supplied destination. Falling back here can send test events to live logs.
        self.audit = audit if audit is not None else AuditLog(app_config.settings.paths.audit_path)
        self._enrollment = None
        if external_adapter is not None:
            self.external = external_adapter
        else:
            self.external = OpenFoodFactsAdapter() if app_config.settings.enable_external_adapters else NullExternalAdapter()

    # -- API chính ---------------------------------------------------------
    def process(self,
                image_path: str | Path,
                claimed_sku: str | None = None,
                claimed_lot: str | None = None,
                claimed_price: float | None = None,
                operator: str = "",
                enrich: bool = False,
                ) -> CheckoutResponse:
        """Một lượt quầy.

        ``claimed_sku=None`` là chế độ nhận diện tự do (hệ thống tự đề xuất mặt
        hàng). Có ``claimed_sku`` là chế độ kiểm chứng lời khai — chế độ đúng của
        quầy tự thanh toán và là chế độ được đánh giá chính trong luận văn.
        """
        started = perf_counter()
        self.refresh_catalog()
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        verification: Optional[VerificationResult] = None

        vision_started = perf_counter()
        embedding = self.encoder.embed_path(image_path)
        img_hash = image_sha256(image_path)

        with self.bank._lock:
            self.refresh_catalog()
            recognition = self.recognizer.recognize_embedding(embedding, img_hash)
            if claimed_sku and self.verifier is not None:
                verification = self.verifier.verify_embedding(embedding, claimed_sku, img_hash)
            resolved_sku = claimed_sku or (recognition.prediction_set[0] if recognition.is_confident
                                          else (recognition.top1.sku_id if recognition.top1 else ''))
            review_skus = set(recognition.prediction_set) | {resolved_sku}
            enrollment_review = any(s in self.bank.entries and self.bank.entries[s].requires_operator
                                    for s in review_skus)
            calibration_alpha = self.calibration.alpha if self.calibration else None
        vision_ms = (perf_counter() - vision_started) * 1000
        policy_started = perf_counter()

        # SKU dùng để tra hồ sơ: ưu tiên lời khai (vì hồ sơ gắn với lời khai,
        # và chúng ta đang kiểm chứng lời khai đó), nếu không có thì lấy top-1.
        record: Optional[ProvenanceRecord] = None
        if resolved_sku:
            record = self.repo.resolve(resolved_sku, claimed_lot)

        decision: PolicyDecision = self.policy.decide(
            recognition=recognition, verification=verification, record=record,
            claimed_price=claimed_price,
        )
        if enrollment_review:
            if decision.outcome != 'block':
                decision.outcome = 'assist'
            decision.requires_operator = True
            decision.reasons.append('enrollment_review_required')
            decision.messages.append('Mặt hàng mới cần nhân viên xác nhận; kiểm tra đăng ký chưa cho phép bán tự động.')
            decision.decision_state = 'abstain'
        policy_ms = (perf_counter() - policy_started) * 1000

        enrichment = None
        if enrich and record and record.sku and record.sku.gtin:
            res = self.external.enrich_by_gtin(record.sku.gtin)
            enrichment = res.to_json()
            if record.lot:
                enrichment["origin_cross_check"] = OpenFoodFactsAdapter.cross_check_origin(
                    record.lot.origin_country, res)

        # Audit + persistence
        record_row = DecisionRecord(
            request_id=request_id, created_at=created_at, outcome=decision.outcome,
            reasons=decision.reasons, messages=decision.messages, image_sha256=img_hash,
            claimed_sku=claimed_sku or "", claimed_lot=claimed_lot or "",
            resolved_sku=resolved_sku, resolved_lot=(record.lot.lot_id if record and record.lot else ""),
            vision_status=recognition.status if recognition else "",
            vision_top1=(recognition.top1.sku_id if recognition and recognition.top1 else ""),
            vision_distance=(recognition.top1.distance if recognition and recognition.top1 else None),
            verification_status=verification.status if verification else "",
            provenance_status=decision.provenance_status,
            encoder_tag=self.encoder.tag,
            calibration_alpha=calibration_alpha,
            operator=operator,
            payload={"audit_notes": decision.audit_notes, 'decision_state': decision.decision_state,
                     'timings_ms': {'vision_search': vision_ms, 'policy_lookup': policy_ms}},
        )
        self.repo.record_decision(record_row)
        entry = self.audit.append("checkout_decision", {
            "request_id": request_id, "image_sha256": img_hash, "outcome": decision.outcome,
            "reasons": decision.reasons, "claimed_sku": claimed_sku or "",
            "resolved_lot": record_row.resolved_lot, "encoder_tag": self.encoder.tag,
            "operator": operator,
            'decision_state': decision.decision_state,
            'decision_inputs': decision.audit_notes.get('decision_inputs', {}),
        })

        # Ghi sự kiện bán hàng vào chuỗi truy xuất khi giao dịch được cho qua.
        if decision.outcome in ("allow", "allow_warning") and record and record.lot:
            self.repo.add_event(TraceEvent(
                event_id=new_id("evt"), lot_id=record.lot.lot_id, biz_step="retail_selling",
                disposition="in_transit", location_name="POS", actor=operator or "self_checkout",
                payload={"request_id": request_id, "decision": decision.outcome},
            ))

        return CheckoutResponse(
            request_id=request_id, outcome=decision.outcome, reasons=decision.reasons,
            messages=decision.messages, origin_summary=decision.origin_summary,
            provenance_status=decision.provenance_status, requires_operator=decision.requires_operator,
            recognition=recognition.to_json() if recognition else None,
            verification=verification.to_json() if verification else None,
            provenance=record.to_json() if record else None,
            enrichment=enrichment,
            audit={"index": entry.index, "entry_hash": entry.entry_hash, "prev_hash": entry.prev_hash},
            created_at=created_at,
            decision_state=decision.decision_state,
            timings_ms={'vision_search': vision_ms, 'policy_lookup': policy_ms,
                        'total': (perf_counter() - started) * 1000},
        )

    # -- đăng ký sản phẩm mới ---------------------------------------------
    def refresh_catalog(self):
        validate = getattr(self.encoder, 'validate_artifacts', None)
        if validate is not None:
            validate()
        with self.bank._lock:
            self._refresh_catalog()

    def _refresh_catalog(self):
        calibration_stamp = (self.calibration_path.stat().st_mtime_ns
                             if self.calibration_path and self.calibration_path.exists() else None)
        calibration_changed = self.calibration_path is not None and calibration_stamp != self._calibration_stamp
        if not self.bank_path:
            bank_changed = False
        elif not self.bank_path.exists():
            if self._bank_stamp is not None:
                raise FileNotFoundError(f'Danh mục đang dùng đã bị xóa: {self.bank_path}')
            bank_changed = False
        else:
            stamp = self.bank_path.stat().st_mtime_ns
            bank_changed = stamp != self._bank_stamp
        if not bank_changed and not calibration_changed:
            return
        with self.bank._lock:
            if bank_changed:
                self.bank.load(self.bank_path)
            if calibration_changed:
                self.calibration = (CalibrationModel.load(self.calibration_path, self.encoder.tag)
                                    if calibration_stamp is not None else None)
            if self.calibration and self.calibration.bank_fingerprint != self.bank.fingerprint():
                self.calibration = None
            self.recognizer = Recognizer(self.encoder, self.bank, self.calibration)
            self.verifier = Verifier(self.encoder, self.bank, self.calibration) if self.calibration else None
            if bank_changed:
                self._bank_stamp = stamp
            if calibration_changed:
                self._calibration_stamp = calibration_stamp

    @property
    def enrollment(self):
        self.refresh_catalog()
        if self._enrollment is None:
            from rpv.services.enrollment import EnrollmentService
            parent = (self.bank_path or Path(app_config.settings.paths.bank_path)).parent
            self._enrollment = EnrollmentService(self.encoder, self.bank, parent / 'enrollment')
        return self._enrollment

    def publish_enrollment(self, session_id):
        def publish(entry, report):
            path = self.bank_path or Path(app_config.settings.paths.bank_path)
            with self.bank.storage_lock(path):
                backup = None
                self.refresh_catalog()
                if report['bank_fingerprint'] != self.bank.fingerprint():
                    raise ValueError('Danh mục đã đổi; chạy kiểm tra hồ sơ lại trước khi đăng ký.')
                if path.exists():
                    disk = PrototypeBank(self.encoder).load(path)
                    if disk.fingerprint() != self.bank.fingerprint():
                        raise ValueError('Danh mục trên đĩa đã đổi. Khởi động lại ứng dụng rồi kiểm tra hồ sơ lại.')
                    backup = path.parent / 'bank_backups' / f'{uuid.uuid4().hex}.json'
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.write_bytes(path.read_bytes())
                previous = self.bank.entries.get(entry.sku_id)
                try:
                    self.bank.entries[entry.sku_id] = entry
                    self.bank.save(path)
                except BaseException:
                    if previous is None:
                        self.bank.entries.pop(entry.sku_id, None)
                    else:
                        self.bank.entries[entry.sku_id] = previous
                    raise
                self.calibration = None
                self.bank_path = path
                self._bank_stamp = path.stat().st_mtime_ns
                self.recognizer = Recognizer(self.encoder, self.bank, None)
                self.verifier = None
                self.audit.append('guided_enrollment_published', {
                    'sku_id': entry.sku_id, 'session_id': entry.enrollment_id,
                    'revision': report['revision'], 'assessment_status': report['status'],
                    'operating_mode': 'operator_required', 'requires_recalibration': True,
                })
                return {'bank_backup': str(backup) if backup else None, 'requires_recalibration': True,
                        'operating_mode': 'operator_required'}
        return self.enrollment.publish(session_id, publish)

    def register_product(self, sku_id: str, image_paths: list[str], label: str = "",
                         product_id: str = "", operator: str = "",
                         bank_path: str | Path | None = None) -> dict:
        """Legacy registration. Recalibration is required before automatic decisions."""
        destination = Path(bank_path or self.bank_path or app_config.settings.paths.bank_path)
        with self.bank.storage_lock(destination):
            self.refresh_catalog()
            if destination.exists():
                # Reload inside the process lock: another API/demo may have
                # registered a SKU since this service last read the file.
                self.bank.load(destination)
            previous = self.bank.entries.get(sku_id)
            try:
                entry = self.bank.register(sku_id, image_paths, label=label, product_id=product_id)
                entry.requires_operator = True
                self.bank.save(destination)
            except BaseException:
                if previous is None:
                    self.bank.entries.pop(sku_id, None)
                else:
                    self.bank.entries[sku_id] = previous
                raise
            # Record this write's timestamp while still holding the process lock.
            # Otherwise another writer could update disk before we stat it,
            # making old memory look up to date until the next registration.
            self.bank_path = destination
            self._bank_stamp = destination.stat().st_mtime_ns if destination.exists() else None
            self.calibration = None
            self.recognizer = Recognizer(self.encoder, self.bank, None)
            self.verifier = None
        audit = self.audit.append("prototype_registered", {
            "sku_id": sku_id, "n_shots": entry.n_shots, "image_hashes": entry.image_hashes,
            "encoder_tag": self.encoder.tag, "operator": operator,
        })
        return {
            "sku_id": sku_id, "n_shots": entry.n_shots, "n_centroids": int(entry.centroids.shape[0]),
            "intra_spread": entry.intra_spread, "catalog_size": len(self.bank),
            "audit_index": audit.index,
            "encoder_tag": self.encoder.tag,
            'requires_recalibration': True,
            "note": "Đăng ký bằng few-shot: không train lại mô hình. "
                    "Muốn bán được thì lô hàng vẫn phải có hồ sơ nguồn gốc.",
        }
