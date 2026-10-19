"""Đọc và ghi hồ sơ truy xuất bằng SQLite.

Sử dụng sqlite3 có sẵn trong Python. Lược đồ được lưu trực tiếp tại
``rpv/domain/schema.sql`` để đọc và kiểm tra các bảng, khóa và ràng buộc.

FastAPI và Gradio có thể gọi repository từ nhiều thread. Kết nối dùng
``check_same_thread=False`` và các truy cập đi qua RLock. Cấu hình mặc
định của sqlite3 chỉ cho phép dùng kết nối trong thread đã tạo, nên
truy cập từ thread khác sẽ gây ProgrammingError. Khóa giúp tuần tự hóa
truy cập dùng chung kết nối; SQLite vẫn ghi tuần tự.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from rpv.domain.models import (
    Evidence, Lot, LotStatus, Product, SKU, Supplier,
    TraceEvent, DecisionRecord, TRACEABILITY_REQUIRED_FIELDS,
)

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "domain" / "schema.sql"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class ProvenanceRecord:
    """Hồ sơ nguồn gốc đã lắp ghép của một lô — thứ hiển thị cho người dùng."""
    sku: Optional[SKU] = None
    product: Optional[Product] = None
    lot: Optional[Lot] = None
    supplier: Optional[Supplier] = None
    evidence: list[Evidence] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.lot is not None

    def verified_evidence_types(self, today: date | None = None) -> set[str]:
        today = today or date.today()
        return {e.evidence_type for e in self.evidence if e.is_currently_valid(today)}

    def missing_traceability_fields(self) -> list[str]:
        """Đối chiếu bộ trường tối thiểu theo Thông tư 02/2024/TT-BKHCN."""
        lot, sku, product, sup = self.lot, self.sku, self.product, self.supplier
        present = {
            "product_name": bool(product and product.name),
            "product_image": bool(product and product.image_uri),
            "producer_name": bool(sup and sup.name),
            "producer_address": bool(sup and sup.address),
            "production_time": bool(lot and (lot.production_date or lot.received_date)),
            "trace_code": bool(lot and lot.trace_code),
            "brand_or_code": bool((product and product.brand) or (sku and (sku.gtin or sku.sku_id))),
        }
        return [f for f in TRACEABILITY_REQUIRED_FIELDS if not present.get(f, False)]

    def to_json(self) -> dict:
        return {
            "sku": self.sku.to_json() if self.sku else None,
            "product": self.product.to_json() if self.product else None,
            "lot": self.lot.to_json() if self.lot else None,
            "supplier": self.supplier.to_json() if self.supplier else None,
            "evidence": [e.to_json() for e in self.evidence],
            "events": [e.to_json() for e in self.events],
            "missing_traceability_fields": self.missing_traceability_fields(),
        }


class ProvenanceRepository:
    def __init__(self, db_path: str | Path, timeout: float = 15.0) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self.conn = sqlite3.connect(str(self.db_path), timeout=timeout, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        # WAL: đọc không bị chặn bởi ghi. Ở quầy, việc tra hồ sơ phải chạy được
        # trong lúc một giao dịch khác đang ghi quyết định.
        try:
            self.conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.DatabaseError:            # vài filesystem không hỗ trợ WAL
            pass
        self.conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")

    # -- vòng đời ----------------------------------------------------------
    def _write(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            try:
                cur = self.conn.execute(sql, params)
                self.conn.commit()
                return cur
            except BaseException:
                # A rejected foreign key/constraint leaves SQLite's implicit
                # transaction open. Release its write lock before another
                # API/demo connection tries to write.
                self.conn.rollback()
                raise

    def _rows(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def _one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    def init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self.conn.close()
                self._closed = True

    def __enter__(self) -> "ProvenanceRepository":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- ghi ---------------------------------------------------------------
    def upsert_supplier(self, s: Supplier) -> Supplier:
        self._write(
            """INSERT INTO suppliers(supplier_id,name,address,gln,country,contact,certifications)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(supplier_id) DO UPDATE SET
                 name=excluded.name, address=excluded.address, gln=excluded.gln,
                 country=excluded.country, contact=excluded.contact,
                 certifications=excluded.certifications""",
            (s.supplier_id, s.name, s.address, s.gln, s.country, s.contact, json.dumps(s.certifications)),
        )
        return s

    def upsert_product(self, p: Product) -> Product:
        self._write(
            """INSERT INTO products(product_id,name,category,brand,parent_category,description,image_uri)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(product_id) DO UPDATE SET
                 name=excluded.name, category=excluded.category, brand=excluded.brand,
                 parent_category=excluded.parent_category, description=excluded.description,
                 image_uri=excluded.image_uri""",
            (p.product_id, p.name, p.category, p.brand, p.parent_category, p.description, p.image_uri),
        )
        return p

    def upsert_sku(self, s: SKU) -> SKU:
        problems = s.validate()
        if problems:
            raise ValueError(f"SKU {s.sku_id} không hợp lệ: {'; '.join(problems)}")
        self._write(
            """INSERT INTO skus(sku_id,product_id,name,unit,price,gtin,requires_weighing,is_barcodeless,active)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(sku_id) DO UPDATE SET
                 product_id=excluded.product_id, name=excluded.name, unit=excluded.unit,
                 price=excluded.price, gtin=excluded.gtin, requires_weighing=excluded.requires_weighing,
                 is_barcodeless=excluded.is_barcodeless, active=excluded.active""",
            (s.sku_id, s.product_id, s.name, s.unit, s.price, s.gtin,
             int(s.requires_weighing), int(s.is_barcodeless), int(s.active)),
        )
        return s

    def upsert_lot(self, lot: Lot) -> Lot:
        problems = lot.validate()
        if problems:
            raise ValueError(f"Lô {lot.lot_id} không hợp lệ: {'; '.join(problems)}")
        if not lot.trace_code:
            lot.trace_code = f"RPV.{lot.sku_id}.{lot.lot_id}"
        self._write(
            """INSERT INTO lots(lot_id,sku_id,supplier_id,origin_country,origin_region,farm_or_facility,
                                gln,trace_code,production_date,received_date,expiry_date,quantity,unit,status,notes)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(lot_id) DO UPDATE SET
                 sku_id=excluded.sku_id, supplier_id=excluded.supplier_id,
                 origin_country=excluded.origin_country, origin_region=excluded.origin_region,
                 farm_or_facility=excluded.farm_or_facility, gln=excluded.gln,
                 trace_code=excluded.trace_code, production_date=excluded.production_date,
                 received_date=excluded.received_date, expiry_date=excluded.expiry_date,
                 quantity=excluded.quantity, unit=excluded.unit, status=excluded.status,
                 notes=excluded.notes""",
            (lot.lot_id, lot.sku_id, lot.supplier_id, lot.origin_country, lot.origin_region,
             lot.farm_or_facility, lot.gln, lot.trace_code, lot.production_date, lot.received_date,
             lot.expiry_date, lot.quantity, lot.unit, lot.status, lot.notes),
        )
        return lot

    def add_evidence(self, e: Evidence) -> Evidence:
        self._write(
            """INSERT INTO evidence(evidence_id,lot_id,evidence_type,uri,sha256,issuer,issued_at,
                                    valid_until,status,verified_by,verified_at,note)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(evidence_id) DO UPDATE SET
                 evidence_type=excluded.evidence_type, uri=excluded.uri, sha256=excluded.sha256,
                 issuer=excluded.issuer, issued_at=excluded.issued_at, valid_until=excluded.valid_until,
                 status=excluded.status, verified_by=excluded.verified_by,
                 verified_at=excluded.verified_at, note=excluded.note""",
            (e.evidence_id, e.lot_id, e.evidence_type, e.uri, e.sha256, e.issuer, e.issued_at,
             e.valid_until, e.status, e.verified_by, e.verified_at, e.note),
        )
        return e

    def add_event(self, ev: TraceEvent) -> TraceEvent:
        if ev.event_time is None:
            ev.event_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._write(
            """INSERT INTO trace_events(event_id,lot_id,biz_step,disposition,event_time,
                                        location_gln,location_name,actor,payload)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(event_id) DO UPDATE SET
                 biz_step=excluded.biz_step, disposition=excluded.disposition,
                 event_time=excluded.event_time, location_gln=excluded.location_gln,
                 location_name=excluded.location_name, actor=excluded.actor, payload=excluded.payload""",
            (ev.event_id, ev.lot_id, ev.biz_step, ev.disposition, ev.event_time,
             ev.location_gln, ev.location_name, ev.actor, json.dumps(ev.payload, ensure_ascii=False)),
        )
        return ev

    def set_lot_status(self, lot_id: str, status: str, note: str = "") -> bool:
        status = LotStatus(status).value
        cur = self._write(
            "UPDATE lots SET status=?, notes=CASE WHEN ?='' THEN notes ELSE ? END WHERE lot_id=?",
            (status, note, note, lot_id),
        )
        return cur.rowcount > 0

    def record_decision(self, d: DecisionRecord) -> DecisionRecord:
        self._write(
            """INSERT INTO decisions(request_id,created_at,outcome,reasons,messages,image_sha256,
                    claimed_sku,claimed_lot,resolved_sku,resolved_lot,vision_status,vision_top1,
                    vision_distance,verification_status,provenance_status,encoder_tag,
                    calibration_alpha,operator,payload)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (d.request_id, d.created_at, d.outcome, json.dumps(d.reasons), json.dumps(d.messages, ensure_ascii=False),
             d.image_sha256, d.claimed_sku, d.claimed_lot, d.resolved_sku, d.resolved_lot, d.vision_status,
             d.vision_top1, d.vision_distance, d.verification_status, d.provenance_status, d.encoder_tag,
             d.calibration_alpha, d.operator, json.dumps(d.payload, ensure_ascii=False)),
        )
        return d

    # -- đọc ---------------------------------------------------------------
    def get_sku(self, sku_id: str) -> Optional[SKU]:
        r = self._one("SELECT * FROM skus WHERE sku_id=?", (sku_id,))
        return self._row_to_sku(r) if r else None

    def find_sku_by_gtin(self, gtin: str) -> Optional[SKU]:
        r = self._one("SELECT * FROM skus WHERE gtin=? AND gtin<>''", (gtin,))
        return self._row_to_sku(r) if r else None

    def get_product(self, product_id: str) -> Optional[Product]:
        r = self._one("SELECT * FROM products WHERE product_id=?", (product_id,))
        return Product(**dict(r)) if r else None

    def get_supplier(self, supplier_id: str) -> Optional[Supplier]:
        r = self._one("SELECT * FROM suppliers WHERE supplier_id=?", (supplier_id,))
        if not r:
            return None
        d = dict(r)
        d["certifications"] = json.loads(d.get("certifications") or "[]")
        return Supplier(**d)

    def get_lot(self, lot_id: str) -> Optional[Lot]:
        r = self._one("SELECT * FROM lots WHERE lot_id=?", (lot_id,))
        return Lot(**dict(r)) if r else None

    def list_skus(self, active_only: bool = True) -> list[SKU]:
        q = "SELECT * FROM skus" + (" WHERE active=1" if active_only else "")
        return [self._row_to_sku(r) for r in self._rows(q + " ORDER BY sku_id")]

    def list_lots_for_sku(self, sku_id: str, statuses: Sequence[str] | None = None) -> list[Lot]:
        if statuses:
            marks = ",".join("?" * len(statuses))
            rows = self._rows(
                f"SELECT * FROM lots WHERE sku_id=? AND status IN ({marks}) "
                "ORDER BY received_date DESC, lot_id DESC",
                (sku_id, *statuses),
            )
        else:
            rows = self._rows(
                "SELECT * FROM lots WHERE sku_id=? ORDER BY received_date DESC, lot_id DESC",
                (sku_id,))
        return [Lot(**dict(r)) for r in rows]

    def list_evidence(self, lot_id: str) -> list[Evidence]:
        rows = self._rows("SELECT * FROM evidence WHERE lot_id=? ORDER BY evidence_type", (lot_id,))
        return [Evidence(**dict(r)) for r in rows]

    def list_events(self, lot_id: str) -> list[TraceEvent]:
        out = []
        for r in self._rows("SELECT * FROM trace_events WHERE lot_id=? ORDER BY event_time, event_id", (lot_id,)):
            d = dict(r)
            d["payload"] = json.loads(d.get("payload") or "{}")
            out.append(TraceEvent(**d))
        return out

    def resolve(self, sku_id: str, lot_id: str | None = None) -> ProvenanceRecord:
        """Lắp ghép hồ sơ nguồn gốc. Nếu không chỉ định lô, chọn lô đang active
        nhận gần nhất (FIFO ngược) — quy ước phải nói rõ vì nó ảnh hưởng tới
        việc lô nào bị gán cho một lần bán."""
        sku = self.get_sku(sku_id)
        rec = ProvenanceRecord(sku=sku)
        if sku:
            rec.product = self.get_product(sku.product_id)
        lot: Optional[Lot] = None
        if lot_id:
            lot = self.get_lot(lot_id)
        elif sku:
            candidates = self.list_lots_for_sku(sku_id, statuses=[LotStatus.ACTIVE.value])
            lot = candidates[0] if candidates else None
            if lot is None:
                all_lots = self.list_lots_for_sku(sku_id)
                lot = all_lots[0] if all_lots else None
        if lot:
            rec.lot = lot
            rec.supplier = self.get_supplier(lot.supplier_id)
            rec.evidence = self.list_evidence(lot.lot_id)
            rec.events = self.list_events(lot.lot_id)
        return rec

    def stats(self) -> dict:
        def one(q: str) -> int:
            return int(self._one(q)[0])
        return {
            "suppliers": one("SELECT COUNT(*) FROM suppliers"),
            "products": one("SELECT COUNT(*) FROM products"),
            "skus": one("SELECT COUNT(*) FROM skus"),
            "lots": one("SELECT COUNT(*) FROM lots"),
            "evidence": one("SELECT COUNT(*) FROM evidence"),
            "trace_events": one("SELECT COUNT(*) FROM trace_events"),
            "decisions": one("SELECT COUNT(*) FROM decisions"),
        }

    def decision_breakdown(self) -> dict[str, int]:
        rows = self._rows("SELECT outcome, COUNT(*) c FROM decisions GROUP BY outcome ORDER BY outcome")
        return {r["outcome"]: int(r["c"]) for r in rows}

    def list_lot_summaries(self) -> list[dict]:
        """Danh sách lô gọn cho giao diện. Có mặt ở đây để UI không phải chạm
        vào ``conn`` — chạm trực tiếp là cách dễ nhất để vượt qua khoá đồng bộ."""
        return [{"lot_id": r["lot_id"], "sku_id": r["sku_id"], "status": r["status"]}
                for r in self._rows("SELECT lot_id, sku_id, status FROM lots ORDER BY lot_id")]

    @staticmethod
    def _row_to_sku(r: sqlite3.Row) -> SKU:
        d = dict(r)
        d["requires_weighing"] = bool(d["requires_weighing"])
        d["is_barcodeless"] = bool(d["is_barcodeless"])
        d["active"] = bool(d["active"])
        return SKU(**d)
