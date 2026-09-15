-- Lược đồ dữ liệu RPV. SQLite (stdlib, không cần server) — đủ cho khóa luận và
-- demo 1 quầy. Chuyển sang PostgreSQL cần sửa placeholder, kiểu dữ liệu,
-- PRAGMA, quản lý kết nối và giao dịch trong repository, không chỉ đổi driver.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id    TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    address        TEXT DEFAULT '',
    gln            TEXT DEFAULT '',
    country        TEXT DEFAULT 'VN',
    contact        TEXT DEFAULT '',
    certifications TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS products (
    product_id      TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    category        TEXT DEFAULT 'default',
    brand           TEXT DEFAULT '',
    parent_category TEXT DEFAULT '',
    description     TEXT DEFAULT '',
    image_uri       TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS skus (
    sku_id            TEXT PRIMARY KEY,
    product_id        TEXT NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    name              TEXT NOT NULL,
    unit              TEXT DEFAULT 'kg',
    price             REAL NOT NULL DEFAULT 0 CHECK (price >= 0 AND price <= 1.7976931348623157e308),
    gtin              TEXT DEFAULT '',
    requires_weighing INTEGER DEFAULT 1,
    is_barcodeless    INTEGER DEFAULT 1,
    active            INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_skus_product ON skus(product_id);
CREATE INDEX IF NOT EXISTS idx_skus_gtin ON skus(gtin);

CREATE TABLE IF NOT EXISTS lots (
    lot_id           TEXT PRIMARY KEY,
    sku_id           TEXT NOT NULL REFERENCES skus(sku_id) ON DELETE CASCADE,
    supplier_id      TEXT NOT NULL REFERENCES suppliers(supplier_id),
    origin_country   TEXT DEFAULT 'UNKNOWN',
    origin_region    TEXT DEFAULT '',
    farm_or_facility TEXT DEFAULT '',
    gln              TEXT DEFAULT '',
    trace_code       TEXT DEFAULT '',
    production_date  TEXT,
    received_date    TEXT,
    expiry_date      TEXT,
    quantity         REAL DEFAULT 0,
    unit             TEXT DEFAULT 'kg',
    status           TEXT NOT NULL DEFAULT 'active'
                     CHECK (status IN ('active', 'quarantined', 'recalled', 'expired', 'sold_out')),
    notes            TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_lots_sku ON lots(sku_id);
CREATE INDEX IF NOT EXISTS idx_lots_status ON lots(status);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id   TEXT PRIMARY KEY,
    lot_id        TEXT NOT NULL REFERENCES lots(lot_id) ON DELETE CASCADE,
    evidence_type TEXT DEFAULT 'other',
    uri           TEXT DEFAULT '',
    sha256        TEXT DEFAULT '',
    issuer        TEXT DEFAULT '',
    issued_at     TEXT,
    valid_until   TEXT,
    status        TEXT DEFAULT 'pending',
    verified_by   TEXT DEFAULT '',
    verified_at   TEXT,
    note          TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_evidence_lot ON evidence(lot_id);

CREATE TABLE IF NOT EXISTS trace_events (
    event_id      TEXT PRIMARY KEY,
    lot_id        TEXT NOT NULL REFERENCES lots(lot_id) ON DELETE CASCADE,
    biz_step      TEXT DEFAULT 'receiving',
    disposition   TEXT DEFAULT 'in_progress',
    event_time    TEXT,
    location_gln  TEXT DEFAULT '',
    location_name TEXT DEFAULT '',
    actor         TEXT DEFAULT '',
    payload       TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_lot ON trace_events(lot_id);

CREATE TABLE IF NOT EXISTS decisions (
    request_id         TEXT PRIMARY KEY,
    created_at         TEXT NOT NULL,
    outcome            TEXT NOT NULL,
    reasons            TEXT DEFAULT '[]',
    messages           TEXT DEFAULT '[]',
    image_sha256       TEXT DEFAULT '',
    claimed_sku        TEXT DEFAULT '',
    claimed_lot        TEXT DEFAULT '',
    resolved_sku       TEXT DEFAULT '',
    resolved_lot       TEXT DEFAULT '',
    vision_status      TEXT DEFAULT '',
    vision_top1        TEXT DEFAULT '',
    vision_distance    REAL,
    verification_status TEXT DEFAULT '',
    provenance_status  TEXT DEFAULT '',
    encoder_tag        TEXT DEFAULT '',
    calibration_alpha  REAL,
    operator           TEXT DEFAULT '',
    payload            TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_decisions_created ON decisions(created_at);
CREATE INDEX IF NOT EXISTS idx_decisions_outcome ON decisions(outcome);
