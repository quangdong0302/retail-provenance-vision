"""Test tự động cho phần lõi (chạy được không cần torch/GPU/mạng).

Chạy: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from rpv.audit.hash_chain import AuditLog, verify_chain
from rpv.data.dataset import DatasetIndex, DatasetSplits, build_splits, index_dataset
from tests.image_fixtures import write_image_fixtures
from rpv.domain.models import (
    Evidence, EvidenceStatus, EvidenceType, Lot, LotStatus, Product, SKU, Supplier, TraceEvent,
)
from rpv.provenance.adapters import (
    NullExternalAdapter, build_gs1_digital_link, parse_gs1_digital_link, EpcisExporter,
)
from rpv.provenance.policy import PolicyEngine, PROVENANCE_OK, PROVENANCE_MISSING
from rpv.provenance.repository import ProvenanceRepository, new_id
from rpv.vision.calibration import Calibrator, CalibrationModel, calibrate_global_threshold
from rpv.vision.encoders import ClassicalEncoder, l2_normalize
from rpv.vision.prototypes import BankIncompatibleError, PrototypeBank, kmeans
from rpv.vision.recognizer import Recognizer, STATUS_MATCH, STATUS_UNKNOWN, STATUS_EMPTY_CATALOG
from rpv.vision.verification import Verifier, VERIFY_CONFIRMED, VERIFY_MISMATCH, VERIFY_NOT_IN_CATALOG

_TEMP_DIR = tempfile.TemporaryDirectory(prefix="rpv_tests_", ignore_cleanup_errors=True)
_TMP = Path(_TEMP_DIR.name)
_DATA = _TMP / "images"
write_image_fixtures(_DATA)
_index = index_dataset(_DATA)
_INDEX = DatasetIndex(
    _index, unknown_classes=[c for c in _index if c.startswith('unknown_')],
    records={p: {'group_id': p, 'path': p} for paths in _index.values() for p in paths},
    grouping='object_session',
)


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.enc = ClassicalEncoder()

    def test_output_shape_and_normalization(self):
        paths = sorted((_DATA / "cam_sanh").glob("*.jpg"))[:4]
        E = self.enc.embed_paths(paths, use_tta=False)
        self.assertEqual(E.shape, (4, self.enc.dim))
        np.testing.assert_allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-5)

    def test_deterministic(self):
        p = sorted((_DATA / "cam_sanh").glob("*.jpg"))[0]
        np.testing.assert_allclose(self.enc.embed_path(p), self.enc.embed_path(p), atol=1e-6)

    def test_brightness_invariance(self):
        """Đổi độ sáng không được làm embedding nhảy sang lớp khác.
        Đây là lý do descriptor dùng chromaticity thay vì RGB thô."""
        from PIL import Image
        p = sorted((_DATA / "cam_sanh").glob("*.jpg"))[0]
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
        bright = np.clip(arr * 1.35, 0, 255).astype(np.uint8)
        dark = np.clip(arr * 0.7, 0, 255).astype(np.uint8)
        E = self.enc.embed_arrays([arr.astype(np.uint8), bright, dark])
        d_bright = float(np.linalg.norm(E[0] - E[1]))
        d_dark = float(np.linalg.norm(E[0] - E[2]))
        other = self.enc.embed_path(sorted((_DATA / "khoai_lang_mat").glob("*.jpg"))[0])
        d_other = float(np.linalg.norm(E[0] - other))
        self.assertLess(d_bright, d_other)
        self.assertLess(d_dark, d_other)

    def test_intra_class_closer_than_inter_class(self):
        a = self.enc.embed_paths(sorted((_DATA / "cam_sanh").glob("*.jpg"))[:5])
        b = self.enc.embed_paths(sorted((_DATA / "unknown_dien_thoai").glob("*.jpg"))[:5])
        intra = np.mean([np.linalg.norm(a[i] - a[j]) for i in range(5) for j in range(i + 1, 5)])
        inter = np.mean([np.linalg.norm(x - y) for x in a for y in b])
        self.assertLess(intra, inter)


class TestSplits(unittest.TestCase):
    def test_no_leakage_between_roles(self):
        splits = build_splits(_INDEX, k_shot=3,
                              class_seed=1, calib_test_seed=123)
        splits.assert_disjoint()          # raise nếu có ảnh dùng ở 2 vai trò
        self.assertGreater(splits.summary()["n_novel_classes"], 0)
        self.assertGreater(splits.summary()["n_unknown_test"], 0)

    def test_novel_classes_disjoint_from_train(self):
        splits = build_splits(_INDEX, k_shot=3,
                              class_seed=1, calib_test_seed=123)
        self.assertEqual(set(splits.train_classes) & set(splits.novel_classes), set())

    def test_unknown_classes_never_in_support(self):
        splits = build_splits(_INDEX, k_shot=3,
                              class_seed=1, calib_test_seed=123)
        self.assertFalse(any(c.startswith("unknown_") for c in splits.support))

    def test_flat_images_require_prepare_data(self):
        with self.assertRaisesRegex(ValueError, 'prepare-data'):
            build_splits(index_dataset(_DATA), 3, 1, 123)

    def test_legacy_splits_rejected_before_embedding(self):
        from unittest.mock import patch
        from rpv.evaluation.pipeline import evaluate_system, fit_catalog
        encoder = ClassicalEncoder()
        with patch.object(encoder, 'embed_paths') as embed:
            for run in (fit_catalog, evaluate_system):
                with self.subTest(run=run.__name__), self.assertRaisesRegex(ValueError, 'prepare-data'):
                    run(encoder=encoder, splits=DatasetSplits(protocol='legacy'))
            embed.assert_not_called()

    def test_unknown_calibration_and_novel_catalog_are_rejected(self):
        splits = build_splits(_INDEX, 3, 1, 123)
        splits.unknown_calib = [splits.unknown_test[0]]
        with self.assertRaisesRegex(AssertionError, 'unknown'):
            splits.assert_disjoint()
        splits.unknown_calib = []
        novel = splits.novel_classes[0]
        splits.support[novel] = splits.novel_support[novel]
        with self.assertRaisesRegex(AssertionError, 'known'):
            splits.assert_disjoint()


class TestPrototypeBank(unittest.TestCase):
    def setUp(self):
        self.enc = ClassicalEncoder()
        self.bank = PrototypeBank(self.enc, centroids_per_sku=2)
        for sku in ("cam_sanh", "khoai_lang_mat", "nho_ninh_thuan"):
            self.bank.register(sku, sorted((_DATA / sku).glob("*.jpg"))[:4], label=sku)

    def test_register_and_query(self):
        self.assertEqual(len(self.bank), 3)
        emb = self.enc.embed_path(sorted((_DATA / "cam_sanh").glob("*.jpg"))[-1])
        d = self.bank.distances(emb)
        self.assertEqual(min(d, key=d.get), "cam_sanh")

    def test_reregistration_preserves_operator_gate_and_enrollment(self):
        previous = self.bank.entries['cam_sanh']
        previous.requires_operator = True
        previous.enrollment_id = 'reviewed-enrollment'
        entry = self.bank.register('cam_sanh', sorted((_DATA / 'cam_sanh').glob('*.jpg'))[:2])
        path = _TMP / f"bank_gate_{new_id('b')}.json"
        self.bank.save(path)
        loaded = PrototypeBank(self.enc)
        loaded.load(path)
        for item in (entry, loaded.entries['cam_sanh']):
            self.assertTrue(item.requires_operator)
            self.assertEqual(item.enrollment_id, previous.enrollment_id)
            self.assertEqual(item.registered_at, previous.registered_at)
            self.assertEqual(item.n_shots, 2)

    def test_batch_matches_single(self):
        paths = sorted((_DATA / "nho_ninh_thuan").glob("*.jpg"))[:3]
        E = self.enc.embed_paths(paths)
        D, skus = self.bank.distances_batch(E)
        for i in range(E.shape[0]):
            single = self.bank.distances(E[i])
            for j, s in enumerate(skus):
                self.assertAlmostEqual(D[i, j], single[s], places=5)

    def test_save_load_roundtrip(self):
        path = _TMP / "bank.json"
        self.bank.save(path)
        other = PrototypeBank(ClassicalEncoder()).load(path)
        self.assertEqual(other.sku_ids, self.bank.sku_ids)
        np.testing.assert_allclose(other.entries["cam_sanh"].centroids,
                                   self.bank.entries["cam_sanh"].centroids, atol=1e-5)

    def test_incompatible_encoder_rejected(self):
        """Bank của encoder khác phải bị TỪ CHỐI, không được load im lặng."""
        path = _TMP / "bank2.json"
        self.bank.save(path)
        payload = json.loads(path.read_text())
        payload["_meta"]["encoder_tag"] = "dinov2_vitl14_raw_s224"
        payload["_meta"]["dim"] = 1024
        path.write_text(json.dumps(payload))
        with self.assertRaises(BankIncompatibleError):
            PrototypeBank(ClassicalEncoder()).load(path)

    def test_remove_and_empty_query(self):
        for sku in list(self.bank.sku_ids):
            self.bank.remove(sku)
        self.assertEqual(self.bank.distances(np.zeros(self.enc.dim, np.float32)), {})

    def test_kmeans_returns_normalized_centroids(self):
        V = l2_normalize(np.random.default_rng(0).normal(size=(12, 8)).astype(np.float32))
        C = kmeans(V, 3, seed=1)
        self.assertEqual(C.shape, (3, 8))
        np.testing.assert_allclose(np.linalg.norm(C, axis=1), 1.0, atol=1e-5)

    def test_register_rejects_empty(self):
        with self.assertRaises(ValueError):
            self.bank.register("x", [])


class TestCalibration(unittest.TestCase):
    def test_global_threshold_respects_business_constraint(self):
        known = [0.1, 0.2, 0.3, 0.4, 0.5]
        unknown = [0.8, 0.9, 1.0]
        res = calibrate_global_threshold(known, unknown, 0.90)
        self.assertGreaterEqual(res["known_recall"], 0.90)
        self.assertEqual(res["unknown_reject_rate"], 1.0)

    def test_recall_constraint_is_trivially_satisfiable_so_reject_rate_must_be_checked(self):
        """Ràng buộc known-recall luôn đạt được bằng cách chấp nhận tất cả.
        Lỗi này chỉ lộ ra khi viết test: hàm phải CẢNH BÁO khi ngưỡng tìm được
        không chặn nổi hàng lạ, thay vì báo thành công."""
        res = calibrate_global_threshold([0.9, 0.95], [0.1, 0.2], 0.99)
        self.assertGreaterEqual(res["known_recall"], 0.99)
        self.assertEqual(res["unknown_reject_rate"], 0.0)
        self.assertIn("warnings", res)
        self.assertTrue(any("hàng lạ" in w for w in res["warnings"]))

    def test_warns_when_no_unknown_samples(self):
        res = calibrate_global_threshold([0.1, 0.2, 0.3], [], 0.90)
        self.assertTrue(any("ngoài danh mục" in w for w in res.get("warnings", [])))

    def test_conformal_quantile_monotone_in_alpha(self):
        scores = list(np.linspace(0.1, 1.0, 40))
        c = Calibrator("tag")
        q_loose = c.__class__("tag", alpha=0.30).fit({"a": scores}).quantile_for("a")
        q_tight = c.__class__("tag", alpha=0.01).fit({"a": scores}).quantile_for("a")
        self.assertLess(q_loose, q_tight)   # alpha nhỏ -> bán kính rộng hơn

    def test_per_class_fallback_when_too_few_samples(self):
        model = Calibrator("tag", min_calib_per_class=8).fit(
            {"rich": list(np.linspace(0.1, 0.5, 20)), "poor": [0.2, 0.3]})
        self.assertIn("rich", model.per_class_threshold)
        self.assertIn("poor", model.fallback_classes)
        self.assertEqual(model.threshold_for("poor"), model.global_threshold)

    def test_save_load_rejects_wrong_encoder(self):
        model = Calibrator("enc_a").fit({"a": [0.1, 0.2, 0.3]})
        p = _TMP / "calib.json"
        model.save(p)
        CalibrationModel.load(p, "enc_a")
        with self.assertRaises(RuntimeError):
            CalibrationModel.load(p, "enc_b")


class TestRecognizerVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enc = ClassicalEncoder()
        cls.splits = build_splits(_INDEX, k_shot=3,
                                  class_seed=1, calib_test_seed=123)
        cls.bank = PrototypeBank(cls.enc)
        for c, paths in cls.splits.support.items():
            cls.bank.register(c, paths, label=c)
        true_d = {}
        for c, paths in cls.splits.calib.items():
            E = cls.enc.embed_paths(paths)
            D, skus = cls.bank.distances_batch(E)
            true_d[c] = [float(x) for x in D[:, skus.index(c)]]
        cls.calib = Calibrator(cls.enc.tag, alpha=0.10).fit(true_d)

    def test_recognizer_returns_prediction_set(self):
        rec = Recognizer(self.enc, self.bank, self.calib)
        res = rec.recognize(self.splits.test["cam_sanh"][0])
        self.assertIn(res.status, {STATUS_MATCH, "ambiguous", STATUS_UNKNOWN})
        self.assertTrue(all(c.in_prediction_set for c in res.candidates if c.sku_id in res.prediction_set))

    def test_empty_catalog_is_explicit(self):
        rec = Recognizer(self.enc, PrototypeBank(self.enc), None)
        self.assertEqual(rec.recognize(self.splits.test["cam_sanh"][0]).status, STATUS_EMPTY_CATALOG)

    def test_calibration_encoder_mismatch_refused(self):
        bad = Calibrator("other_tag").fit({"a": [0.1, 0.2, 0.3]})
        with self.assertRaises(RuntimeError):
            Recognizer(self.enc, self.bank, bad)

    def test_verifier_confirms_genuine_claim_more_often_than_impostor(self):
        v = Verifier(self.enc, self.bank, self.calib)
        genuine = impostor = 0
        classes = sorted(self.splits.test)
        for i, c in enumerate(classes):
            paths = self.splits.test[c]
            if not paths:
                continue
            E = self.enc.embed_path(paths[0])
            genuine += int(v.verify_embedding(E, c).accepted)
            other = classes[(i + 3) % len(classes)]
            if other != c:
                impostor += int(v.verify_embedding(E, other).accepted)
        self.assertGreater(genuine, impostor)

    def test_verifier_unknown_sku(self):
        v = Verifier(self.enc, self.bank, self.calib)
        res = v.verify(self.splits.test["cam_sanh"][0], "sku_khong_ton_tai")
        self.assertEqual(res.status, VERIFY_NOT_IN_CATALOG)


class TestRepository(unittest.TestCase):
    def setUp(self):
        self.repo = ProvenanceRepository(_TMP / f"db_{new_id('t')}.db")
        self.repo.init_schema()
        self.repo.upsert_supplier(Supplier("sup_1", "HTX Test", "Long An", "8931234567890"))
        self.repo.upsert_product(Product("prd_1", "Xoài cát", category="fresh_produce", image_uri="x.jpg"))
        self.repo.upsert_sku(SKU("sku_1", "prd_1", "Xoài cát Hòa Lộc", price=42000))

    def tearDown(self):
        self.repo.close()

    def test_lot_and_evidence_roundtrip(self):
        today = date.today()
        self.repo.upsert_lot(Lot("lot_1", "sku_1", "sup_1", origin_country="VN",
                                 origin_region="Đồng Tháp", production_date=str(today),
                                 received_date=str(today), expiry_date=str(today + timedelta(days=10))))
        self.repo.add_evidence(Evidence("ev_1", "lot_1", EvidenceType.DELIVERY_NOTE.value,
                                        status=EvidenceStatus.VERIFIED.value))
        rec = self.repo.resolve("sku_1")
        self.assertTrue(rec.found)
        self.assertEqual(rec.lot.lot_id, "lot_1")
        self.assertIn(EvidenceType.DELIVERY_NOTE.value, rec.verified_evidence_types())

    def test_trace_code_autogenerated(self):
        lot = self.repo.upsert_lot(Lot("lot_2", "sku_1", "sup_1"))
        self.assertTrue(lot.trace_code)

    def test_missing_traceability_fields_detected(self):
        self.repo.upsert_lot(Lot("lot_3", "sku_1", "sup_1"))     # thiếu ngày sản xuất
        rec = self.repo.resolve("sku_1", "lot_3")
        self.assertIn("production_time", rec.missing_traceability_fields())

    def test_invalid_gtin_rejected(self):
        with self.assertRaises(ValueError):
            self.repo.upsert_sku(SKU("sku_bad", "prd_1", "Sai GTIN", gtin="12345"))

    def test_expiry_helpers(self):
        # Explicit business date avoids a UTC/local-date mismatch near midnight.
        today = date(2026, 9, 13)
        past = Lot("l", "sku_1", "sup_1", expiry_date=str(today - timedelta(days=1)))
        self.assertTrue(past.is_expired(today))
        self.assertEqual(past.days_to_expiry(today), -1)


class TestPolicyEngine(unittest.TestCase):
    def setUp(self):
        self.repo = ProvenanceRepository(_TMP / f"db_pol_{new_id('t')}.db")
        self.repo.init_schema()
        self.repo.upsert_supplier(Supplier("sup_1", "HTX Test", "Long An", "8931234567890"))
        self.repo.upsert_product(Product("prd_1", "Xoài cát", category="fresh_produce",
                                         brand="HTX", image_uri="x.jpg"))
        self.repo.upsert_sku(SKU("sku_1", "prd_1", "Xoài", price=42000))
        self.today = date.today()
        self.engine = PolicyEngine()

    def tearDown(self):
        self.repo.close()

    def _good_lot(self, lot_id="lot_ok", status=LotStatus.ACTIVE.value, expiry_days=20):
        self.repo.upsert_lot(Lot(lot_id, "sku_1", "sup_1", origin_country="VN",
                                 origin_region="Đồng Tháp", farm_or_facility="Trại A",
                                 production_date=str(self.today - timedelta(days=2)),
                                 received_date=str(self.today),
                                 expiry_date=str(self.today + timedelta(days=expiry_days)),
                                 status=status))
        for t in (EvidenceType.DELIVERY_NOTE.value, EvidenceType.SUPPLIER_DECLARATION.value):
            self.repo.add_evidence(Evidence(new_id("ev"), lot_id, t,
                                            status=EvidenceStatus.VERIFIED.value,
                                            verified_at=str(self.today)))
        self.repo.add_event(TraceEvent(new_id("evt"), lot_id, "receiving",
                                       event_time=str(self.today)))
        return self.repo.resolve("sku_1", lot_id)

    def _vision_ok(self):
        from rpv.vision.verification import VerificationResult
        return VerificationResult(VERIFY_CONFIRMED, "sku_1", 0.2, "sku_1", 0.2, 0.5, 0.6, 0.0)

    def test_happy_path_allows(self):
        d = self.engine.decide(verification=self._vision_ok(), record=self._good_lot(), today=self.today)
        self.assertEqual(d.outcome, "allow")
        self.assertEqual(d.provenance_status, PROVENANCE_OK)
        self.assertFalse(d.requires_operator)

    def test_sold_out_and_invalid_status_cannot_allow_checkout(self):
        record = self._good_lot()
        for status, reason in (('sold_out', 'LOT_SOLD_OUT'), ('bogus', 'LOT_STATUS_INVALID'),
                               (None, 'LOT_STATUS_INVALID')):
            record.lot.status = status
            with self.subTest(status=status):
                result = self.engine.decide(verification=self._vision_ok(), record=record)
                self.assertEqual(result.outcome, 'block')
                self.assertEqual(result.provenance_status, 'provenance_blocked')
                self.assertIn(reason, result.reasons)

    def test_invalid_prices_are_rejected_and_old_bad_records_require_assist(self):
        record = self._good_lot()
        for price in (float('nan'), float('inf'), -float('inf'), -1, None):
            with self.subTest(price=price):
                record.sku.price = price
                with self.assertRaises(ValueError):
                    self.repo.upsert_sku(record.sku)
                result = self.engine.decide(verification=self._vision_ok(), record=record, claimed_price=1)
                self.assertEqual(result.outcome, 'assist')
                self.assertIn('PRICE_INVALID', result.reasons)
        record.sku.price = 1000
        for price in (float('nan'), float('inf'), -1):
            result = self.engine.decide(verification=self._vision_ok(), record=record, claimed_price=price)
            self.assertEqual(result.outcome, 'assist')
            self.assertIn('PRICE_INVALID', result.reasons)

    def test_repository_and_schema_reject_invalid_status_and_prices(self):
        import sqlite3
        self._good_lot()
        with self.assertRaises(ValueError):
            self.repo.set_lot_status('lot_ok', 'bogus')
        self.assertEqual(self.repo.get_lot('lot_ok').status, 'active')
        with self.assertRaises(sqlite3.IntegrityError):
            self.repo._write('UPDATE lots SET status=? WHERE lot_id=?', ('bogus', 'lot_ok'))
        for price in (float('nan'), float('inf'), -1, None):
            with self.subTest(price=price), self.assertRaises(sqlite3.IntegrityError):
                self.repo._write('UPDATE skus SET price=? WHERE sku_id=?', (price, 'sku_1'))
        self.assertEqual(self.repo.get_sku('sku_1').price, 42000)
        self.assertTrue(self.repo.set_lot_status('lot_ok', 'sold_out'))

    def test_missing_lot_requires_assist_and_never_claims_origin(self):
        d = self.engine.decide(verification=self._vision_ok(),
                               record=self.repo.resolve("sku_1"), today=self.today)
        self.assertEqual(d.outcome, "assist")
        self.assertEqual(d.provenance_status, PROVENANCE_MISSING)
        self.assertIn("Chưa xác minh", d.origin_summary)

    def test_expired_lot_blocks_even_with_perfect_vision(self):
        rec = self._good_lot("lot_exp", expiry_days=-3)
        d = self.engine.decide(verification=self._vision_ok(), record=rec, today=self.today)
        self.assertEqual(d.outcome, "block")
        self.assertIn("LOT_EXPIRED", d.reasons)

    def test_recalled_lot_blocks(self):
        rec = self._good_lot("lot_rec", status=LotStatus.RECALLED.value)
        d = self.engine.decide(verification=self._vision_ok(), record=rec, today=self.today)
        self.assertEqual(d.outcome, "block")
        self.assertIn("LOT_RECALLED", d.reasons)

    def test_vision_mismatch_blocks(self):
        from rpv.vision.verification import VerificationResult
        v = VerificationResult(VERIFY_MISMATCH, "sku_1", 0.9, "sku_9", 0.2, 0.5, 0.0, 0.7)
        d = self.engine.decide(verification=v, record=self._good_lot("lot_m"), today=self.today)
        self.assertEqual(d.outcome, "block")
        self.assertIn("VISION_MISMATCH", d.reasons)
        self.assertIn("PRICE_RISK", d.reasons)

    def test_missing_evidence_escalates(self):
        self.repo.upsert_lot(Lot("lot_noev", "sku_1", "sup_1", origin_country="VN",
                                 origin_region="X", production_date=str(self.today),
                                 received_date=str(self.today),
                                 expiry_date=str(self.today + timedelta(days=9))))
        d = self.engine.decide(verification=self._vision_ok(),
                               record=self.repo.resolve("sku_1", "lot_noev"), today=self.today)
        self.assertEqual(d.outcome, "assist")
        self.assertIn("PROVENANCE_INCOMPLETE", d.reasons)

    def test_imported_lot_requires_more_documents(self):
        self.repo.upsert_lot(Lot("lot_imp", "sku_1", "sup_1", origin_country="US",
                                 origin_region="Washington", production_date=str(self.today),
                                 received_date=str(self.today),
                                 expiry_date=str(self.today + timedelta(days=30))))
        self.repo.add_evidence(Evidence(new_id("ev"), "lot_imp", EvidenceType.DELIVERY_NOTE.value,
                                        status=EvidenceStatus.VERIFIED.value))
        d = self.engine.decide(verification=self._vision_ok(),
                               record=self.repo.resolve("sku_1", "lot_imp"), today=self.today)
        self.assertIn("PROVENANCE_INCOMPLETE", d.reasons)
        self.assertEqual(d.outcome, "assist")

    def test_severity_takes_the_worst_signal(self):
        """Nhiều tín hiệu cùng lúc -> kết cục là mức nghiêm trọng nhất, không phải mức đầu tiên."""
        rec = self._good_lot("lot_worst", expiry_days=1)     # near expiry (warning)
        rec.lot.status = LotStatus.RECALLED.value            # + recalled (block)
        d = self.engine.decide(verification=self._vision_ok(), record=rec, today=self.today)
        self.assertEqual(d.outcome, "block")


class TestAuditChain(unittest.TestCase):
    def setUp(self):
        self.path = _TMP / f"audit_{new_id('a')}.jsonl"
        self.log = AuditLog(self.path)

    def test_chain_valid_after_appends(self):
        for i in range(5):
            self.log.append("test_event", {"i": i})
        res = self.log.verify()
        self.assertTrue(res["valid"], res["problems"])
        self.assertEqual(res["n_entries"], 5)

    def test_tampering_detected(self):
        for i in range(4):
            self.log.append("e", {"i": i})
        lines = self.path.read_text(encoding="utf-8").strip().split("\n")
        rec = json.loads(lines[2])
        rec["payload"]["i"] = 999                      # sửa nội dung, giữ nguyên hash
        lines[2] = json.dumps(rec, ensure_ascii=False)
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        res = AuditLog(self.path).verify()
        self.assertFalse(res["valid"])
        self.assertTrue(any("bị sửa" in p for p in res["problems"]))

    def test_deletion_detected(self):
        for i in range(4):
            self.log.append("e", {"i": i})
        lines = self.path.read_text(encoding="utf-8").strip().split("\n")
        del lines[1]
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertFalse(AuditLog(self.path).verify()["valid"])

    def test_empty_chain_is_valid(self):
        self.assertTrue(verify_chain([])["valid"])


class TestAdapters(unittest.TestCase):
    def test_digital_link_roundtrip(self):
        url = build_gs1_digital_link("09506000134352", batch_lot="LOT/2026-01", serial="77")
        parsed = parse_gs1_digital_link(url)
        self.assertEqual(parsed["gtin"], "09506000134352")
        self.assertEqual(parsed["batch_lot"], "LOT/2026-01")
        self.assertEqual(parsed["serial"], "77")

    def test_digital_link_rejects_bad_gtin(self):
        with self.assertRaises(ValueError):
            build_gs1_digital_link("123")

    def test_digital_link_query_params(self):
        parsed = parse_gs1_digital_link("https://id.x/01/09506000134352?10=L1&17=261231")
        self.assertEqual(parsed["batch_lot"], "L1")
        self.assertEqual(parsed["expiry_date"], "261231")

    def test_external_adapter_disabled_is_explicit_not_silent(self):
        res = NullExternalAdapter().enrich_by_gtin("09506000134352")
        self.assertFalse(res.found)
        self.assertEqual(res.evidence_grade, "none")
        self.assertTrue(res.disclaimer)

    def test_epcis_export_structure(self):
        ev = TraceEvent("e1", "lot_1", "receiving", event_time="2026-09-01T10:00:00+07:00",
                        location_gln="8931234567890")
        doc = EpcisExporter.export([ev], trace_code="urn:rpv:lot:lot_1")
        self.assertEqual(doc["type"], "EPCISDocument")
        self.assertEqual(len(doc["epcisBody"]["eventList"]), 1)
        self.assertIn("bizStep", doc["epcisBody"]["eventList"][0])


class TestMetrics(unittest.TestCase):
    def test_fpr_uses_smallest_threshold_meeting_target(self):
        from rpv.evaluation.metrics import fpr_at_tpr
        result = fpr_at_tpr(range(1, 21), [10, 20, 21])
        self.assertEqual(result['threshold'], 19)
        self.assertEqual(result['tpr'], .95)
        self.assertAlmostEqual(result['fpr'], 1 / 3)
        self.assertEqual(fpr_at_tpr([1, 2, 2, 3], [2, 4], .6)['threshold'], 2)
        for target in (0, -1, 1.1, float('nan')):
            with self.subTest(target=target), self.assertRaises(ValueError):
                fpr_at_tpr([1], [2], target)

    def test_metrics_reject_mismatched_lengths_and_nonfinite_scores(self):
        from rpv.evaluation import metrics
        for metric in (metrics.closed_set_accuracy, metrics.per_class_recall,
                       metrics.macro_f1, metrics.confusion_pairs):
            for truth, predicted in ((['a', 'b'], ['a']), ([], ['a']), (['a'], ['a', 'b'])):
                with self.subTest(metric=metric.__name__, truth=truth), self.assertRaises(ValueError):
                    metric(truth, predicted)
        with self.assertRaises(ValueError):
            metrics.risk_coverage_curve([True, False], [.5])
        for bad in (float('nan'), float('inf'), -float('inf')):
            for metric in (metrics.auroc, metrics.fpr_at_tpr):
                for first, second in (([bad], [1]), ([1], [bad]), ([], [bad])):
                    with self.subTest(metric=metric.__name__, bad=bad), self.assertRaises(ValueError):
                        metric(first, second)
            with self.assertRaises(ValueError):
                metrics.risk_coverage_curve([True], [bad])

    def test_auroc_perfect_and_random(self):
        from rpv.evaluation.metrics import auroc
        self.assertEqual(auroc([1.0, 1.1, 1.2], [0.1, 0.2, 0.3]), 1.0)
        self.assertAlmostEqual(auroc([0.5, 0.5], [0.5, 0.5]), 0.5, places=6)

    def test_macro_f1_bounds(self):
        from rpv.evaluation.metrics import macro_f1
        self.assertEqual(macro_f1(["a", "b"], ["a", "b"]), 1.0)
        self.assertEqual(macro_f1(["a", "a"], ["b", "b"]), 0.0)

    def test_risk_coverage_monotone_answered(self):
        from rpv.evaluation.metrics import coverage_at_risk, risk_coverage_curve
        correct = [True] * 8 + [False] * 2
        conf = list(np.linspace(1, 0, 10))
        curve = risk_coverage_curve(correct, conf, n_points=5)
        self.assertEqual(curve[-1]["coverage"], 1.0)
        self.assertLessEqual(coverage_at_risk(curve, 0.0), 1.0)

    def test_verification_metrics_far_frr(self):
        from rpv.evaluation.metrics import verification_metrics
        recs = [{"genuine": True, "accepted": True, "score": 0.1},
                {"genuine": True, "accepted": False, "score": 0.6},
                {"genuine": False, "accepted": False, "score": 0.9},
                {"genuine": False, "accepted": True, "score": 0.2}]
        m = verification_metrics(recs)
        self.assertAlmostEqual(m["false_accept_rate"], 0.5)
        self.assertAlmostEqual(m["false_reject_rate"], 0.5)
        self.assertIn("eer", m)


# ===========================================================================
# Các lớp test dưới đây gắn với những lỗi CỤ THỂ đã sửa khi chuyển sang DINOv2.
# Mỗi test tồn tại vì có một lỗi thật, không phải để tăng số dòng coverage.
# ===========================================================================


class TestStableHashAndJson(unittest.TestCase):
    """hash() của Python bị random hoá theo tiến trình; JSON không có NaN."""

    def test_stable_hash_is_fixed_across_processes(self):
        import subprocess
        import sys as _sys
        from rpv.util import stable_hash_hex
        here = stable_hash_hex("lot_happy", "delivery_note")
        out = subprocess.run(
            [_sys.executable, "-B", "-c",
             "from rpv.util import stable_hash_hex;print(stable_hash_hex('lot_happy','delivery_note'))"],
            capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": "12345", "PATH": os.environ.get("PATH", ""),
                 "PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
        )
        self.assertEqual(here, out.stdout.strip())

    def test_builtin_hash_would_not_be_stable(self):
        """Ghi lại lý do tồn tại của stable_hash: giá trị phải KHÔNG phụ thuộc seed."""
        from rpv.util import stable_hash_int
        self.assertEqual(stable_hash_int("x"), stable_hash_int("x"))
        self.assertNotEqual(stable_hash_int("x"), stable_hash_int("y"))

    def test_json_dumps_never_emits_nan_or_infinity(self):
        from rpv.util import json_dumps
        payload = {"a": float("nan"), "b": float("inf"), "c": [float("-inf"), 1.5],
                   "d": {"e": float("nan")}, "ok": True}
        text = json_dumps(payload, indent=None)
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        back = json.loads(text)
        self.assertIsNone(back["a"])
        self.assertIsNone(back["b"])
        self.assertEqual(back["c"], [None, 1.5])
        self.assertIs(back["ok"], True)


class TestEmbeddingCache(unittest.TestCase):
    """Cache phải trả về ĐÚNG vector đã tính, và không bao giờ làm chết đường chạy."""

    def setUp(self):
        from rpv.config import reload_settings
        self.dir = _TMP / f"cache_{new_id('c')}"
        self.previous_cache_dir = os.environ.get('RPV_CACHE_DIR')
        os.environ["RPV_CACHE_DIR"] = str(self.dir)
        os.environ["RPV_EMBED_CACHE"] = "1"
        reload_settings()

    def tearDown(self):
        from rpv.config import reload_settings
        if self.previous_cache_dir is None:
            os.environ.pop("RPV_CACHE_DIR", None)
        else:
            os.environ['RPV_CACHE_DIR'] = self.previous_cache_dir
        reload_settings()

    def test_cached_result_matches_fresh_computation(self):
        enc = ClassicalEncoder()
        paths = sorted((_DATA / "cam_sanh").glob("*.jpg"))[:4]
        first = enc.embed_paths(paths)
        cache = enc._cache()
        self.assertIsNotNone(cache)
        self.assertGreaterEqual(len(cache), len(paths))
        second = ClassicalEncoder().embed_paths(paths)       # instance mới, đọc từ file
        np.testing.assert_allclose(first, second, atol=0)

    def test_cache_key_separates_tta_from_no_tta(self):
        enc = ClassicalEncoder()
        p = sorted((_DATA / "cam_sanh").glob("*.jpg"))[:2]
        with_tta = enc.embed_paths(p, use_tta=True)
        without = enc.embed_paths(p, use_tta=False)
        self.assertFalse(np.allclose(with_tta, without),
                         "TTA và không TTA phải là hai khoá cache khác nhau")

    def test_corrupt_cache_file_is_ignored_not_fatal(self):
        from rpv.vision.cache import EmbeddingCache
        c = EmbeddingCache("t", 4, self.dir, 10)
        c.put("k", np.ones(4, np.float32))
        c.flush()
        c.path.write_bytes(b"khong phai npz")
        fresh = EmbeddingCache("t", 4, self.dir, 10)
        self.assertIsNone(fresh.get("k"))
        self.assertEqual(len(fresh), 0)

    def test_cache_rejects_wrong_dim(self):
        from rpv.vision.cache import EmbeddingCache
        c = EmbeddingCache("t", 4, self.dir, 10)
        with self.assertRaises(ValueError):
            c.put("k", np.ones(5, np.float32))


class TestPairwiseDistances(unittest.TestCase):
    """Bản GEMM phải cho cùng kết quả với cách tính trực tiếp."""

    def test_matches_naive_norm(self):
        from rpv.vision.prototypes import pairwise_distances
        rng = np.random.default_rng(0)
        A = rng.normal(size=(7, 16)).astype(np.float32)
        B = rng.normal(size=(5, 16)).astype(np.float32)
        naive = np.linalg.norm(A[:, None, :] - B[None], axis=2)
        np.testing.assert_allclose(pairwise_distances(A, B), naive, atol=1e-4)

    def test_never_returns_negative_or_nan(self):
        from rpv.vision.prototypes import pairwise_distances
        A = np.full((3, 8), 0.35, dtype=np.float32)
        D = pairwise_distances(A, A)                 # vector trùng nhau -> d^2 âm do sai số
        self.assertTrue(np.all(np.isfinite(D)))
        self.assertTrue(np.all(D >= 0.0))

    def test_dim_mismatch_is_explicit(self):
        enc = ClassicalEncoder()
        bank = PrototypeBank(enc, centroids_per_sku=1)
        bank.register("a", sorted((_DATA / "cam_sanh").glob("*.jpg"))[:3])
        with self.assertRaises(ValueError):
            bank.distances_batch(np.zeros((2, 5), np.float32))


class TestConformalPValue(unittest.TestCase):
    """p-value phải là p-value: đơn điệu, trong (0,1], và có ý nghĩa thống kê."""

    def _model(self, n=200):
        rng = np.random.default_rng(7)
        scores = list(rng.normal(0.5, 0.1, n))
        return Calibrator("tag", alpha=0.1, min_calib_per_class=5).fit({"a": scores}), scores

    def test_monotone_and_bounded(self):
        model, _ = self._model()
        ps = [model.p_value("a", x) for x in (0.0, 0.3, 0.5, 0.7, 5.0)]
        self.assertTrue(all(0.0 < x <= 1.0 for x in ps), ps)
        self.assertEqual(ps, sorted(ps, reverse=True), "p-value phải giảm khi khoảng cách tăng")

    def test_approximately_uniform_under_null(self):
        """Điểm cùng phân phối với calibration có p-value xấp xỉ đều.
        Công thức 1-d/threshold không có tính chất này, nên không diễn giải
        được mức ý nghĩa 5%.
        """
        model, _ = self._model(500)
        rng = np.random.default_rng(11)
        ps = np.array([model.p_value("a", x) for x in rng.normal(0.5, 0.1, 2000)])
        self.assertLess(abs(float(ps.mean()) - 0.5), 0.05)
        # Sai số Monte-Carlo với n=500 mẫu calib: biên 0.05 là mức nói được điều
        # cần nói (p-value không lệch hệ thống) mà không làm test chập chờn.
        self.assertLess(abs(float((ps <= 0.1).mean()) - 0.1), 0.05)

    def test_falls_back_to_global_for_unseen_class(self):
        model, _ = self._model()
        self.assertTrue(np.isfinite(model.p_value("chua_tung_thay", 0.5)))

    def test_calibration_json_roundtrip_keeps_scores(self):
        model, _ = self._model()
        path = _TMP / f"calib_{new_id('c')}.json"
        model.save(path)
        back = CalibrationModel.load(path, "tag")
        self.assertAlmostEqual(model.p_value("a", 0.55), back.p_value("a", 0.55), places=6)

    def test_unknown_keys_in_file_are_ignored(self):
        model, _ = self._model()
        path = _TMP / f"calib_fwd_{new_id('c')}.json"
        model.save(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["truong_moi_cua_ban_sau"] = 123
        path.write_text(json.dumps(raw), encoding="utf-8")
        self.assertEqual(CalibrationModel.load(path, "tag").encoder_tag, "tag")


class TestVisionPreprocessing(unittest.TestCase):
    """Tiền xử lý ảnh cho ViT — test được mà không cần PyTorch."""

    def test_output_shape_and_normalization(self):
        from rpv.vision.encoders import preprocess_for_vit
        arr = np.asarray(Image.open(sorted((_DATA / "cam_sanh").glob("*.jpg"))[0]).convert("RGB"))
        x = preprocess_for_vit(arr, 224)
        self.assertEqual(x.shape, (3, 224, 224))
        self.assertEqual(x.dtype, np.float32)

    def test_center_crop_preserves_aspect_ratio(self):
        """Ảnh 2:1 giữ tỷ lệ vật thể sau resize cạnh ngắn và cắt giữa.
        Resize thẳng về hình vuông có thể làm biến dạng vật thể dài.
        """
        from rpv.vision.encoders import preprocess_for_vit
        wide = np.zeros((100, 200, 3), np.uint8)
        wide[40:60, 90:110] = 255                     # ô vuông 20x20 ở giữa
        cc = preprocess_for_vit(wide, 224, center_crop=True)
        sq = preprocess_for_vit(wide, 224, center_crop=False)
        def ratio(x):
            m = x[0] > x[0].min() + 0.1
            ys, xs = np.nonzero(m)
            return (xs.max() - xs.min() + 1) / max(1, ys.max() - ys.min() + 1)
        self.assertLess(abs(ratio(cc) - 1.0), 0.25, "center-crop phải giữ ô vuông là vuông")
        # Ảnh 100x200 bị ép về 224x224 -> chiều cao giãn 2.24x, chiều rộng 1.12x,
        # nên ô vuông thành hình chữ nhật đứng. Kiểm tra độ méo, không kiểm tra
        # chiều méo, để test nói đúng điều nó muốn nói.
        self.assertGreater(abs(float(np.log(ratio(sq)))), 0.4,
                           "resize hình vuông làm méo vật thể (đây là hành vi mong đợi)")

    def test_extreme_aspect_ratio_still_produces_valid_tensor(self):
        from rpv.vision.encoders import preprocess_for_vit
        for shape in ((5, 500, 3), (500, 5, 3), (1, 1, 3)):
            x = preprocess_for_vit(np.zeros(shape, np.uint8), 224)
            self.assertEqual(x.shape, (3, 224, 224))


@unittest.skipUnless(__import__("rpv.vision.encoders", fromlist=["torch_available"]).torch_available(),
                     "cần PyTorch")
class TestDinoV2Encoder(unittest.TestCase):
    """Kiểm tra phần validate cấu hình của DINOv2 (không tải trọng số)."""

    def test_bad_variant_rejected(self):
        from rpv.vision.encoders import DinoV2Encoder
        with self.assertRaises(ValueError):
            DinoV2Encoder(variant="vits99")

    def test_image_size_must_be_multiple_of_patch(self):
        from rpv.vision.encoders import DinoV2Encoder
        with self.assertRaises(ValueError) as ctx:
            DinoV2Encoder(image_size=250)
        self.assertIn("14", str(ctx.exception))


class TestEncoderFactory(unittest.TestCase):
    def test_dinov2_is_the_default_encoder(self):
        from rpv.config import VisionConfig
        old = os.environ.pop("RPV_ENCODER", None)
        try:
            self.assertEqual(VisionConfig().encoder, "dinov2")
        finally:
            if old is not None:
                os.environ["RPV_ENCODER"] = old

    def test_unknown_encoder_rejected(self):
        from rpv.vision.encoders import load_encoder
        with self.assertRaises(ValueError):
            load_encoder("resnet50")

    def test_missing_torch_never_silently_downgrades(self):
        """Thiếu torch phải là lỗi, không được âm thầm chuyển sang classical:
        đổi encoder ngầm làm mọi số liệu so sánh mất ý nghĩa."""
        from rpv.vision.encoders import load_encoder, torch_available
        if torch_available():
            self.skipTest("môi trường có torch")
        with self.assertRaises(RuntimeError):
            load_encoder("dinov2")


class TestScenarioGuards(unittest.TestCase):
    def test_too_few_skus_raises_clear_error_not_indexerror(self):
        """Bộ kịch bản cần 8 SKU nhưng bản trước chỉ kiểm tra >= 7."""
        from rpv.evaluation.scenarios import build_provenance_scenarios
        repo = ProvenanceRepository(_TMP / f"db_sc_{new_id('t')}.db")
        repo.init_schema()
        try:
            with self.assertRaises(ValueError) as ctx:
                build_provenance_scenarios(repo, {f"sku_{i}": [f"/tmp/{i}.jpg"] for i in range(7)})
            self.assertIn("8", str(ctx.exception))
        finally:
            repo.close()

    def test_seeded_evidence_hash_is_reproducible(self):
        from rpv.util import stable_hash_hex
        self.assertEqual(stable_hash_hex("lot_happy", "delivery_note"),
                         stable_hash_hex("lot_happy", "delivery_note"))
        self.assertEqual(len(stable_hash_hex("a", "b")), 64)


class TestConcurrency(unittest.TestCase):
    """Kiểm tra repository và audit khi được gọi từ nhiều thread.
    FastAPI và Gradio dùng nhiều thread; test tuần tự không kiểm tra được
    các trường hợp truy cập đồng thời này.
    """

    def test_repository_usable_from_other_threads(self):
        import threading
        repo = ProvenanceRepository(_TMP / f"db_thr_{new_id('t')}.db")
        repo.init_schema()
        repo.upsert_supplier(Supplier("sup_1", "HTX Test"))
        repo.upsert_product(Product("prd_1", "X"))
        repo.upsert_sku(SKU("sku_1", "prd_1", "X"))
        errors: list[str] = []

        def worker(i: int):
            try:
                repo.upsert_lot(Lot(f"lot_{i}", "sku_1", "sup_1", origin_country="VN"))
                repo.resolve("sku_1", f"lot_{i}")
                repo.stats()
            except Exception as exc:                 # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        try:
            self.assertEqual(errors, [])
            self.assertEqual(repo.stats()["lots"], 12)
        finally:
            repo.close()

    def test_audit_chain_stays_valid_under_concurrent_threads(self):
        import threading
        log = AuditLog(_TMP / f"audit_thr_{new_id('a')}.jsonl")

        def worker(i: int):
            for j in range(10):
                log.append("concurrent", {"worker": i, "j": j})

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        res = log.verify()
        self.assertTrue(res["valid"], res["problems"])
        self.assertEqual(res["n_entries"], 60)

    def test_second_log_instance_sees_appends_from_the_first(self):
        """Hai instance trên cùng file (CLI tạo mới mỗi lệnh) không được làm đứt chuỗi."""
        path = _TMP / f"audit_two_{new_id('a')}.jsonl"
        a, b = AuditLog(path), AuditLog(path)
        a.append("x", {"i": 0})
        b.append("x", {"i": 1})          # b phải đọc lại đuôi file, không dùng head cũ
        a.append("x", {"i": 2})
        self.assertTrue(AuditLog(path).verify()["valid"])

    def test_corrupt_audit_line_reported_not_crashed(self):
        path = _TMP / f"audit_bad_{new_id('a')}.jsonl"
        log = AuditLog(path)
        log.append("x", {"i": 0})
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("{khong phai json}\n")
        res = log.verify()
        self.assertFalse(res["valid"])
        self.assertTrue(res["problems"])


class TestRecognizerOutputContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enc = ClassicalEncoder()
        cls.splits = build_splits(_INDEX, k_shot=3,
                                  class_seed=1, calib_test_seed=123)
        cls.bank = PrototypeBank(cls.enc)
        for c, paths in cls.splits.support.items():
            cls.bank.register(c, paths, label=c)
        true_d = {}
        for c, paths in cls.splits.calib.items():
            E = cls.enc.embed_paths(paths)
            D, skus = cls.bank.distances_batch(E)
            true_d[c] = [float(x) for x in D[:, skus.index(c)]]
        cls.calib = Calibrator(cls.enc.tag, alpha=0.10).fit(true_d)

    def test_prediction_set_is_never_truncated_by_top_k(self):
        from dataclasses import replace
        accepted = set(self.bank.sku_ids[:10])
        distances = {sku: .1 if sku in accepted else 2.0 for sku in self.bank.sku_ids}
        calibration = replace(self.calib, conformal_quantile={}, global_conformal_quantile=1.0)
        result = Recognizer(self.enc, self.bank, calibration, top_k=3).recognize_distances(distances)
        self.assertEqual(result.status, 'ambiguous')
        self.assertEqual(set(result.prediction_set), accepted)
        self.assertEqual({c.sku_id for c in result.candidates}, accepted)
        self.assertTrue(all(c.in_prediction_set for c in result.candidates))

    def test_accepted_candidate_outside_nearest_top_k_is_still_shown(self):
        from dataclasses import replace
        skus = self.bank.sku_ids[:4]
        distances = {sku: 2.0 for sku in self.bank.sku_ids}
        distances.update(dict(zip(skus, [.1, .2, .3, .4])))
        calibration = replace(self.calib, global_conformal_quantile=.01,
                              conformal_quantile={skus[-1]: .5})
        result = Recognizer(self.enc, self.bank, calibration, top_k=2).recognize_distances(distances)
        self.assertEqual(result.prediction_set, [skus[-1]])
        self.assertEqual([c.sku_id for c in result.candidates], skus[:2] + skus[-1:])
        self.assertEqual(result.top1.sku_id, skus[0])
        all_result = Recognizer(self.enc, self.bank, calibration, top_k=0).recognize_distances(distances)
        self.assertEqual(len(all_result.candidates), len(self.bank))

    def test_result_json_is_valid_json_even_without_calibration(self):
        from rpv.util import json_dumps
        rec = Recognizer(self.enc, self.bank, None)
        text = json_dumps(rec.recognize(self.splits.test["cam_sanh"][0]).to_json(), indent=None)
        self.assertNotIn("Infinity", text)
        self.assertNotIn("NaN", text)
        json.loads(text)

    def test_empty_catalog_json_is_valid(self):
        from rpv.util import json_dumps
        rec = Recognizer(self.enc, PrototypeBank(self.enc), None)
        res = rec.recognize(self.splits.test["cam_sanh"][0])
        self.assertEqual(res.status, STATUS_EMPTY_CATALOG)
        self.assertEqual(res.prediction_set, [])
        json.loads(json_dumps(res.to_json(), indent=None))

    def test_candidates_include_conformal_p_value(self):
        rec = Recognizer(self.enc, self.bank, self.calib)
        res = rec.recognize(self.splits.test["cam_sanh"][0])
        self.assertTrue(all(0.0 < c.p_value <= 1.0 for c in res.candidates), res.to_json())


class TestSuspiciousMarginPolicy(unittest.TestCase):
    """Lời khai "hợp lệ về ngưỡng" nhưng SKU khác gần hơn nhiều là hình dạng của
    gian lận đổi mã tinh vi: phải gọi người, không được cho qua kèm ghi chú."""

    def test_large_margin_escalates_to_assist(self):
        from rpv.vision.verification import VerificationResult
        engine = PolicyEngine()
        repo = ProvenanceRepository(_TMP / f"db_mar_{new_id('t')}.db")
        repo.init_schema()
        try:
            repo.upsert_supplier(Supplier("sup_1", "HTX", "Long An"))
            repo.upsert_product(Product("prd_1", "X", category="fresh_produce",
                                        brand="B", image_uri="x.jpg"))
            repo.upsert_sku(SKU("sku_1", "prd_1", "X", price=1000))
            today = date.today()
            repo.upsert_lot(Lot("lot_1", "sku_1", "sup_1", origin_country="VN",
                                origin_region="R", farm_or_facility="F",
                                production_date=str(today), received_date=str(today),
                                expiry_date=str(today + timedelta(days=30))))
            for t in (EvidenceType.DELIVERY_NOTE.value, EvidenceType.SUPPLIER_DECLARATION.value):
                repo.add_evidence(Evidence(new_id("ev"), "lot_1", t,
                                           status=EvidenceStatus.VERIFIED.value))
            repo.add_event(TraceEvent(new_id("evt"), "lot_1", "receiving", event_time=str(today)))
            rec = repo.resolve("sku_1", "lot_1")

            calm = VerificationResult(VERIFY_CONFIRMED, "sku_1", 0.30, "sku_9", 0.28, 0.5,
                                      0.4, 0.02, note="gần nhau", suspicious_margin=False)
            self.assertEqual(engine.decide(verification=calm, record=rec, today=today).outcome,
                             "allow_warning")

            fishy = VerificationResult(VERIFY_CONFIRMED, "sku_1", 0.49, "sku_9", 0.10, 0.5,
                                       0.2, 0.39, note="lệch xa", suspicious_margin=True)
            d = engine.decide(verification=fishy, record=rec, today=today)
            self.assertEqual(d.outcome, "assist")
            self.assertTrue(d.requires_operator)
        finally:
            repo.close()


class TestPrototypeBankRobustness(unittest.TestCase):
    def setUp(self):
        self.enc = ClassicalEncoder()
        self.paths = sorted((_DATA / "cam_sanh").glob("*.jpg"))[:5]

    def test_unknown_fields_in_saved_bank_are_ignored(self):
        bank = PrototypeBank(self.enc, centroids_per_sku=1)
        bank.register("cam_sanh", self.paths)
        path = _TMP / f"bank_fwd_{new_id('b')}.json"
        bank.save(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["entries"]["cam_sanh"]["truong_phien_ban_sau"] = "x"
        path.write_text(json.dumps(raw), encoding="utf-8")
        self.assertEqual(len(PrototypeBank(self.enc).load(path)), 1)

    def test_future_format_version_rejected(self):
        bank = PrototypeBank(self.enc, centroids_per_sku=1)
        bank.register("cam_sanh", self.paths)
        path = _TMP / f"bank_ver_{new_id('b')}.json"
        bank.save(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["_meta"]["format_version"] = 99
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(BankIncompatibleError):
            PrototypeBank(self.enc).load(path)

    def test_summary_reports_least_consistent_skus(self):
        bank = PrototypeBank(self.enc, centroids_per_sku=1)
        bank.register("cam_sanh", self.paths)
        bank.register("khoai_lang_mat", sorted((_DATA / "khoai_lang_mat").glob("*.jpg"))[:5])
        worst = bank.summary()["most_inconsistent_skus"]
        self.assertEqual(len(worst), 2)
        self.assertGreaterEqual(worst[0]["intra_spread"], worst[1]["intra_spread"])


class TestEpcisDocument(unittest.TestCase):
    def test_creation_date_is_present(self):
        """creationDate là trường bắt buộc của EPCISDocument; bản trước để None
        nên tài liệu xuất ra không hợp lệ với schema GS1."""
        from rpv.provenance.adapters import EpcisExporter
        doc = EpcisExporter.export([TraceEvent("evt_1", "lot_1")], "urn:rpv:lot:lot_1")
        self.assertTrue(doc["creationDate"])
        self.assertIn("T", doc["creationDate"])


class TestConfigValidation(unittest.TestCase):
    def test_bad_env_value_fails_loudly(self):
        from rpv.config import VisionConfig
        os.environ["RPV_BATCH_SIZE"] = "khong-phai-so"
        try:
            with self.assertRaises(ValueError):
                VisionConfig()
        finally:
            os.environ.pop("RPV_BATCH_SIZE", None)

    def test_reload_settings_picks_up_environment(self):
        from rpv.config import reload_settings
        os.environ["RPV_TOP_K"] = "9"
        try:
            self.assertEqual(reload_settings().vision.top_k, 9)
        finally:
            os.environ.pop("RPV_TOP_K", None)
            reload_settings()


class TestDotEnvWindows(unittest.TestCase):
    def test_launcher_inline_comments_are_repaired_without_replacing_overrides(self):
        from unittest.mock import patch
        from rpv.config import load_env_file, VisionConfig
        with tempfile.TemporaryDirectory() as td:
            env_file = Path(td) / '.env'
            env_file.write_text('RPV_BATCH_SIZE=16\nRPV_ENCODER=dinov2\nRPV_TEST_PATH=data/images\n',
                                encoding='utf-8')
            with patch.dict(os.environ, {'RPV_BATCH_SIZE': '16              # giữ batch size hiện tại',
                                         'RPV_ENCODER': 'dinov2 # backbone',
                                         'RPV_TEST_PATH': 'data/My # photos'}):
                load_env_file(env_file)
                self.assertEqual(VisionConfig().batch_size, 16)
                self.assertEqual(VisionConfig().encoder, 'dinov2')
                self.assertEqual(os.environ['RPV_TEST_PATH'], 'data/My # photos')
                os.environ['RPV_BATCH_SIZE'] = '64'
                load_env_file(env_file)
                self.assertEqual(VisionConfig().batch_size, 64)

    def test_windows_paths_comments_and_precedence(self):
        from rpv.config import load_env_file
        with tempfile.TemporaryDirectory() as td:
            env_file = Path(td) / ".env"
            env_file.write_text(
                "\ufeffRPV_TEST_WIN_PATH=C:\\Users\\Dong\\Pictures\\rpv # comment\n"
                'RPV_TEST_HASH="C:\\data\\#images"\n'
                "export RPV_TEST_EMPTY=\n",
                encoding="utf-8",
            )
            old = os.environ.get("RPV_TEST_WIN_PATH")
            os.environ["RPV_TEST_WIN_PATH"] = "keep-me"
            try:
                load_env_file(env_file)
                self.assertEqual(os.environ["RPV_TEST_WIN_PATH"], "keep-me")
                self.assertEqual(os.environ["RPV_TEST_HASH"], r"C:\data\#images")
                self.assertEqual(os.environ["RPV_TEST_EMPTY"], "")
            finally:
                for key in ("RPV_TEST_WIN_PATH", "RPV_TEST_HASH", "RPV_TEST_EMPTY"):
                    os.environ.pop(key, None)
                if old is not None:
                    os.environ["RPV_TEST_WIN_PATH"] = old

class TestDeepReviewRegressions(unittest.TestCase):
    def test_invalid_evidence_date_fails_closed(self):
        e = Evidence("e", "lot", EvidenceType.DELIVERY_NOTE.value,
                     status=EvidenceStatus.VERIFIED.value, valid_until="not-a-date")
        self.assertFalse(e.is_currently_valid())

    def test_invalid_lot_date_and_negative_quantity_rejected(self):
        repo = ProvenanceRepository(_TMP / f"validation_{new_id('t')}.db")
        repo.init_schema()
        repo.upsert_supplier(Supplier("sup", "Supplier"))
        repo.upsert_product(Product("prd", "Product"))
        repo.upsert_sku(SKU("sku", "prd", "SKU"))
        try:
            with self.assertRaises(ValueError):
                repo.upsert_lot(Lot("bad-date", "sku", "sup", expiry_date="31/12/2026"))
            with self.assertRaises(ValueError):
                repo.upsert_lot(Lot("bad-qty", "sku", "sup", quantity=-1))
        finally:
            repo.close()

    def test_lot_sku_conflict_marks_provenance_blocked(self):
        from rpv.provenance.repository import ProvenanceRecord
        rec = ProvenanceRecord(
            sku=SKU("sku-a", "prd", "A"),
            lot=Lot("lot-b", "sku-b", "sup", origin_country="VN"),
        )
        decision = PolicyEngine().decide(record=rec)
        self.assertEqual(decision.outcome, "block")
        self.assertEqual(decision.provenance_status, "provenance_blocked")

    def test_dataset_needs_train_support_calib_and_test_per_class(self):
        paths = [f'{i}.jpg' for i in range(4)]
        index = DatasetIndex({'sku': paths}, records={p: {'group_id': p, 'path': p} for p in paths})
        with self.assertRaisesRegex(ValueError, 'calib'):
            build_splits(index, k_shot=3, class_seed=1, calib_test_seed=2)

    def test_two_audit_instances_share_lock_under_concurrency(self):
        from concurrent.futures import ThreadPoolExecutor
        path = _TMP / f"audit_multi_{new_id('t')}.jsonl"
        logs = [AuditLog(path), AuditLog(path)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: logs[i % 2].append("event", {"i": i}), range(100)))
        result = AuditLog(path).verify()
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["n_entries"], 100)


class TestCliCheckpointOverride(unittest.TestCase):
    def test_blank_checkpoint_clears_configured_head_for_baseline(self):
        from unittest.mock import patch
        from rpv.cli import _apply_overrides, _settings, build_parser
        from rpv.config import reload_settings
        try:
            with patch.dict(os.environ, {'RPV_PROJECTION_CKPT': 'existing-head.pt'}):
                args = build_parser().parse_args(['evaluate', '--encoder', 'classical', '--ckpt', ''])
                _apply_overrides(args)
                self.assertEqual(_settings().vision.checkpoint, '')
        finally:
            reload_settings()


def tearDownModule():
    _TEMP_DIR.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
