"""Regression checks for the supervisor's policy, OOD, risk and latency requirements."""
from dataclasses import replace
from datetime import date, timedelta
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from rpv import config as app_config
from rpv.audit.hash_chain import AuditLog
from rpv.domain.models import Product, SKU, Supplier, Lot, Evidence
from rpv.evaluation.performance import benchmark_checkout, summarize_timings
from rpv.evaluation.robustness import evaluate_robustness
from rpv.provenance.policy import PolicyEngine
from rpv.provenance.repository import ProvenanceRecord, ProvenanceRepository
from rpv.services.checkout import CheckoutService
from rpv.vision.calibration import CalibrationModel
from rpv.vision.prototypes import PrototypeBank
from rpv.vision.recognizer import Recognizer
from rpv.vision.rejection import energy_score, fit_risk_threshold
from rpv.vision.verification import VerificationResult, Verifier
from tests.test_thesis import PixelEncoder


class TestRejection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='rpv_review_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.encoder = PixelEncoder()
        self.bank = PrototypeBank(self.encoder)
        for label, color in [('a', 'red'), ('b', 'blue')]:
            path = self.root / f'{label}.png'
            Image.new('RGB', (10, 10), color).save(path)
            with patch.object(self.encoder, '_cache', return_value=None):
                self.bank.register(label, [str(path)])
        self.calibration = CalibrationModel(self.encoder.tag, .1, .5, global_conformal_quantile=.5,
                                            ood={'method': 'energy', 'temperature': .05, 'threshold': .01})

    def test_energy_uses_absolute_logits_and_is_numerically_stable(self):
        self.assertAlmostEqual(energy_score([1, 1], 1), 1 - np.log(2))
        self.assertGreater(energy_score([100, 100], 1), energy_score([1, 1], 1))
        for values in ([], [float('nan')], [-1]):
            with self.assertRaises(ValueError):
                energy_score(values)

    def test_ood_overrides_distance_acceptance_in_recognizer_and_verifier(self):
        distances = {'a': .4, 'b': .9}
        recognition = Recognizer(self.encoder, self.bank, self.calibration).recognize_distances(distances)
        verification = Verifier(self.encoder, self.bank, self.calibration).verify_distances(distances, 'a')
        self.assertEqual(recognition.status, 'unknown')
        self.assertEqual(recognition.prediction_set, [])
        self.assertEqual(recognition.conformal_prediction_set, ['a'])
        self.assertEqual(verification.status, 'out_of_distribution')
        self.assertFalse(verification.accepted)

    def test_risk_threshold_cannot_split_ties_and_empty_selection_never_accepts(self):
        risk = fit_risk_threshold([True, False], [1, 1], [True, True], .01)
        self.assertIsNone(risk['confidence_threshold'])
        self.calibration.risk_control = risk
        self.calibration.ood['threshold'] = 10
        distances = {'a': .1, 'b': .9}
        self.assertEqual(Recognizer(self.encoder, self.bank, self.calibration).recognize_distances(distances).status, 'ambiguous')
        self.assertEqual(Verifier(self.encoder, self.bank, self.calibration).verify_distances(distances, 'a').status, 'ambiguous')

    def test_calibration_artifact_preserves_energy_and_empty_risk_gate(self):
        self.calibration.risk_control = fit_risk_threshold([False], [.2], [True])
        path = self.root / 'calibration.json'
        self.calibration.save(path)
        restored = CalibrationModel.load(path)
        self.assertEqual(restored.ood, self.calibration.ood)
        self.assertFalse(restored.risk_accepts({'a': .01, 'b': 1.0}))

    def test_benchmark_isolates_operational_database_and_audit(self):
        with ProvenanceRepository(self.root / 'operational.db') as repo:
            repo.init_schema()
            audit = AuditLog(self.root / 'operational.jsonl')
            original_audit = audit.path.read_bytes()
            service = CheckoutService(self.encoder, self.bank, repo, self.calibration, audit=audit)
            settings = replace(app_config.settings, paths=replace(app_config.settings.paths, cache_dir=self.root / 'cache'))
            with patch.object(app_config, 'settings', settings):
                report = benchmark_checkout(service, [str(self.root / 'a.png')], repeats=2, warmup=1)
                self.assertIs(app_config.settings, settings)
            self.assertEqual(repo.stats()['decisions'], 0)
            self.assertEqual(audit.path.read_bytes(), original_audit)
            self.assertTrue(report['modes']['uncached']['audit_valid'])
            self.assertEqual(report['modes']['uncached']['summary']['total']['n'], 2)

    def test_free_recognition_resolves_the_accepted_sku_not_raw_nearest(self):
        self.calibration.ood['threshold'] = 10
        self.calibration.conformal_quantile = {'a': .01, 'b': .5}
        with ProvenanceRepository(self.root / 'resolve.db') as repo:
            repo.init_schema()
            repo.upsert_product(Product('p', 'B'))
            repo.upsert_sku(SKU('b', 'p', 'B'))
            service = CheckoutService(self.encoder, self.bank, repo, self.calibration,
                                      audit=AuditLog(self.root / 'resolve.jsonl'))
            with patch.object(self.encoder, '_cache', return_value=None), \
                    patch.object(self.bank, 'distances', return_value={'a': .1, 'b': .3}):
                response = service.process(self.root / 'a.png')
            self.assertEqual(response.recognition['prediction_set'], ['b'])
            self.assertEqual(response.recognition['top1']['sku_id'], 'a')
            self.assertEqual(response.provenance['sku']['sku_id'], 'b')

    def test_robustness_reads_only_held_out_images_and_keeps_thresholds(self):
        from rpv.data.dataset import DatasetSplits
        query = self.root / 'query.png'
        Image.new('RGB', (12, 12), 'red').save(query)
        # Build a minimal split using distinct support and test files.
        splits = DatasetSplits(support={'a': [str(self.root / 'a.png')]}, calib={},
                               test={'a': [str(query)]}, unknown_calib=[], unknown_test=[], train_classes=['a'])
        before = self.calibration.to_json()
        with patch.object(self.encoder, '_cache', return_value=None):
            report = evaluate_robustness(self.encoder, self.bank, self.calibration, splits, max_per_class=1)
        self.assertEqual(len(report['conditions']), 6)
        self.assertEqual(self.calibration.to_json(), before)


class TestDecisionStates(unittest.TestCase):
    def setUp(self):
        self.today = date(2026, 10, 9)
        self.record = ProvenanceRecord(
            sku=SKU('a', 'p', 'A'), product=Product('p', 'A', category='fresh_produce', brand='HTX', image_uri='a.jpg'),
            supplier=Supplier('s', 'S', address='VN'),
            lot=Lot('lot', 'a', 's', origin_country='VN', expiry_date='2026-11-01', production_date='2026-10-01', trace_code='trace'),
            evidence=[Evidence(f'ev_{kind}', 'lot', kind, status='verified', valid_until='2026-10-10')
                      for kind in ('delivery_note', 'supplier_declaration')])
        self.verification = VerificationResult('confirmed', 'a', .1, 'a', .1, .5, .9, 0)
        self.policy = PolicyEngine()

    def decide(self, verification=None):
        return self.policy.decide(verification=verification or self.verification, record=self.record, today=self.today)

    def test_four_states(self):
        self.assertEqual(self.decide().decision_state, 'match')
        wrong = replace(self.verification, status='mismatch', best_sku='b')
        self.assertEqual(self.decide(wrong).decision_state, 'mismatch')
        unknown = replace(self.verification, status='out_of_distribution')
        self.assertEqual(self.decide(unknown).decision_state, 'abstain')
        self.record.evidence = []
        self.assertEqual(self.decide().decision_state, 'provenance_exception')

    def test_policy_uses_supplied_business_date_for_certificate_validity(self):
        self.assertEqual(self.decide().provenance_status, 'provenance_ok')
        result = self.policy.decide(verification=self.verification, record=self.record, today=self.today + timedelta(days=2))
        self.assertEqual(result.decision_state, 'provenance_exception')
        self.assertEqual(result.outcome, 'assist')
        self.assertIn('PROVENANCE_EXPIRED', result.reasons)

    def test_missing_expiry_and_same_sku_expired_lot_cannot_match(self):
        self.record.lot.expiry_date = None
        self.assertEqual(self.decide().decision_state, 'provenance_exception')
        self.record.lot = replace(self.record.lot, lot_id='another_lot', expiry_date='2026-10-01')
        self.assertEqual(self.decide().outcome, 'block')
        self.assertEqual(self.decide().decision_state, 'provenance_exception')

    def test_absent_vision_never_allows_checkout(self):
        result = self.policy.decide(record=self.record, today=self.today)
        self.assertEqual(result.outcome, 'assist')
        self.assertEqual(result.decision_state, 'abstain')

    def test_budgets_use_strict_p95_limits(self):
        result = summarize_timings([{'vision_search': 150, 'policy_lookup': 49, 'total': 299}])
        self.assertFalse(result['vision_search']['meets_budget_p95'])
        self.assertTrue(result['total']['meets_budget_p95'])
