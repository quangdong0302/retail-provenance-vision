"""Regressions for catalog binding, refusal and reported coverage semantics."""
from dataclasses import replace
from datetime import date
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rpv import config as app_config
from rpv.data.dataset import DatasetSplits
from rpv.domain.models import SKU, Lot
from rpv.evaluation.pipeline import evaluate_system
from rpv.provenance.policy import PolicyEngine
from rpv.provenance.repository import ProvenanceRecord
from rpv.vision.calibration import CalibrationModel
from rpv.vision.prototypes import PrototypeBank, PrototypeEntry
from rpv.vision.recognizer import Recognizer, RecognitionResult
from rpv.vision.verification import Verifier, VerificationResult
from tests.test_prototype_improvements import TinyEncoder


def catalog():
    encoder = TinyEncoder()
    bank = PrototypeBank(encoder, 1)
    bank.entries = {sku: PrototypeEntry(sku, np.array([v], np.float32), 2)
                    for sku, v in [('a', [1., 0.]), ('b', [0., 1.])]}
    calibration = CalibrationModel(encoder.tag, .1, 2., global_conformal_quantile=2.,
                                   decision_rule='prototype_selective')
    return encoder, bank, calibration


class TestDecisionIntegrity(unittest.TestCase):
    def test_verifier_never_confirms_selective_mode_without_risk_gate(self):
        encoder, bank, calibration = catalog()
        row = {'a': .1, 'b': 1.}
        self.assertEqual(Recognizer(encoder, bank, calibration).recognize_distances(row).status, 'ambiguous')
        result = Verifier(encoder, bank, calibration).verify_distances(row, 'a')
        self.assertFalse(result.accepted)
        self.assertEqual(result.status, 'ambiguous')

    def test_operator_required_is_enforced_by_legacy_verifier_too(self):
        encoder, bank, calibration = catalog()
        calibration.decision_rule = 'conformal'
        calibration.global_conformal_quantile = .5
        bank.entries['a'].requires_operator = True
        result = Verifier(encoder, bank, calibration).verify_distances({'a': .1, 'b': 1.}, 'a')
        self.assertFalse(result.accepted)

    def test_empty_distance_map_cannot_impersonate_an_empty_catalog(self):
        encoder, bank, calibration = catalog()
        with self.assertRaises(ValueError):
            Recognizer(encoder, bank, calibration).recognize_distances({})
        with self.assertRaises(ValueError):
            Verifier(encoder, bank, calibration).verify_distances({}, 'a')

    def test_tie_and_operator_handoff_have_distinct_audit_reasons(self):
        encoder, bank, calibration = catalog()
        calibration.risk_control = {'enabled': True, 'confidence_threshold': -10.}
        recognizer = Recognizer(encoder, bank, calibration)
        self.assertEqual(recognizer.recognize_distances({'a': .1, 'b': .1}).rejection_reason, 'tie')
        bank.entries['a'].requires_operator = True
        self.assertEqual(recognizer.recognize_distances({'a': .1, 'b': 1.}).rejection_reason, 'operator_required')

    def record_b(self):
        return ProvenanceRecord(sku=SKU('b', 'product', 'B'),
                                lot=Lot('lot_b', 'b', 'supplier', origin_country='VN', expiry_date='2027-01-01'))

    def test_visual_sku_a_cannot_authorize_record_for_sku_b(self):
        recognition = RecognitionResult('match', [], ['a'])
        result = PolicyEngine().decide(recognition=recognition, record=self.record_b(), today=date(2026, 10, 15))
        self.assertEqual((result.outcome, result.decision_state), ('block', 'mismatch'))
        self.assertIn('VISION_MISMATCH', result.reasons)
        self.assertFalse(result.audit_notes['decision_inputs']['identity_confident'])

    def test_verified_claim_cannot_be_rebound_to_another_sku_record(self):
        verification = VerificationResult('confirmed', 'a', .1, 'a', .1, .5, .9, 0.)
        result = PolicyEngine().decide(verification=verification, record=self.record_b(), today=date(2026, 10, 15))
        self.assertEqual((result.outcome, result.decision_state), ('block', 'mismatch'))
        self.assertFalse(result.audit_notes['decision_inputs']['identity_confident'])

    def test_invalid_match_without_one_sku_requires_an_operator(self):
        result = PolicyEngine().decide(recognition=RecognitionResult('match', [], []),
                                       record=self.record_b(), today=date(2026, 10, 15))
        self.assertEqual((result.outcome, result.decision_state), ('assist', 'abstain'))


class TestEvaluationSemantics(unittest.TestCase):
    def test_conformal_coverage_after_ood_is_not_top1_accuracy(self):
        encoder, bank, calibration = catalog()
        calibration.ood = {'method': 'energy', 'temperature': .1, 'threshold': 10.}
        calibration.risk_control = {'enabled': True, 'confidence_threshold': -10., 'target_risk': .04}
        splits = DatasetSplits(support={'a': ['a_support'], 'b': ['b_support']},
                               test={'a': ['a_query'], 'b': ['b_query']},
                               unknown_test=['unknown_query'], train_classes=['a', 'b'])
        with patch.object(encoder, 'embed_paths', side_effect=lambda paths: np.tile([1., 0.], (len(paths), 1))):
            result, _, _ = evaluate_system(encoder=encoder, splits=splits, alpha=.1,
                                            catalog=(bank, calibration), verbose=False)
        self.assertEqual(result.selective['empirical_coverage_after_ood'], 1.)
        self.assertEqual(result.selective['decision_set_label_coverage'], .5)
        for row in result.predictions:
            self.assertIn('conformal_prediction_set', row)
            self.assertIn('conformal_prediction_set_after_ood', row)
            self.assertIn('confidence', row)
        self.assertEqual(result.selective['operating_point_known']['target_risk'], .04)
        self.assertFalse(result.selective['operating_point_known_and_unknown']['target_met_on_test'])


class TestNotebookArtifactBinding(unittest.TestCase):
    def setUp(self):
        from notebooks.notebook_utils import Workflow
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.head = self.root / 'head.pt'
        self.head.write_bytes(b'first_head')
        training = dict(backbone_tag='fixture', out_dim=2)
        tag = 'fixture_proj2_' + hashlib.sha256(self.head.read_bytes()).hexdigest()[:16] + '_tta0'
        encoder, self.bank, self.calibration = catalog()
        encoder.tag = self.calibration.encoder_tag = tag
        self.calibration.bank_fingerprint = self.bank.fingerprint()
        self.bank_path = self.root / 'prototypes.json'
        self.calib_path = self.root / 'calibration.json'
        self.bank.save(self.bank_path)
        self.calibration.save(self.calib_path)
        report_dir = self.root / 'report'
        report_dir.mkdir()
        self.calibration.save(report_dir / 'calibration.json')
        self.result_path = self.root / 'evaluate_head.json'
        self.result_path.write_text(json.dumps(dict(encoder=dict(tag=tag),
                                    reports=dict(json=str(report_dir / 'evaluation.json')))), encoding='utf-8')
        self.workflow = Workflow.__new__(Workflow)
        self.workflow.head, self.workflow.folder = self.head, self.root
        self.workflow.settings = replace(app_config.settings, vision=replace(app_config.settings.vision,
                                                prototype_color_weight=0., use_tta=False))
        self.workflow.environment = lambda: {'RPV_BANK': str(self.bank_path), 'RPV_CALIB': str(self.calib_path)}
        guard = patch.object(self.workflow, 'require_head', return_value=training)
        guard.start()
        self.addCleanup(guard.stop)

    def test_matching_artifacts_can_be_read_without_loading_a_backbone(self):
        self.assertEqual(self.workflow.evaluation_result()['encoder']['tag'], self.calibration.encoder_tag)

    def test_retraining_invalidates_old_calibration(self):
        self.head.write_bytes(b'retrained_head')
        with self.assertRaises(RuntimeError):
            self.workflow.require_calibration()

    def test_changed_prototype_invalidates_old_thresholds(self):
        self.bank.entries['a'].centroids = np.array([[0., 1.]], np.float32)
        self.bank.save(self.bank_path)
        with self.assertRaises(RuntimeError):
            self.workflow.require_calibration()

    def test_same_head_with_changed_thresholds_invalidates_old_report(self):
        self.calibration.ood = {'method': 'distance', 'threshold': .3}
        self.calibration.save(self.calib_path)
        with self.assertRaises(RuntimeError):
            self.workflow.evaluation_result()

    def test_old_head_report_is_not_used_for_plots_and_comparison(self):
        result = json.loads(self.result_path.read_text(encoding='utf-8'))
        result['encoder']['tag'] = 'old_head'
        self.result_path.write_text(json.dumps(result), encoding='utf-8')
        with self.assertRaises(RuntimeError):
            self.workflow.evaluation_result()


class TestConfidenceStudy(unittest.TestCase):
    def test_risk_uses_explicit_unknown_prevalence(self):
        from scripts.review_confidence import mixture_risk
        self.assertAlmostEqual(mixture_risk(.5, .01, .02, 0.), .02)
        self.assertAlmostEqual(mixture_risk(.5, .01, .02, .5), .015 / .26)
        self.assertAlmostEqual(mixture_risk(.5, .01, .02, .75), .0175 / .14)

    def test_abstaining_everywhere_is_not_reported_as_zero_risk(self):
        from scripts.review_confidence import measure
        d = np.array([[.1, .8]], np.float32)
        result = measure(d, np.array([0]), d, np.array([.01]), np.array([.01]), 1., None, None)
        self.assertEqual(result['known_coverage'], 0.)
        for name in ('known_risk', 'natural_mixture_risk', 'balanced_mixture_risk'):
            self.assertIsNone(result[name])

    def test_hybrid_has_separate_distance_and_margin_information(self):
        from scripts.review_confidence import score
        d = np.array([[.1, .4], [.4, 1.6]], np.float32)
        energies = np.array([.01, .16])
        np.testing.assert_allclose(score(d, energies, 0.), [.75, .75])
        self.assertGreater(score(d, energies, 1.)[0], score(d, energies, 1.)[1])


class TestDatasetPublication(unittest.TestCase):
    def test_transient_windows_lock_retries_then_publishes_intact(self):
        from rpv.data.managed import _publish_dataset
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            staged = root / 'rpv_prepare_fixture' / 'dataset'
            staged.mkdir(parents=True)
            (staged / 'manifest.json').write_text('content', encoding='utf-8')
            original_rename = Path.rename
            error = PermissionError('temporary sharing lock')
            error.winerror = 5
            calls = []
            def rename(path, target):
                calls.append(target)
                if len(calls) == 1:
                    raise error
                return original_rename(path, target)
            with patch.object(Path, 'rename', autospec=True, side_effect=rename), patch('time.sleep'):
                _publish_dataset(staged, root / 'prepared')
            self.assertEqual(len(calls), 2)
            self.assertEqual((root / 'prepared/manifest.json').read_text(encoding='utf-8'), 'content')

    def test_existing_destination_is_not_overwritten(self):
        from rpv.data.managed import _publish_dataset
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            staged = root / 'rpv_prepare_fixture' / 'dataset'
            staged.mkdir(parents=True)
            destination = root / 'prepared'
            destination.mkdir()
            (destination / 'user.txt').write_text('keep', encoding='utf-8')
            with self.assertRaises(ValueError):
                _publish_dataset(staged, destination)
            self.assertEqual((destination / 'user.txt').read_text(encoding='utf-8'), 'keep')

    def test_persistent_permission_error_is_not_swallowed(self):
        from rpv.data.managed import _publish_dataset
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            staged = root / 'rpv_prepare_fixture' / 'dataset'
            staged.mkdir(parents=True)
            error = PermissionError('persistent permission failure')
            error.winerror = 5
            with patch.object(Path, 'rename', side_effect=error) as rename, patch('time.sleep'), self.assertRaises(PermissionError):
                _publish_dataset(staged, root / 'prepared')
            self.assertEqual(rename.call_count, 5)
            self.assertTrue(staged.is_dir())
