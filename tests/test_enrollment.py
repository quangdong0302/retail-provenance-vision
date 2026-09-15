"""Enrollment regressions: data separation, persisted revisions, and operator-only publication."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from rpv.audit.hash_chain import AuditLog
from rpv.provenance.policy import PolicyDecision
from rpv.provenance.repository import ProvenanceRepository
from rpv.services.checkout import CheckoutService
from rpv.services.enrollment import EnrollmentCriteria, EnrollmentService
from rpv.vision.calibration import CalibrationModel
from rpv.vision.encoders import BaseEncoder, l2_normalize
from rpv.vision.prototypes import PrototypeBank
from rpv.vision.recognizer import Recognizer


class MeanEncoder(BaseEncoder):
    tag = 'enrollment_test'
    dim = 3

    def embed_arrays(self, images):
        return l2_normalize(np.asarray([np.asarray(im).mean(axis=(0, 1)) for im in images], dtype=np.float32))

    def embed_path(self, path):
        with Image.open(path) as im:
            return self.embed_arrays([np.asarray(im.convert('RGB'))])[0]


class AlwaysAllow:
    def decide(self, **kwargs):
        return PolicyDecision('allow')


class EnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='rpv_enrollment_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.encoder = MeanEncoder()
        self.bank = PrototypeBank(self.encoder, 1)
        self.serial = 0
        self.bank.register('green', [self.photo('green')])
        self.bank.register('blue', [self.photo('blue')])
        self.criteria = EnrollmentCriteria(min_side=8, min_sharpness=0, min_validation=4, min_negative=4)
        self.enrollment = EnrollmentService(self.encoder, self.bank, self.root / 'enrollment', self.criteria)
        self.session = self.enrollment.create('red', 'Mặt hàng đỏ')['id']

    def photo(self, color):
        self.serial += 1
        pixels = np.random.default_rng(self.serial).integers(0, 60, size=(16, 16, 3), dtype=np.uint8)
        pixels[:, :, {'red': 0, 'green': 1, 'blue': 2}[color]] += 170
        path = self.root / f'photo_{self.serial}.png'
        Image.fromarray(pixels).save(path)
        return path

    def add(self, role, color, count, group):
        result = self.enrollment.add_images(self.session, role, [self.photo(color) for _ in range(count)], group,
                                            color if role == 'negative' else '')
        self.assertFalse(result['rejected'], result)
        return result

    def complete(self):
        self.add('support', 'red', 2, 'support_a')
        self.add('support', 'red', 1, 'support_b')
        self.add('validation', 'red', 2, 'val_a')
        self.add('validation', 'red', 2, 'val_b')
        self.add('negative', 'green', 2, 'neg_a')
        self.add('negative', 'blue', 2, 'neg_b')

    def test_quality_and_duplicate_images_rejected_without_using_validation(self):
        photo = self.photo('red')
        self.enrollment.add_images(self.session, 'support', [photo], 'a')
        renamed = self.root / 'renamed.png'
        with Image.open(photo) as im:
            im.save(renamed, compress_level=0)
        result = self.enrollment.add_images(self.session, 'validation', [renamed], 'b')
        self.assertEqual(len(result['rejected']), 1)
        self.assertEqual(len(result['session']['samples']), 1)
        bad = self.root / 'tiny.png'
        Image.new('RGB', (2, 2)).save(bad)
        self.assertEqual(len(self.enrollment.add_images(self.session, 'support', [bad], 'a')['rejected']), 1)

    def test_same_group_cannot_cross_roles_and_catalog_samples_cannot_validate(self):
        self.add('support', 'red', 1, 'object_1')
        with self.assertRaisesRegex(ValueError, 'nhiều vai trò'):
            self.add('validation', 'red', 1, 'object_1')
        original_bank_photo = self.root / 'photo_1.png'
        result = self.enrollment.add_images(self.session, 'negative', [original_bank_photo], 'new_group', 'green')
        self.assertEqual(len(result['rejected']), 1)

    def test_assessment_uses_heldout_images_and_does_not_mutate_live_bank(self):
        before = self.bank.fingerprint()
        self.complete()
        d = self.enrollment.assess(self.session)
        r = d['report']
        self.assertEqual(r['status'], 'validation_passed')
        self.assertEqual(r['operating_mode'], 'operator_required')
        self.assertEqual(r['accuracy'], 1)
        self.assertEqual(r['negative_assignment_rate'], 0)
        self.assertEqual(len(r['predictions']), 8)
        self.assertEqual(r['counts']['support'], 3)
        self.assertEqual(self.bank.fingerprint(), before)
        self.assertNotIn('red', self.bank.entries)

    def test_missing_negative_examples_cannot_pass(self):
        self.add('support', 'red', 3, 's')
        self.add('validation', 'red', 4, 'v')
        report = self.enrollment.assess(self.session)['report']
        self.assertEqual(report['status'], 'needs_samples')
        self.assertIsNone(report['negative_assignment_rate'])

    def test_history_compares_only_same_validation_and_catalog(self):
        self.complete()
        self.enrollment.assess(self.session)
        self.add('support', 'red', 1, 'support_c')
        after = self.enrollment.assess(self.session)['report']
        self.assertTrue(after['comparison']['comparable'])
        self.assertEqual(after['comparison']['accuracy_delta'], 0)
        self.add('validation', 'red', 1, 'val_c')
        self.assertFalse(self.enrollment.assess(self.session)['report']['comparison']['comparable'])

    def test_persistence_and_stale_assessment(self):
        self.complete()
        self.enrollment.assess(self.session)
        restored = EnrollmentService(self.encoder, self.bank, self.root / 'enrollment')
        self.assertTrue(restored.get(self.session)['assessment_current'])
        self.add('support', 'red', 1, 's3')
        with self.assertRaisesRegex(ValueError, 'đổi'):
            restored.publish(self.session, lambda *_: self.fail('must not publish'))
        self.enrollment.assess(self.session)
        self.bank.register('other', [self.photo('blue')])
        self.assertFalse(restored.get(self.session)['assessment_current'])
        with self.assertRaises(ValueError):
            restored.publish(self.session, lambda *_: self.fail('must not publish'))

    def test_validation_cannot_be_deleted_to_hide_errors(self):
        self.complete()
        sample = next(s for s in self.enrollment.get(self.session)['samples'] if s['role'] == 'validation')
        with self.assertRaisesRegex(ValueError, 'Chỉ được bỏ'):
            self.enrollment.remove_support(self.session, sample['id'])

    def test_changed_stored_image_invalidates_assessment(self):
        sample_id = self.add('support', 'red', 1, 's')['accepted'][0]
        self.enrollment.sample_path(self.session, sample_id).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bị sửa'):
            self.enrollment.assess(self.session)

    def test_removing_support_does_not_allow_reusing_it_as_validation(self):
        photo = self.photo('red')
        result = self.enrollment.add_images(self.session, 'support', [photo], 's')
        self.enrollment.remove_support(self.session, result['accepted'][0])
        reused = self.enrollment.add_images(self.session, 'validation', [photo], 'new_group')
        self.assertEqual(len(reused['rejected']), 1)
        with self.assertRaises(ValueError):
            self.enrollment.add_images(self.session, 'validation', [self.photo('red')], 's')

    def test_missing_hard_rival_examples_cannot_pass(self):
        self.add('support', 'red', 2, 's1')
        self.add('support', 'red', 1, 's2')
        self.add('validation', 'red', 2, 'v1')
        self.add('validation', 'red', 2, 'v2')
        self.add('negative', 'green', 2, 'n1')
        self.add('negative', 'green', 2, 'n2')
        report = self.enrollment.assess(self.session)['report']
        self.assertEqual(report['status'], 'needs_review')
        self.assertIn('blue', report['missing_rival_checks'])

    @unittest.skipUnless(importlib.util.find_spec('gradio'), 'Demo dependency optional')
    def test_lot_intake_preserves_guided_prototype_without_new_images(self):
        from unittest.mock import patch
        import app.demo as demo
        entry = self.bank.register('red', [self.photo('red')])
        entry.enrollment_id = self.session
        entry.requires_operator = True
        before = self.bank.fingerprint()
        repo = ProvenanceRepository(self.root / 'intake.db')
        repo.init_schema()
        self.addCleanup(repo.close)
        service = CheckoutService(self.encoder, self.bank, repo, audit=AuditLog(self.root / 'intake_audit.jsonl'))
        with patch.dict(demo._STATE, {'encoder': self.encoder, 'bank': self.bank, 'calib': None,
                                      'repo': repo, 'service': service}, clear=True):
            with patch.object(service, 'register_product', side_effect=AssertionError('must preserve prototype')):
                result = demo.intake('red', 'Red', 100, 'fresh_produce', 'Supplier', '', 'VN', '', '',
                                     None, None, [], False, None)
        self.assertIn('Giữ nguyên ảnh mẫu', result[0])
        self.assertEqual(self.bank.fingerprint(), before)
        self.assertTrue(self.bank.entries['red'].requires_operator)

    def test_empty_explicit_audit_is_used_without_falling_back_to_live_log(self):
        audit = AuditLog(self.root / 'isolated_audit.jsonl')
        repo = ProvenanceRepository(self.root / 'isolated.db')
        repo.init_schema()
        self.addCleanup(repo.close)
        service = CheckoutService(self.encoder, self.bank, repo, audit=audit)
        self.assertIs(service.audit, audit)
        service.audit.append('isolated_event', {})
        self.assertEqual(len(audit), 1)

    def test_legacy_registration_uses_service_bank_path(self):
        from unittest.mock import patch
        path = self.root / 'configured_bank.json'
        repo = ProvenanceRepository(self.root / 'configured.db')
        repo.init_schema()
        self.addCleanup(repo.close)
        service = CheckoutService(self.encoder, self.bank, repo,
                                  audit=AuditLog(self.root / 'configured_audit.jsonl'), bank_path=path)
        with patch.object(self.bank, 'save') as save:
            service.register_product('red', [self.photo('red')])
            save.assert_called_once_with(path)

    def test_failed_registration_save_preserves_catalog_and_calibration(self):
        from unittest.mock import patch
        path = self.root / 'failed_bank.json'
        before = self.bank.fingerprint()
        calibration = CalibrationModel(self.encoder.tag, .1, 1, bank_fingerprint=before)
        repo = ProvenanceRepository(self.root / 'failed.db')
        repo.init_schema()
        self.addCleanup(repo.close)
        service = CheckoutService(self.encoder, self.bank, repo, calibration,
                                  audit=AuditLog(self.root / 'failed_audit.jsonl'), bank_path=path)
        with patch.object(self.bank, 'save', side_effect=OSError('test: disk full')):
            with self.assertRaises(OSError):
                service.register_product('red', [self.photo('red')])
        self.assertEqual(self.bank.fingerprint(), before)
        self.assertIs(service.calibration, calibration)

    def test_removed_catalog_is_not_silently_served_from_memory(self):
        path = self.root / 'removed_bank.json'
        self.bank.save(path)
        repo = ProvenanceRepository(self.root / 'removed.db')
        repo.init_schema()
        self.addCleanup(repo.close)
        service = CheckoutService(self.encoder, self.bank, repo,
                                  audit=AuditLog(self.root / 'removed_audit.jsonl'), bank_path=path)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            service.refresh_catalog()

    def test_first_request_refreshes_bank_changed_before_service_construction(self):
        path = self.root / 'startup_race.json'
        self.bank.save(path)
        stale = PrototypeBank(self.encoder).load(path)
        self.bank.register('new_sku', [self.photo('red')])
        self.bank.save(path)
        repo = ProvenanceRepository(self.root / 'startup_race.db')
        repo.init_schema()
        self.addCleanup(repo.close)
        service = CheckoutService(self.encoder, stale, repo,
                                  audit=AuditLog(self.root / 'startup_audit.jsonl'), bank_path=path)
        service.refresh_catalog()
        self.assertIn('new_sku', service.bank)

    def test_published_product_stays_manual_after_restart_and_calibration(self):
        self.complete()
        self.enrollment.assess(self.session)
        path = self.root / 'bank.json'
        self.bank.save(path)
        old = CalibrationModel(self.encoder.tag, .1, 1, bank_fingerprint=self.bank.fingerprint())
        repo = ProvenanceRepository(self.root / 'test.db')
        repo.init_schema()
        self.addCleanup(repo.close)
        audit = AuditLog(self.root / 'audit.jsonl')
        service = CheckoutService(self.encoder, self.bank, repo, old, audit=audit, bank_path=path)
        service._enrollment = self.enrollment
        published = service.publish_enrollment(self.session)
        self.assertTrue(published['published'])
        self.assertIsNone(service.calibration)
        loaded = PrototypeBank(self.encoder).load(path)
        self.assertTrue(loaded.entries['red'].requires_operator)
        restored = CheckoutService(self.encoder, loaded, repo, old, audit=audit, bank_path=path)
        self.assertIsNone(restored.calibration)
        new = replace(old, bank_fingerprint=loaded.fingerprint(), global_conformal_quantile=.15)
        restored = CheckoutService(self.encoder, loaded, repo, new, policy=AlwaysAllow(), audit=audit, bank_path=path)
        response = restored.process(self.photo('red'), claimed_sku='red')
        self.assertEqual(response.outcome, 'assist')
        self.assertTrue(response.requires_operator)
        self.assertIn('enrollment_review_required', response.reasons)
        recognition = Recognizer(self.encoder, loaded, new).recognize(self.photo('red'))
        self.assertFalse(recognition.is_confident)
        with self.assertRaises(ValueError):
            self.enrollment.add_images(self.session, 'support', [self.photo('red')], 'new')

    @unittest.skipUnless(importlib.util.find_spec('fastapi') and importlib.util.find_spec('httpx'), 'API dependencies optional')
    def test_api_upload_survives_temporary_file_cleanup(self):
        from fastapi.testclient import TestClient
        from api.main import app, get_service
        class Stub:
            enrollment = self.enrollment

            def refresh_catalog(self):
                pass
        app.dependency_overrides[get_service] = lambda: Stub()
        self.addCleanup(app.dependency_overrides.pop, get_service)
        client = TestClient(app)
        self.addCleanup(client.close)
        photo = self.photo('red')
        response = client.post(f'/enrollment/{self.session}/images', data={'role': 'support', 'group_id': 's'},
                               files={'files': ('red.png', photo.read_bytes(), 'image/png')})
        self.assertEqual(response.status_code, 200, response.text)
        sample_id = response.json()['accepted'][0]
        self.assertEqual(client.get(f'/enrollment/{self.session}/images/{sample_id}').status_code, 200)
        self.assertEqual(client.post(f'/enrollment/{self.session}/assess').status_code, 200)


if __name__ == '__main__':
    unittest.main()
