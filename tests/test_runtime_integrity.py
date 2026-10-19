"""Regression checks for mistakes which can silently change model decisions."""
from dataclasses import replace
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from rpv import config
from rpv.audit.hash_chain import AuditLog
from rpv.provenance.repository import ProvenanceRepository
from rpv.services.checkout import CheckoutService
from rpv.vision.cache import EmbeddingCache
from rpv.vision.calibration import CalibrationModel
from rpv.vision.encoders import ClassicalEncoder, DinoV2Encoder
from rpv.vision.prototypes import PrototypeBank, PrototypeEntry
from rpv.vision.recognizer import Recognizer


class RuntimeIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='rpv_integrity_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'asymmetric.png'
        Image.fromarray(np.random.default_rng(17).integers(0, 256, (96, 96, 3), dtype=np.uint8)).save(self.image)

    def bank(self):
        encoder = ClassicalEncoder()
        bank = PrototypeBank(encoder)
        vector = np.zeros((1, encoder.dim), dtype=np.float32)
        vector[0, 0] = 1
        bank.entries['a'] = PrototypeEntry('a', vector, 1)
        return encoder, bank

    def test_bad_configuration_is_rejected_before_encoder_can_skip_batches(self):
        for value in (-1, 0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                config.VisionConfig(batch_size=value)
        for value in (float('nan'), -0.1, 1.0):
            with self.subTest(risk=value), self.assertRaises(ValueError):
                config.CalibrationConfig(target_selective_risk=value)

    def test_explicit_invalid_batch_cannot_return_zero_vectors(self):
        encoder = ClassicalEncoder()
        for batch in (-1, 0):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                encoder.embed_paths([self.image], batch_size=batch)

    def test_tta_is_frozen_per_encoder_and_bound_to_artifact_tag(self):
        without = replace(config.settings, vision=replace(config.settings.vision, use_tta=False, use_cache=False))
        with_tta = replace(without, vision=replace(without.vision, use_tta=True))
        with patch.object(config, 'settings', without):
            encoder = ClassicalEncoder()
            before = encoder.embed_path(self.image)
            bank = PrototypeBank(encoder)
            bank.register('a', [self.image])
            path = self.root / 'bank.json'
            bank.save(path)
        with patch.object(config, 'settings', with_tta):
            np.testing.assert_array_equal(encoder.embed_path(self.image), before)
            other = ClassicalEncoder()
            self.assertNotEqual(encoder.tag, other.tag)
            with self.assertRaises(RuntimeError):
                PrototypeBank(other).load(path)

    def test_nonfinite_cache_rows_are_recomputed_not_used_for_prediction(self):
        cache = EmbeddingCache('integrity', 3, self.root)
        np.savez(cache.path, keys=np.array(['bad', 'good']), dim=np.array([3]),
                 vecs=np.array([[np.nan, 0, 1], [1, 0, 0]], np.float32))
        self.assertIsNone(cache.get('bad'))
        np.testing.assert_array_equal(cache.get('good'), [1, 0, 0])
        for vector in (np.zeros(3), np.array([np.inf, 0, 0])):
            with self.assertRaises(ValueError):
                cache.put('bad', vector)

    def test_cache_cannot_associate_wrong_number_of_keys_with_vectors(self):
        cache = EmbeddingCache('integrity', 3, self.root)
        np.savez(cache.path, keys=np.array(['first', 'second']), dim=np.array([3]),
                 vecs=np.array([[1, 0, 0]], np.float32))
        self.assertIsNone(cache.get('first'))

    def test_backbone_extraction_never_detaches_shared_head(self):
        encoder = object.__new__(DinoV2Encoder)
        encoder.head, encoder.dim, encoder.backbone_dim = object(), 2, 3
        encoder.use_tta = False
        def compute(arrays, use_tta):
            self.assertIsNotNone(encoder.head)
            self.assertEqual(encoder.dim, 2)
            return np.tile(np.array([[1, 0, 0]], np.float32), (len(arrays), 1))
        with patch.object(encoder, '_cache', return_value=None), \
                patch.object(encoder, '_compute_from_arrays', side_effect=compute):
            self.assertEqual(encoder.backbone_features([self.image]).shape, (1, 3))

    def test_live_head_replacement_requires_restart(self):
        import hashlib
        encoder = object.__new__(DinoV2Encoder)
        path = self.root / 'head.pt'
        path.write_bytes(b'checkpoint-v1')
        encoder.head_path = str(path)
        encoder._head_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        stat = path.stat()
        encoder._head_stamp = (stat.st_mtime_ns, stat.st_size)
        encoder.validate_artifacts()
        path.write_bytes(b'checkpoint-v2')
        os.utime(path, ns=(stat.st_mtime_ns + 1_000_000_000, stat.st_mtime_ns + 1_000_000_000))
        with self.assertRaisesRegex(RuntimeError, 'Khởi động lại'):
            encoder.validate_artifacts()

    def test_failed_calibration_write_preserves_last_complete_model(self):
        encoder, bank = self.bank()
        old = CalibrationModel(encoder.tag, .1, 1, bank_fingerprint=bank.fingerprint())
        path = self.root / 'calibration.json'
        old.save(path)
        with patch('rpv.vision.calibration.os.replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                replace(old, alpha=.2).save(path)
        self.assertEqual(CalibrationModel.load(path).alpha, .1)
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_invalid_calibration_cannot_silently_change_prediction_gates(self):
        import json
        encoder, bank = self.bank()
        base = CalibrationModel(encoder.tag, .1, 1, bank_fingerprint=bank.fingerprint()).to_json()
        path = self.root / 'calibration.json'
        for changes in ({'alpha': 0}, {'global_conformal_quantile': float('nan')},
                        {'global_calib_scores': [1, .2]},
                        {'risk_control': {'enabled': 'false', 'confidence_threshold': 0}}):
            with self.subTest(changes=changes):
                path.write_text(json.dumps({**base, **changes}), encoding='utf-8')
                with self.assertRaises(ValueError):
                    CalibrationModel.load(path, encoder.tag)

    def test_mutated_prototypes_cannot_use_a_preexisting_recognizer_threshold(self):
        encoder, bank = self.bank()
        calibration = CalibrationModel(encoder.tag, .1, 1, global_conformal_quantile=.8,
                                       bank_fingerprint=bank.fingerprint())
        recognizer = Recognizer(encoder, bank, calibration)
        bank.entries['a'].centroids[0] = np.roll(bank.entries['a'].centroids[0], 1)
        with self.assertRaises(RuntimeError):
            recognizer.recognize_distances({'a': .5})

    def test_distance_shortlist_cannot_silently_become_a_singleton(self):
        encoder, bank = self.bank()
        bank.entries['b'] = PrototypeEntry('b', np.roll(bank.entries['a'].centroids, 1, axis=1), 1)
        with self.assertRaises(ValueError):
            Recognizer(encoder, bank).recognize_distances({'a': .1})

    def test_live_service_reloads_threshold_changes_without_bank_changes(self):
        encoder, bank = self.bank()
        bank_path, calibration_path = self.root / 'bank.json', self.root / 'calibration.json'
        bank.save(bank_path)
        calibration = CalibrationModel(encoder.tag, .1, 1, global_conformal_quantile=.8,
                                       bank_fingerprint=bank.fingerprint())
        calibration.save(calibration_path)
        repo = ProvenanceRepository(self.root / 'repo.db')
        self.addCleanup(repo.close)
        repo.init_schema()
        service = CheckoutService(encoder, bank, repo, calibration, audit=AuditLog(self.root / 'audit.jsonl'),
                                  bank_path=bank_path, calibration_path=calibration_path)
        service.refresh_catalog()
        self.assertEqual(service.recognizer.recognize_distances({'a': .5}).status, 'match')
        stamp = calibration_path.stat().st_mtime_ns
        replace(calibration, global_conformal_quantile=.2).save(calibration_path)
        os.utime(calibration_path, ns=(stamp + 1_000_000_000, stamp + 1_000_000_000))
        service.refresh_catalog()
        self.assertEqual(service.recognizer.recognize_distances({'a': .5}).status, 'unknown')
        calibration_path.unlink()
        service.refresh_catalog()
        self.assertIsNone(service.calibration)
        self.assertEqual(service.recognizer.recognize_distances({'a': .1}).status, 'ambiguous')

    def test_calibrate_does_not_silently_delete_a_runtime_registered_sku(self):
        from types import SimpleNamespace
        from contextlib import redirect_stdout
        import io
        from rpv import cli
        encoder, bank = self.bank()
        path = self.root / 'bank.json'
        previous = PrototypeBank(encoder)
        previous.entries = dict(bank.entries)
        previous.entries['runtime_sku'] = PrototypeEntry('runtime_sku', bank.entries['a'].centroids.copy(), 1)
        previous.save(path)
        old_bytes = path.read_bytes()
        settings = replace(config.settings, paths=replace(config.settings.paths, bank_path=path,
                           calib_path=self.root / 'calibration.json'))
        calibration = CalibrationModel(encoder.tag, .1, 1, bank_fingerprint=bank.fingerprint())
        args = SimpleNamespace(k_shot=None, alpha=None, replace_catalog=False)
        with patch.object(cli, '_settings', return_value=settings), \
                patch.object(cli, '_build_splits'), \
                patch('rpv.vision.encoders.load_encoder', return_value=encoder), \
                patch('rpv.evaluation.pipeline.fit_catalog', return_value=(bank, calibration)), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'replace-catalog'):
                cli.cmd_calibrate(args)
            self.assertEqual(path.read_bytes(), old_bytes)
            args.replace_catalog = True
            self.assertEqual(cli.cmd_calibrate(args), 0)
        self.assertEqual(PrototypeBank(encoder).load(path).sku_ids, ['a'])

    def test_env_repair_preserves_hash_paths_and_recognizes_an_empty_field_comment(self):
        file = self.root / '.env'
        file.write_text('RPV_TEST_PATH=data/images\n# leave blank for fresh training\nRPV_PROJECTION_CKPT=\n', encoding='utf-8')
        with patch.dict(os.environ, {'RPV_TEST_PATH': 'data/images # photos',
                                     'RPV_PROJECTION_CKPT': '# leave blank for fresh training'}):
            config.load_env_file(file)
            self.assertEqual(os.environ['RPV_TEST_PATH'], 'data/images # photos')
            self.assertEqual(os.environ['RPV_PROJECTION_CKPT'], '')
