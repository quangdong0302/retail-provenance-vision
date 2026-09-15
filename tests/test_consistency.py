"""Configuration and integration checks use only temporary state, never real artifacts."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
import sqlite3
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from rpv import config
from rpv.vision.calibration import Calibrator
from rpv.vision.encoders import ClassicalEncoder
from rpv.provenance.policy import PolicyEngine
from rpv.vision.prototypes import BankIncompatibleError, PrototypeBank


class RepositoryTransactionTests(unittest.TestCase):
    def test_failed_write_does_not_keep_other_connections_locked(self):
        from rpv.provenance.repository import ProvenanceRepository
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'shared.db'
            with ProvenanceRepository(path, timeout=.2) as first, \
                    ProvenanceRepository(path, timeout=.2) as second:
                first.init_schema()
                with self.assertRaises(sqlite3.IntegrityError):
                    first._write('INSERT INTO skus(sku_id,product_id,name) VALUES(?,?,?)',
                                 ('rejected', 'missing_product', 'Rejected'))
                self.assertFalse(first.conn.in_transaction)
                second._write('INSERT INTO products(product_id,name) VALUES(?,?)',
                              ('valid', 'Another connection'))
                self.assertEqual(first._one('SELECT COUNT(*) FROM products')[0], 1)
                self.assertEqual(first._one('SELECT COUNT(*) FROM skus')[0], 0)


class ConfigurationConsistencyTests(unittest.TestCase):
    def test_malformed_bank_is_rejected_without_replacing_catalog(self):
        import copy
        import numpy as np
        from rpv.vision.prototypes import PrototypeEntry
        bank = PrototypeBank(ClassicalEncoder())
        bank.entries['good'] = PrototypeEntry('good', np.ones((1, bank.encoder.dim)), 1)
        original = bank.entries
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bank.json'
            bank.save(path)
            valid = json.loads(path.read_text(encoding='utf-8'))
            cases = [[], {'entries': {}}, {**valid, 'entries': {'bad': {}}}]
            for change in ({'centroids': [[float('nan')] * bank.encoder.dim]},
                           {'centroids': []}, {'sku_id': 'different'},
                           {'requires_operator': 'false'}, {'n_shots': '1'}):
                payload = copy.deepcopy(valid)
                payload['entries']['good'].update(change)
                cases.append(payload)
            for payload in cases:
                with self.subTest(payload_type=type(payload).__name__):
                    path.write_text(json.dumps(payload), encoding='utf-8')
                    with self.assertRaises(BankIncompatibleError):
                        bank.load(path)
                    self.assertIs(bank.entries, original)

    def test_preimported_consumers_see_reloaded_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            try:
                with patch.dict(os.environ, {'RPV_CACHE_DIR': directory, 'RPV_EMBED_CACHE': '1',
                                             'RPV_ALPHA': '0.23', 'RPV_HIGH_VALUE': '12345'}):
                    config.reload_settings()
                    self.assertEqual(ClassicalEncoder()._cache().path.parent, Path(directory))
                    self.assertAlmostEqual(Calibrator('test').alpha, .23)
                    self.assertEqual(PolicyEngine().cfg.high_value_threshold, 12345)
            finally:
                config.reload_settings()

    def test_rejected_bank_does_not_replace_in_memory_catalog(self):
        encoder = ClassicalEncoder()
        bank = PrototypeBank(encoder)
        original = bank.entries
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bank.json'
            bank.save(path)
            data = json.loads(path.read_text(encoding='utf-8'))
            from rpv.vision.prototypes import PrototypeEntry
            import numpy as np
            entry = PrototypeEntry(sku_id='bad', label='bad',
                                   centroids=np.zeros((1, encoder.dim + 1), dtype=np.float32), n_shots=1)
            data['entries'] = {'bad': entry.to_json()}
            path.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(BankIncompatibleError):
                bank.load(path)
            self.assertIs(bank.entries, original)


@unittest.skipUnless(importlib.util.find_spec('fastapi') and importlib.util.find_spec('httpx'),
                     'API dependencies not installed')
class HealthConsistencyTests(unittest.TestCase):
    def test_api_rejects_nonfinite_and_negative_prices_before_processing(self):
        from fastapi.testclient import TestClient
        from api import main
        service, repo = Mock(), Mock()
        client = TestClient(main.app)
        self.addCleanup(client.close)
        with patch.dict(main.app.dependency_overrides,
                        {main.get_service: lambda: service, main.get_repo: lambda: repo}):
            for price in ('nan', 'inf', '-inf', '-1'):
                with self.subTest(price=price):
                    response = client.post('/checkout', data={'claimed_price': price},
                                           files={'file': ('test.png', b'unused', 'image/png')})
                    self.assertEqual(response.status_code, 422)
                    response = client.post('/catalog/skus', json={
                        'sku_id': 'test', 'product_id': 'test', 'name': 'Test', 'price': price})
                    self.assertEqual(response.status_code, 422)
        service.process.assert_not_called()
        repo.upsert_sku.assert_not_called()

    def test_unavailable_vision_returns_503(self):
        from fastapi.testclient import TestClient
        from api import main
        client = TestClient(main.app)  # No context manager: do not start the real lifespan.
        self.addCleanup(client.close)
        with patch.dict(main._state, {'startup_error': 'test: missing encoder'}, clear=True):
            response = client.get('/health')
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()['status'], 'degraded')

    def test_ready_service_with_empty_catalog_is_healthy(self):
        from fastapi.testclient import TestClient
        from api import main
        service = SimpleNamespace(encoder=SimpleNamespace(tag='test', describe=lambda: {}),
                                  bank=[], calibration=None, refresh_catalog=Mock())
        client = TestClient(main.app)
        self.addCleanup(client.close)
        with patch.dict(main._state, {'service': service}, clear=True):
            response = client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['catalog_size'], 0)
        service.refresh_catalog.assert_called_once_with()

    def test_invalid_catalog_refresh_is_reported_unhealthy(self):
        from fastapi.testclient import TestClient
        from api import main
        service = SimpleNamespace(encoder=SimpleNamespace(tag='test', describe=lambda: {}),
                                  bank=[], calibration=None,
                                  refresh_catalog=Mock(side_effect=ValueError('test: bad bank')))
        client = TestClient(main.app)
        self.addCleanup(client.close)
        with patch.dict(main._state, {'service': service}, clear=True):
            response = client.get('/health')
        self.assertEqual(response.status_code, 503)
        self.assertIn('bad bank', response.json()['error'])


@unittest.skipUnless(importlib.util.find_spec('gradio'), 'Gradio not installed')
class DemoConsistencyTests(unittest.TestCase):
    def test_checkout_displays_every_accepted_candidate(self):
        from app import demo
        candidates = [{'label': f'Accepted {i}', 'distance': .1, 'in_prediction_set': True}
                      for i in range(10)]
        response = SimpleNamespace(
            outcome='assist', origin_summary='Test', provenance_status='missing',
            messages=[], reasons=[], recognition={'candidates': candidates}, verification={},
            audit={}, to_json=lambda: {},
        )
        service = Mock()
        service.process.return_value = response
        with patch.object(demo, '_service', return_value=service):
            rendered, _ = demo.checkout('unused.png', None, None, None, False)
        for candidate in candidates:
            self.assertIn(candidate['label'], rendered)


class RunnerConsistencyTests(unittest.TestCase):
    def test_relative_paths_use_project_root_from_another_working_directory(self):
        from scripts import run_all
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'data/prepared'
            dataset.mkdir(parents=True)
            (dataset / 'manifest.json').write_text('{}', encoding='utf-8')
            calls = []
            with patch.object(run_all, 'ROOT', root), \
                    patch.dict(os.environ, {'RPV_DATA_DIR': 'data/prepared', 'RPV_HEAD_OUT': 'var/new.pt'}), \
                    patch('sys.argv', ['run_all.py', '--encoder', 'dinov2', '--train-head']), \
                    patch.object(run_all, 'run', side_effect=lambda *args, env: calls.append((args, dict(env)))), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(run_all.main(), 0)
            self.assertTrue(all(env['RPV_DATA_DIR'] == str(dataset.resolve()) for _, env in calls))
            train = next(args for args, _ in calls if args[2] == 'train-head')
            self.assertEqual(train[-1], str(root / 'var/new.pt'))

    def test_classical_training_head_rejected_before_running_steps(self):
        from scripts import run_all
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'manifest.json').write_text('{}', encoding='utf-8')
            with patch.dict(os.environ, {'RPV_DATA_DIR': directory}), \
                    patch('sys.argv', ['run_all.py', '--encoder', 'classical', '--train-head']), \
                    patch.object(run_all, 'run') as run, \
                    patch.object(run_all.argparse.ArgumentParser, 'error', side_effect=ValueError), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    run_all.main()
                run.assert_not_called()
