"""Regression tests for the thesis data and evaluation protocol."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from rpv.data.dataset import build_splits, index_dataset, split_known_unknown
from rpv.data.managed import prepare_dataset, split_manifest
from rpv.evaluation.pipeline import evaluate_system, fit_catalog
from rpv.evaluation.metrics import risk_coverage_curve, coverage_at_risk
from rpv.vision.calibration import CalibrationModel, _conformal_quantile, _digest_scores
from rpv.vision.encoders import BaseEncoder, preprocess_for_vit, IMAGENET_MEAN, IMAGENET_STD
from rpv.vision.prototypes import PrototypeBank
from rpv.vision.recognizer import Recognizer
from rpv.vision.verification import Verifier


class PixelEncoder(BaseEncoder):
    dim = 3
    tag = 'test_pixel_v1'

    def embed_arrays(self, images):
        vectors = np.array([np.asarray(im, float).mean(axis=(0, 1)) + 1 for im in images], np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class TestManagedDataset(unittest.TestCase):
    def setUp(self):
        quiet = patch('rpv.data.managed.log')
        quiet.start()
        self.addCleanup(quiet.stop)
        self.tmp = tempfile.TemporaryDirectory(prefix='rpv_thesis_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        for c in range(10):
            folder = self.source / f'Product {c}'
            folder.mkdir(parents=True)
            for i in range(24):
                Image.new('RGB', (5, 5), (10 + c * 20, i * 7, 50 + c)).save(folder / f'{i}.png')

    def prepared(self):
        result = prepare_dataset(self.source, self.root / 'prepared')
        return result, index_dataset(self.root / 'prepared')

    def test_ratio_by_class_and_unknown_never_used_for_learning(self):
        result, index = self.prepared()
        self.assertEqual(len(result['known_classes']), 7)
        self.assertEqual(len(result['unknown_classes']), 3)
        known, unknown = split_known_unknown(index)
        splits = build_splits(index, 3, 1, 123)
        self.assertEqual(set(splits.train), set(known))
        self.assertEqual(set(splits.validation), set(known))
        self.assertEqual(set(splits.support), set(known))
        self.assertEqual(set(splits.calib), set(known))
        self.assertEqual(set(splits.novel_support), set(unknown))
        self.assertEqual(splits.unknown_calib, [])
        self.assertEqual(len(splits.unknown_test), 72)
        splits.assert_disjoint()
        self.assertTrue((self.source / 'Product 0/0.png').is_file())

    def test_split_reproducible_and_test_fixed_across_k(self):
        _, index = self.prepared()
        a = build_splits(index, 1, 1, 123)
        b = build_splits(index, 3, 1, 123)
        self.assertEqual(a.test, b.test)
        self.assertEqual(a.novel_test, b.novel_test)
        self.assertEqual(split_manifest(b), split_manifest(build_splits(index, 3, 1, 123)))

    def test_aliases_merged_and_duplicates_excluded_before_class_split(self):
        alias = self.source / 'product_0'
        alias.mkdir()
        shutil.copy2(self.source / 'Product 0/0.png', alias / 'copy.png')
        result, index = self.prepared()
        self.assertEqual(len(index), 10)
        self.assertEqual(result['n_duplicates_excluded'], 1)
        self.assertEqual(result['n_images'], 240)

    def test_conflicting_duplicate_labels_rejected(self):
        shutil.copy2(self.source / 'Product 0/0.png', self.source / 'Product 1/copy.png')
        with self.assertRaisesRegex(ValueError, 'khác nhãn'):
            self.prepared()

    def test_changed_or_untracked_image_rejected(self):
        _, index = self.prepared()
        path = Path(next(iter(index.values()))[0])
        Image.new('RGB', (5, 5), 'black').save(path)
        with self.assertRaisesRegex(ValueError, 'thay đổi'):
            index_dataset(self.root / 'prepared')

    def test_destination_never_overwritten(self):
        self.prepared()
        with self.assertRaisesRegex(ValueError, 'đã tồn tại'):
            self.prepared()

    def test_group_split_no_capture_overlap(self):
        csv_path = self.root / 'groups.csv'
        lines = ['path,group_id']
        for path in self.source.glob('*/*.png'):
            lines.append(f'{path.relative_to(self.source).as_posix()},{path.parent.name}_{int(path.stem)//2}')
        csv_path.write_text('\n'.join(lines), encoding='utf-8')
        prepare_dataset(self.source, self.root / 'prepared', groups_csv=csv_path)
        splits = build_splits(index_dataset(self.root / 'prepared'), 1, 1, 123)
        splits.assert_disjoint()
        for label in splits.train:
            train_groups = {splits.groups[p] for p in splits.train[label]}
            self.assertFalse(train_groups & {splits.groups[p] for p in splits.calib[label]})

    def test_full_evaluation_separates_enrollment_and_preserves_bank(self):
        _, index = self.prepared()
        splits = build_splits(index, 3, 1, 123)
        encoder = PixelEncoder()
        with patch.object(encoder, '_cache', return_value=None):
            result, bank, calibration = evaluate_system(encoder=encoder, splits=splits, verbose=False)
        self.assertEqual(set(bank.sku_ids), set(splits.train_classes))
        self.assertEqual(calibration.bank_fingerprint, bank.fingerprint())
        self.assertEqual([r['k_shot'] for r in result.novel_evaluation['runs']], [1, 3, 5])
        known_rows = [p for p in result.predictions if p['truth'] != '__unknown__']
        coverage = np.mean([p['truth'] in p['prediction_set'] for p in known_rows])
        self.assertEqual(result.selective['empirical_coverage'], coverage)
        self.assertEqual(result.open_set['n_unknown'], 72)
        self.assertIn('nearest_wrong_claim', result.verification)
        self.assertNotIn('accuracy_novel_classes_only', result.closed_set)
        from rpv.evaluation.report import write_report
        paths = write_report(result.to_json(), out_dir=self.root / 'reports')
        report = paths['markdown'].read_text(encoding='utf-8')
        self.assertNotIn('Accuracy chỉ lớp mới (novel)', report)
        self.assertIn('Nhận diện lớp unknown sau khi đăng ký few-shot', report)

    def test_calibrate_does_not_read_any_test_or_unknown(self):
        _, index = self.prepared()
        splits = build_splits(index, 3, 1, 123)
        forbidden = set(splits.unknown_test) | {p for ps in splits.test.values() for p in ps}
        encoder = PixelEncoder()
        original = encoder.embed_paths
        def guarded(paths, *args, **kwargs):
            self.assertFalse(set(map(str, paths)) & forbidden)
            return original(paths, *args, **kwargs)
        with patch.object(encoder, '_cache', return_value=None), patch.object(encoder, 'embed_paths', side_effect=guarded):
            fit_catalog(encoder, splits)


class TestThesisMetrics(unittest.TestCase):
    def test_pad_preserves_both_ends_of_long_object(self):
        arr = np.zeros((20, 100, 3), dtype=np.uint8)
        arr[:, :10, 0] = 255
        arr[:, -10:, 2] = 255
        tensor = preprocess_for_vit(arr, 100, resize_mode='pad')
        rgb = tensor.transpose(1, 2, 0) * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        self.assertGreater(rgb[50, 0, 0], .9)
        self.assertGreater(rgb[50, -1, 2], .9)
        self.assertEqual(tensor.shape, (3, 100, 100))

    def test_conformal_exact_order_statistic_and_insufficient_samples(self):
        self.assertEqual(_conformal_quantile(range(10), .2), 8)
        self.assertTrue(np.isinf(_conformal_quantile([.1, .2], .01)))
        with self.assertRaises(ValueError):
            _conformal_quantile([1, 2], 0)

    def test_full_precision_scores_and_infinity_roundtrip(self):
        scores = np.linspace(.123456789, .9, 2001)
        self.assertEqual(_digest_scores(scores), sorted(scores.tolist()))
        calibration = CalibrationModel('x', .1, .5, conformal_quantile={'a': float('inf')},
                                       global_conformal_quantile=float('inf'))
        with tempfile.TemporaryDirectory(prefix='rpv_cal_') as tmp:
            path = Path(tmp) / 'cal.json'
            calibration.save(path)
            loaded = CalibrationModel.load(path)
        self.assertTrue(np.isinf(loaded.quantile_for('a')))
        self.assertTrue(np.isinf(loaded.quantile_for('new')))

    def test_risk_curve_includes_small_prefix_but_never_splits_ties(self):
        curve = risk_coverage_curve([True, False, False], [3, 2, 2])
        self.assertEqual([p['n_answered'] for p in curve], [1, 3])
        self.assertEqual(coverage_at_risk(curve, 0), 1/3)

    def test_stale_bank_rejected_and_uncalibrated_never_confident(self):
        encoder = PixelEncoder()
        bank = PrototypeBank(encoder)
        calibration = CalibrationModel(encoder.tag, .1, .5, bank_fingerprint='wrong')
        with self.assertRaises(RuntimeError):
            Recognizer(encoder, bank, calibration)
        with self.assertRaises(RuntimeError):
            Verifier(encoder, bank, calibration)
