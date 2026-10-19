"""Invariants for the deployed relative-score selective protocol."""
import tempfile
import unittest
from pathlib import Path
from rpv.vision.calibration import CalibrationModel

class TestSelectedProtocol(unittest.TestCase):
    def model(self):
        return CalibrationModel('test',.1,1.,conformal_score='relative',
            conformal_quantile={'a':.7,'b':.7,'c':.7},
            risk_control={'enabled':True,'confidence_method':'nearest_ratio','confidence_threshold':.5},
            calib_scores={'a':[.2,.4,.6]},global_calib_scores=[.2,.4,.6])

    def test_relative_scores_use_competing_class_not_own_radius(self):
        m=self.model();d={'a':.2,'b':.8,'c':1.}
        s=m.nonconformity(d)
        self.assertAlmostEqual(s['a'],.25)
        self.assertAlmostEqual(s['b'],4.)
        self.assertAlmostEqual(s['c'],5.)
        self.assertEqual(s,m.nonconformity({k:v*2 for k,v in d.items()}))
        self.assertTrue(m.risk_accepts(d))
        self.assertFalse(m.risk_accepts({'a':.7,'b':.8,'c':1.}))

    def test_serialization_retains_score_semantics_and_pvalue(self):
        m=self.model()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'calibration.json';m.save(path)
            loaded=CalibrationModel.load(path,'test')
        d={'a':.2,'b':.8,'c':1.}
        self.assertEqual(loaded.nonconformity(d),m.nonconformity(d))
        self.assertEqual(loaded.p_value('a',loaded.nonconformity(d)['a']),.75)
        self.assertEqual(loaded.risk_accepts(d),m.risk_accepts(d))

    def test_future_score_method_cannot_silently_use_distance(self):
        m=self.model();m.conformal_score='unsupported'
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'calibration.json';m.save(path)
            with self.assertRaisesRegex(ValueError,'Unsupported'):
                CalibrationModel.load(path)

    def test_checkpoint_loader_preserves_development_identity(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch is optional for classical-only tests.')
        from rpv.vision.encoders import DinoV2Encoder
        from rpv.vision.projection import ProjectionHead
        head=ProjectionHead(3,3)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'head.pt'
            torch.save(dict(state_dict=head.state_dict(),in_dim=3,out_dim=3,
                backbone_tag='test_backbone',dataset_fingerprint='dataset',
                split_fingerprint='split',development_fingerprint='locked_extra'),path)
            encoder=DinoV2Encoder.__new__(DinoV2Encoder)
            encoder.device='cpu';encoder.backbone_dim=3;encoder.backbone_tag='test_backbone';encoder.variant='vits14'
            encoder._load_head(path)
            self.assertEqual(encoder.head_metadata['development_fingerprint'],'locked_extra')
