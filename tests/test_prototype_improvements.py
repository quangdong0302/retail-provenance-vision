"""Persistence and decision regressions for the Source prototype conversion."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from rpv.vision.cache import EmbeddingCache
from rpv.vision.encoders import BaseEncoder
from rpv.vision.calibration import CalibrationModel
from rpv.vision.prototypes import PrototypeBank, PrototypeEntry
from rpv.vision.recognizer import Recognizer, STATUS_MATCH, STATUS_AMBIGUOUS, STATUS_UNKNOWN
from rpv.data.extra import risk_extra_partition


class TinyEncoder(BaseEncoder):
    tag = cache_tag = 'tiny_persistence'
    dim = 2
    use_tta = False

    def embed_arrays(self, arrays):
        return np.tile(np.array([[1.,0.]],np.float32),(len(arrays),1))


class TestPersistentFeatures(unittest.TestCase):
    def test_color_fusion_disambiguates_same_visual_vector_using_support_only(self):
        from rpv.vision.color import PrototypeColorEncoder
        from rpv.vision.prototypes import BankIncompatibleError
        with tempfile.TemporaryDirectory() as folder:
            encoder = PrototypeColorEncoder(TinyEncoder(),.5)
            paths = []
            for c in ('red','blue'):
                path = Path(folder)/(c+'.png')
                Image.new('RGB',(16,16),c).save(path)
                paths.append(path)
            with patch.object(encoder.base,'_cache',return_value=None), patch.object(encoder.color,'_cache',return_value=None):
                bank = PrototypeBank(encoder,1)
                bank.register('red',paths[:1])
                bank.register('blue',paths[1:])
                D,labels = bank.distances_batch(encoder.embed_paths(paths))
            self.assertEqual([labels[j] for j in D.argmin(1)],['red','blue'])
            self.assertEqual(encoder.dim,98)
            self.assertTrue(all(entry.n_shots==1 for entry in bank.entries.values()))
            self.assertFalse(hasattr(bank,'linear_classifier'))
            saved = Path(folder)/'bank.json'
            bank.save(saved)
            changed = PrototypeColorEncoder(TinyEncoder(),.75)
            with self.assertRaises(BankIncompatibleError):
                PrototypeBank(changed,1).load(saved)

    def test_two_writers_merge_vectors_instead_of_overwriting_other_notebook(self):
        with tempfile.TemporaryDirectory() as folder:
            first = EmbeddingCache('raw',2,folder)
            second = EmbeddingCache('raw',2,folder)
            first.put('a',np.array([1.,0.]))
            second.put('b',np.array([0.,1.]))
            first.flush()
            second.flush()
            reloaded = EmbeddingCache('raw',2,folder)
            np.testing.assert_array_equal(reloaded.get('a'),[1.,0.])
            np.testing.assert_array_equal(reloaded.get('b'),[0.,1.])

    def test_interruption_persists_completed_batch_and_new_encoder_reuses_it(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder)/f'{j}.png' for j in range(3)]
            for j,p in enumerate(paths):
                Image.new('RGB',(4,4),(j*70,0,0)).save(p)
            cache = EmbeddingCache('tiny',2,Path(folder)/'cache')
            encoder = TinyEncoder()
            with patch.object(encoder,'_cache',return_value=cache), patch.object(encoder,'_compute_from_arrays',
                    side_effect=[np.array([[1.,0.]],np.float32),KeyboardInterrupt()]):
                with self.assertRaises(KeyboardInterrupt):
                    encoder.embed_paths(paths,use_tta=False,batch_size=1)
            fresh = TinyEncoder()
            reloaded = EmbeddingCache('tiny',2,Path(folder)/'cache')
            with patch.object(fresh,'_cache',return_value=reloaded):
                vectors = fresh.embed_paths(paths,use_tta=False,batch_size=2)
            self.assertEqual(reloaded.hits,1)
            self.assertEqual(reloaded.misses,2)
            np.testing.assert_array_equal(vectors,np.tile([1.,0.],(3,1)))


class TestSelectivePrototype(unittest.TestCase):
    def make(self):
        encoder = TinyEncoder()
        bank = PrototypeBank(encoder,1)
        bank.entries = {c:PrototypeEntry(c,np.array([p],np.float32),2) for c,p in
                        [('a',[1.,0.]),('b',[0.,1.])]}
        calibration = CalibrationModel(encoder.tag,.1,2.,global_conformal_quantile=2.,
            decision_rule='prototype_selective',ood={'method':'energy','temperature':.05,'threshold':.5},
            risk_control={'enabled':True,'confidence_method':'nearest_ratio','confidence_threshold':.5})
        return bank,calibration,Recognizer(encoder,bank,calibration)

    def test_nearest_mean_can_be_selected_with_broad_diagnostic_conformal_set(self):
        bank,calibration,recognizer = self.make()
        result = recognizer.recognize_distances({'a':.2,'b':.8})
        self.assertEqual(result.conformal_prediction_set,['a','b'])
        self.assertEqual(result.prediction_set,['a'])
        self.assertEqual(result.status,STATUS_MATCH)
        self.assertEqual(result.decision_rule,'prototype_selective')
        bank.entries['a'].requires_operator = True
        self.assertEqual(recognizer.recognize_distances({'a':.2,'b':.8}).status,STATUS_AMBIGUOUS)

    def test_uncalibrated_gate_ties_low_margin_and_ood_do_not_auto_accept(self):
        _,calibration,recognizer = self.make()
        for row,status in [({'a':.7,'b':.8},STATUS_AMBIGUOUS),
                           ({'a':.2,'b':.2},STATUS_AMBIGUOUS),
                           ({'a':1.4,'b':1.8},STATUS_UNKNOWN)]:
            self.assertEqual(recognizer.recognize_distances(row).status,status)
        calibration.risk_control = {}
        self.assertEqual(recognizer.recognize_distances({'a':.2,'b':.8}).status,STATUS_AMBIGUOUS)

    def test_reserved_risk_roles_do_not_reuse_head_development_classes(self):
        partition = {'reserved':[{'label':f'r{j}'} for j in range(12)],
                     'development':[{'label':f'd{j}'} for j in range(12)],
                     'holdout':[{'label':'h'}], 'fingerprint':'partition'}
        split = risk_extra_partition(partition)
        cal = {r['label'] for r in split['calibration']}
        val = {r['label'] for r in split['selection']}
        self.assertEqual(len(cal),6)
        self.assertEqual(len(val),6)
        self.assertFalse(cal & val)
        self.assertEqual(cal | val,{r['label'] for r in partition['reserved']})

    def test_vectorized_validation_matches_runtime_decisions_and_confidence(self):
        from scripts.improve_prototypes import operating,energy
        _,calibration,recognizer = self.make()
        calibration.conformal_score = 'relative'
        D = np.random.default_rng(42).uniform(.1,1.5,(100,2)).astype(np.float32)
        for rule in ('conformal','prototype_selective'):
            calibration.decision_rule = rule
            for method in ('energy','nearest_ratio'):
                calibration.risk_control.update(confidence_method=method,confidence_threshold=.1)
                prediction,eligible,confidence = operating(D,np.array([2.,2.]),energy(D,.05),.5,rule,method)
                for i,row in enumerate(D):
                    result = recognizer.recognize_distances(dict(zip(('a','b'),map(float,row))))
                    self.assertEqual(result.status==STATUS_MATCH,bool(eligible[i] and confidence[i]>=.1))
                    self.assertAlmostEqual(calibration.confidence(dict(zip(('a','b'),map(float,row)))),confidence[i],places=7)
                    if result.status==STATUS_MATCH:
                        self.assertEqual(result.prediction_set[0],('a','b')[prediction[i]])


class TestCatalogTrainingSelection(unittest.TestCase):
    def test_shared_metric_roundtrip_preserves_legacy_head_and_applies_after_projection(self):
        import torch
        from rpv.vision.projection import ProjectionHead
        head=ProjectionHead(3,3,0).eval()
        x=torch.tensor([[1.,2.,3.],[3.,2.,1.]])
        original=head(x).detach()
        payload=head.state_dict()
        payload.update(metric_matrix=torch.diag(torch.tensor([2.,1.,.5])),metric_bias=torch.tensor([.1,0.,-.1]))
        loaded=ProjectionHead(3,3,0).eval()
        loaded.load_state_dict(payload)
        expected=torch.nn.functional.normalize(original@payload['metric_matrix'].T+payload['metric_bias'],dim=-1)
        torch.testing.assert_close(loaded(x),expected)
        legacy=ProjectionHead(3,3,0).eval()
        legacy.load_state_dict(head.state_dict())
        torch.testing.assert_close(legacy(x),original)
        invalid=dict(payload,metric_matrix=torch.full((3,3),float('nan')))
        with self.assertRaises(ValueError):loaded.load_state_dict(invalid)

    def test_epoch_zero_survives_degrading_finetune_and_final_test_is_not_embedded(self):
        import torch
        from rpv.data.dataset import DatasetSplits
        from rpv.data.managed import split_manifest
        from rpv.vision.projection import ProjectionHead
        from rpv.vision.catalog_training import train_catalog_head
        splits = DatasetSplits(train_classes=['a','b'],novel_classes=['main_unknown'])
        for c in splits.train_classes:
            splits.train[c] = [f'{c}_train_{j}' for j in range(12)]
            splits.support[c] = [f'{c}_support_{j}' for j in range(2)]
            splits.validation[c] = [f'{c}_validation']
            splits.test[c] = [f'{c}_FINAL_TEST']
            for j,p in enumerate(splits.train[c]):
                splits.groups[p] = f'{c}_train_group_{j//4}'
            for role in ('support','validation','test'):
                for p in getattr(splits,role)[c]:
                    splits.groups[p] = p+'_group'
        novel_support = {'dev0':['ds00','ds01'],'dev1':['ds10','ds11']}
        novel_query = ['dq0','dq1']
        partition = dict(development=[{'label':'dev0'},{'label':'dev1'}],fingerprint='extra')
        encoder = TinyEncoder()
        encoder.device = 'cpu'
        encoder.backbone_tag = 'raw_test'
        captured = []
        def features(paths):
            captured.extend(paths)
            raw = np.random.default_rng(7).normal(size=(len(paths),2)).astype(np.float32)
            return raw/np.linalg.norm(raw,axis=1,keepdims=True)
        encoder.backbone_features = features
        perfect_known = np.array([[0.,1.],[1.,0.]])
        wrong_known = perfect_known[:,::-1].copy()
        perfect_novel = np.array([[2.,2.,0.,1.],[2.,2.,1.,0.]])
        wrong_novel = np.array([[0.,1.,2.,2.],[1.,0.,2.,2.]])
        with tempfile.TemporaryDirectory() as folder:
            initial = Path(folder)/'initial.pt'
            original = ProjectionHead(2,2)
            torch.save(dict(state_dict=original.state_dict(),backbone_tag=encoder.backbone_tag,
                            split_fingerprint=split_manifest(splits)['fingerprint']),initial)
            output = Path(folder)/'selected.pt'
            with patch('rpv.vision.catalog_training.extra_partition',return_value=partition), \
                 patch('rpv.vision.catalog_training.novel_development',return_value=(novel_support,novel_query,np.array([2,3]))), \
                 patch('rpv.vision.catalog_training.pairwise_distances',side_effect=[perfect_known,perfect_novel,wrong_known,wrong_novel]):
                result = train_catalog_head(encoder,splits,folder,output,epochs=1,
                    episodes_per_epoch=1,n_way=2,q_query=1,init_head=initial,centroids_per_sku=1)
            self.assertEqual(result['best_epoch'],0)
            self.assertEqual(result['best_val_acc'],1.)
            selected = torch.load(output,map_location='cpu',weights_only=True)
            for key,value in original.state_dict().items():
                self.assertTrue(torch.equal(value,selected['state_dict'][key]))
            self.assertFalse(any('FINAL_TEST' in p for p in captured))
            self.assertTrue(set(novel_query).issubset(captured))
