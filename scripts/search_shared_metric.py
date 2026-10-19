"""Learn a shared regularized within-class metric using known-train only.

The matrix has no SKU-specific classifier weights. Novel classes still use
only their support prototypes. Selection uses known/novel development and
reserved validation; final test labels never choose a metric or threshold.
"""
from __future__ import annotations
import os
import sys
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ['RPV_DEVICE']='cpu'
import numpy as np
import torch
from rpv import config
from rpv.data.dataset import index_dataset,build_splits
from rpv.data.extra import extra_partition,risk_extra_partition,novel_development
from rpv.data.managed import split_manifest
from rpv.vision.encoders import load_encoder,l2_normalize
from rpv.vision.color import HueSaturationEncoder
from rpv.vision.cache import cache_key
from rpv.vision.prototypes import pairwise_distances
from scripts.improve_prototypes import save,wait_cached,project,select_profiles


def choose_profile(results,representations):
    base=representations['source_noaux']
    allowed={n for n,r in representations.items() if r['known_val_accuracy']>=base['known_val_accuracy'] and
             r['novel_development_2shot_accuracy']>=base['novel_development_2shot_accuracy']-.01}
    feasible=[p for p in results if p['head'] in allowed and p['validation']['known_risk'] is not None and
              max(p['validation']['known_risk'],p['validation']['balanced_mixture_risk'],p['validation']['natural_mixture_risk'])<.05]
    key=lambda p:(p['validation']['known_coverage'],representations[p['head']]['known_val_accuracy'],
                  -p['validation']['balanced_mixture_risk'])
    baseline=max([p for p in feasible if p['head']=='source_noaux' and p['decision_rule']=='conformal'],key=key)
    joint=[p for p in feasible if representations[p['head']]['known_val_accuracy']>base['known_val_accuracy'] and
           p['validation']['known_coverage']>baseline['validation']['known_coverage']]
    return max(joint or feasible,key=key),baseline,max(feasible,key=key)


def main():
    torch.set_num_threads(2)
    st=config.settings
    root=st.paths.data_dir.resolve()
    folder=st.paths.head_path.parent/'trials'
    previous=folder/'validation_profiles_color.json'
    if not previous.exists():
        shutil.copy2(folder/'validation_profiles.json',previous)
        shutil.copy2(folder/'selection.json',folder/'selection_color.json')
    saved=json.loads(previous.read_text(encoding='utf8'))
    representations=saved['representations']; results=saved['profiles']
    print('[integrity] Full hash verification of main and extra records.',flush=True)
    index=index_dataset(root)
    cfg=st.calibration
    splits=build_splits(index,cfg.k_shot,cfg.class_split_seed,cfg.calib_test_seed)
    partition=extra_partition(root)
    risk=risk_extra_partition(partition)
    ns,nq,ntruth=novel_development(root,partition,len(splits.support))
    classes=sorted(splits.support)
    roles={k:[p for c in classes for p in m[c]] for k,m in
           [('train',splits.train),('support',splits.support),('cal',splits.calib),('val',splits.validation)]}
    roles.update(extra_cal=[str((root/r['path']).resolve()) for r in risk['calibration']],
                 extra_val=[str((root/r['path']).resolve()) for r in risk['selection']])
    paths=list(dict.fromkeys([p for ps in roles.values() for p in ps]+[p for ps in ns.values() for p in ps]+nq))
    verified=dict(index.records)
    for role in ('reserved','development','holdout'):
        verified.update({str((root/r['path']).resolve()):r for r in partition[role]})
    encoder=load_encoder(checkpoint=False); color=HueSaturationEncoder()
    wait_cached(encoder,paths,verified,True); wait_cached(color,paths,verified,True)
    raw=np.stack([encoder._cache().get(cache_key(verified[p]['sha256'],False)) for p in paths])
    hs=np.stack([color._cache().get(cache_key(verified[p]['sha256'],False)) for p in paths])
    lookup={p:i for i,p in enumerate(paths)}
    caltruth=np.array([j for j,c in enumerate(classes) for p in splits.calib[c]])
    valtruth=np.array([j for j,c in enumerate(classes) for p in splits.validation[c]])
    parent=folder/'source_noaux.pt'
    visual=project(raw,parent)
    train=visual[[lookup[p] for p in roles['train']]].astype(float)
    traintruth=np.array([j for j,c in enumerate(classes) for p in splits.train[c]])
    residual=np.concatenate([train[traintruth==j]-train[traintruth==j].mean(0) for j in range(len(classes))])
    covariance=residual.T@residual/len(residual)
    eig,U=np.linalg.eigh(covariance)
    eig=np.maximum(eig,0); scale=float(eig.mean()); mean=train.mean(0)
    payload=torch.load(parent,map_location='cpu',weights_only=True)
    parent_training=json.loads(parent.with_suffix('.training.json').read_text(encoding='utf8'))
    base=representations['source_noaux']
    for ridge in (.1,.3,1.,3.,10.):
        for power in (.5,1.):
            inverse=(eig/max(scale,1e-12)+ridge)**(-power/2)
            matrix=((U*inverse)@U.T).astype(np.float32)
            for centering in (0.,.5):
                bias=(-centering*mean@matrix.T).astype(np.float32)
                metric_name=f'within_r{ridge:g}_p{power:g}_c{centering:g}'
                output=folder/(metric_name+'.pt')
                state={k:v.clone() for k,v in payload['state_dict'].items()}
                state.update(metric_matrix=torch.from_numpy(matrix),metric_bias=torch.from_numpy(bias))
                metric=dict(method='Regularized within-class covariance whitening',ridge=ridge,power=power,
                    centering=centering,fit_split='known-train only',n_images=len(train),n_classes=len(classes))
                checkpoint=dict(payload,state_dict=state,training_config=dict(payload.get('training_config',{}),shared_metric=metric))
                torch.save(checkpoint,output)
                save(output.with_suffix('.training.json'),dict(parent_training,save_path=str(output),training_config=checkpoint['training_config']))
                save(output.with_suffix('.splits.json'),split_manifest(splits))
                transformed=l2_normalize(visual@matrix.T+bias).astype(np.float32)
                for weight in (0.,.15,.3):
                    name=metric_name+f'_hs{weight:g}'
                    Z=transformed if weight==0 else l2_normalize(np.concatenate([transformed,weight*hs],axis=1)).astype(np.float32)
                    centers=l2_normalize(np.stack([Z[[lookup[p] for p in splits.support[c]]].mean(0) for c in classes]))
                    D={k:pairwise_distances(Z[[lookup[p] for p in ps]],centers) for k,ps in roles.items() if k not in {'support','train'}}
                    nc=l2_normalize(np.stack([Z[[lookup[p] for p in ns[c]]].mean(0) for c in sorted(ns)]))
                    nd=pairwise_distances(Z[[lookup[p] for p in nq]],np.r_[centers,nc])
                    representation=dict(known_val_accuracy=float((D['val'].argmin(1)==valtruth).mean()),
                        novel_development_2shot_accuracy=float((nd.argmin(1)==ntruth).mean()),path=str(output),
                        centroids_per_sku=1,color_weight=weight,shared_metric=metric)
                    representations[name]=representation
                    print('[metric]',name,representation['known_val_accuracy'],representation['novel_development_2shot_accuracy'],flush=True)
                    if representation['known_val_accuracy']<base['known_val_accuracy'] or representation['novel_development_2shot_accuracy']<base['novel_development_2shot_accuracy']-.01:
                        continue
                    profiles=select_profiles(name,D['cal'],caltruth,D['val'],valtruth,D['extra_cal'],D['extra_val'])
                    for p in profiles:p.update(centroids_per_sku=1,color_weight=weight)
                    results.extend(profiles)
                save(folder/'validation_profiles.json',dict(representations=representations,profiles=results))
    winner,baseline,best_coverage=choose_profile(results,representations)
    save(folder/'selection.json',dict(state='selected',winner=winner,
        best_baseline=baseline,best_coverage_profile=best_coverage,representations=representations,
        n_profiles=len(results),risk_partition_fingerprint=risk['fingerprint'],test_used=False,
        selection_rule='Prefer simultaneous improvement of known accuracy and coverage over Source; maximize coverage among those profiles. Novel dev no more than 1pp lower; known/natural/50:50 risks <5%; no test used.'))
    print('[metric] Selection complete:',winner,flush=True)


if __name__=='__main__':main()
