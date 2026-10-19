"""Validation-only ablation of a foreground color cue in support prototypes."""
from __future__ import annotations
import os
import sys
from pathlib import Path
import json
import shutil

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
from rpv.vision.color import HueSaturationEncoder
from rpv.vision.encoders import load_encoder,l2_normalize
from rpv.vision.cache import cache_key
from rpv.vision.prototypes import pairwise_distances
from scripts.improve_prototypes import save,wait_cached,project,select_profiles


def main():
    torch.set_num_threads(2)
    st=config.settings
    root=st.paths.data_dir.resolve()
    folder=st.paths.head_path.parent/'trials'
    original=folder/'validation_profiles_dino_only.json'
    if not original.exists():
        shutil.copy2(folder/'validation_profiles.json',original)
        shutil.copy2(folder/'selection.json',folder/'selection_dino_only.json')
    saved=json.loads(original.read_text(encoding='utf8'))
    representations=saved['representations']
    results=saved['profiles']
    print('[integrity] Verifying all main and extra image hashes before cache lookup.',flush=True)
    index=index_dataset(root)
    cfg=st.calibration
    splits=build_splits(index,cfg.k_shot,cfg.class_split_seed,cfg.calib_test_seed)
    partition=extra_partition(root)
    risk=risk_extra_partition(partition)
    support,queries,ntruth=novel_development(root,partition,len(splits.support))
    classes=sorted(splits.support)
    roles={k:[p for c in classes for p in m[c]] for k,m in
           [('support',splits.support),('cal',splits.calib),('val',splits.validation)]}
    roles.update(extra_cal=[str((root/r['path']).resolve()) for r in risk['calibration']],
                 extra_val=[str((root/r['path']).resolve()) for r in risk['selection']])
    paths=list(dict.fromkeys([p for ps in roles.values() for p in ps]+[p for ps in support.values() for p in ps]+queries))
    # Hashes are from the just-verified manifest/extra records, not unverified CSV.
    verified=dict(index.records)
    for role in ('reserved','development','holdout'):
        verified.update({str((root/r['path']).resolve()):r for r in partition[role]})
    encoder=load_encoder(checkpoint=False)
    color=HueSaturationEncoder()
    wait_cached(encoder,paths,verified,True)
    wait_cached(color,paths,verified,True)
    raw=np.stack([encoder._cache().get(cache_key(verified[p]['sha256'],False)) for p in paths])
    hs=np.stack([color._cache().get(cache_key(verified[p]['sha256'],False)) for p in paths])
    lookup={p:i for i,p in enumerate(paths)}
    caltruth=np.array([j for j,c in enumerate(classes) for p in splits.calib[c]])
    valtruth=np.array([j for j,c in enumerate(classes) for p in splits.validation[c]])
    names=['source_noaux','mean_scratch_41','mean_residual_0.25','mean_residual_0.5','mean_residual_0.75','mean_residual_0.9']
    for name in names:
        headpath=representations[name]['path']
        visual=project(raw,headpath)
        for weight in (.15,.3,.5,.75,1.):
            label=f'{name}_hs{weight:g}'
            Z=l2_normalize(np.concatenate([visual,weight*hs],axis=1)).astype(np.float32)
            centers=l2_normalize(np.stack([Z[[lookup[p] for p in splits.support[c]]].mean(0) for c in classes]))
            D={k:pairwise_distances(Z[[lookup[p] for p in ps]],centers) for k,ps in roles.items() if k!='support'}
            ns=l2_normalize(np.stack([Z[[lookup[p] for p in support[c]]].mean(0) for c in sorted(support)]))
            nd=pairwise_distances(Z[[lookup[p] for p in queries]],np.r_[centers,ns])
            representation=dict(known_val_accuracy=float((D['val'].argmin(1)==valtruth).mean()),
                novel_development_2shot_accuracy=float((nd.argmin(1)==ntruth).mean()),path=headpath,
                centroids_per_sku=1,color_weight=weight)
            representations[label]=representation
            print('[representation]',label,representation,flush=True)
            base=representations['source_noaux']
            if (representation['known_val_accuracy']<base['known_val_accuracy'] or
                representation['novel_development_2shot_accuracy']<base['novel_development_2shot_accuracy']-.01):
                continue
            profiles=select_profiles(label,D['cal'],caltruth,D['val'],valtruth,D['extra_cal'],D['extra_val'])
            for p in profiles:
                p.update(centroids_per_sku=1,color_weight=weight)
            results.extend(profiles)
            save(folder/'validation_profiles.json',dict(representations=representations,profiles=results))
    base=representations['source_noaux']
    allowed={name for name,r in representations.items() if r['known_val_accuracy']>=base['known_val_accuracy'] and
             r['novel_development_2shot_accuracy']>=base['novel_development_2shot_accuracy']-.01}
    feasible=[p for p in results if p['head'] in allowed and p['validation']['known_risk'] is not None and
              max(p['validation']['known_risk'],p['validation']['balanced_mixture_risk'],p['validation']['natural_mixture_risk'])<.05]
    winner=max(feasible,key=lambda p:(p['validation']['known_coverage'],representations[p['head']]['known_val_accuracy'],
                                     -p['validation']['balanced_mixture_risk']))
    baseline=[p for p in feasible if p['head']=='source_noaux' and p['decision_rule']=='conformal']
    selection=dict(state='selected',winner=winner,best_baseline=max(baseline,key=lambda p:p['validation']['known_coverage']),
        representations=representations,n_profiles=len(results),risk_partition_fingerprint=risk['fingerprint'],test_used=False,
        selection_rule='Max validation coverage with no lower known accuracy than DINO-only Source prototype; known/natural/50:50 mixture risks <5%; test excluded.')
    save(folder/'validation_profiles.json',dict(representations=representations,profiles=results))
    save(folder/'selection.json',selection)
    print(json.dumps(selection,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
