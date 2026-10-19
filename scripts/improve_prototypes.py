"""Reproducible head/operating-point search using validation, never final test.

Run cache_embeddings.py first. This command refuses cache misses, so it cannot
silently recompute backbone vectors or compete with the background GPU worker.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ['RPV_DEVICE'] = 'cpu'

import numpy as np
import torch
from rpv import config
from rpv.data.dataset import build_splits, index_dataset
from rpv.data.extra import extra_partition, novel_development, risk_extra_partition
from rpv.data.managed import split_manifest
from rpv.vision.cache import cache_key
from rpv.vision.catalog_training import train_catalog_head
from rpv.vision.encoders import load_encoder, l2_normalize
from rpv.vision.projection import ProjectionHead
from rpv.vision.prototypes import pairwise_distances
from rpv.vision.calibration import _conformal_quantile
from rpv.vision.rejection import fit_risk_threshold


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    tmp.replace(path)


def energy(D, temperature):
    logits = -(D.astype(float)**2)/temperature
    m = logits.max(1)
    return -temperature*(m + np.log(np.exp(logits-m[:,None]).sum(1)))


def relative(D):
    ordered = np.sort(D, axis=1)
    nearest = D.argmin(1)
    denom = np.broadcast_to(ordered[:,0,None], D.shape).copy()
    denom[np.arange(len(D)), nearest] = ordered[:,1]
    return D / np.maximum(denom, np.float32(1e-9))


def operating(D, q, e, ood_threshold, rule, confidence_method):
    inside = relative(D) <= q
    counts = inside.sum(1)
    nearest = D.argmin(1)
    eligible = (counts == 1) if rule == 'conformal' else (np.diff(np.sort(D,axis=1)[:,:2],axis=1)[:,0] > 1e-8)
    prediction = inside.argmax(1) if rule == 'conformal' else nearest
    eligible &= e <= ood_threshold
    ds = np.sort(D,axis=1)
    confidence = (np.float32(1)-ds[:,0]/np.maximum(ds[:,1],np.float32(1e-9))) if confidence_method == 'nearest_ratio' else -e
    return prediction, eligible, confidence


def metrics(Dknown, truth, Dunknown, q, temperature, ood_threshold, rule, method, gate):
    predicted, eligible, confidence = operating(Dknown,q,energy(Dknown,temperature),ood_threshold,rule,method)
    threshold = gate['confidence_threshold']
    accepted = eligible & (confidence >= threshold) if threshold is not None else np.zeros(len(Dknown),bool)
    _, ue, uc = operating(Dunknown,q,energy(Dunknown,temperature),ood_threshold,rule,method)
    ua = ue & (uc >= threshold) if threshold is not None else np.zeros(len(Dunknown),bool)
    coverage = float(accepted.mean())
    error_rate = float(((predicted != truth) & accepted).mean())
    far = float(ua.mean())
    risk = error_rate/coverage if coverage else None
    balanced_risk = (error_rate+far)/(coverage+far) if coverage+far else None
    natural_risk = (int(((predicted != truth)&accepted).sum())+int(ua.sum())) / max(1,int(accepted.sum())+int(ua.sum()))
    return dict(known_accuracy=float((Dknown.argmin(1)==truth).mean()),known_coverage=coverage,
                known_risk=risk, unknown_false_accept_rate=far,balanced_mixture_risk=balanced_risk,
                natural_mixture_risk=natural_risk,n_known=len(Dknown),n_unknown=len(Dunknown),
                n_known_accepted=int(accepted.sum()),n_unknown_accepted=int(ua.sum()))


def wait_cached(encoder, paths, metadata, wait):
    cache = encoder._cache()
    keys = [cache_key(metadata[str(Path(p).resolve())]['sha256'],False) for p in paths]
    while True:
        cache._loaded = False
        missing = sum(cache.get(k) is None for k in keys)
        print(f'[cache] waiting for {missing}/{len(keys)} vectors; {cache.path}', flush=True)
        if not missing:
            return
        if not wait:
            raise RuntimeError('Run scripts/cache_embeddings.py first, or pass --wait-cache.')
        time.sleep(15)


def project(raw, path):
    payload = torch.load(path,map_location='cpu',weights_only=True)
    head = ProjectionHead(payload['in_dim'],payload['out_dim'],0)
    head.load_state_dict(payload['state_dict'])
    head.eval()
    with torch.inference_mode():
        return l2_normalize(np.concatenate([head(torch.from_numpy(raw[i:i+2048])).numpy() for i in range(0,len(raw),2048)]))


def select_profiles(name, Dcal, caltruth, Dval, valtruth, Dextra_cal, Dextra_val):
    # Alpha changes conformal diagnostics only for prototype_selective.
    profiles = []
    scal = relative(Dcal)
    for rule in ('conformal','prototype_selective'):
        for alpha in ((.05,.1,.2,.3,.4) if rule == 'conformal' else (.1,)):
            q = np.array([_conformal_quantile(scal[caltruth==j,j],alpha) for j in range(Dcal.shape[1])])
            for temperature in (.03,.05,.1):
                ec = energy(Dcal,temperature)
                for recall in (.8,.9,.99):
                    ot = float(np.sort(ec)[int(np.ceil(len(ec)*recall))-1])
                    for method in ('nearest_ratio','energy'):
                        p,eligible,c = operating(Dcal,q,ec,ot,rule,method)
                        _,ue,uc = operating(Dextra_cal,q,energy(Dextra_cal,temperature),ot,rule,method)
                        for target in (.01,.025,.04,.049):
                            gate = fit_risk_threshold(np.r_[p==caltruth,np.zeros(len(uc),bool)],
                                np.r_[c,uc],np.r_[eligible,ue],target)
                            val = metrics(Dval,valtruth,Dextra_val,q,temperature,ot,rule,method,gate)
                            profiles.append(dict(head=name,decision_rule=rule,alpha=alpha,temperature=temperature,
                                target_known_recall=recall,confidence_method=method,target_risk=target,
                                gate=gate,validation=val))
    return profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-cache',action='store_true')
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--episodes',type=int,default=150)
    args = parser.parse_args()
    torch.set_num_threads(2)
    st = config.settings
    root = st.paths.data_dir.resolve()
    folder = st.paths.head_path.parent / 'trials'
    folder.mkdir(parents=True,exist_ok=True)
    baseline = folder/'source_noaux.pt'
    if not baseline.exists():
        shutil.copy2(st.vision.checkpoint,baseline)
        shutil.copy2(Path(st.vision.checkpoint).with_suffix('.training.json'),baseline.with_suffix('.training.json'))
    cfg = st.calibration
    splits = build_splits(index_dataset(root),cfg.k_shot,cfg.class_split_seed,cfg.calib_test_seed)
    partition = extra_partition(root)  # Full content-hash verification is mandatory.
    risk = risk_extra_partition(partition)
    novel_support, novel_query, _ = novel_development(root,partition,len(splits.train))
    training_paths = list(dict.fromkeys([p for role in (splits.train,splits.support,splits.validation)
                        for ps in role.values() for p in ps]+[p for ps in novel_support.values() for p in ps]+novel_query))
    with (root/'metadata.csv').open(encoding='utf-8-sig',newline='') as stream:
        metadata = {str((root/r['path']).resolve()):r for r in csv.DictReader(stream)}
    encoder = load_encoder(checkpoint=False)
    def refuse_recompute(*args,**kwargs):
        raise RuntimeError('Cache miss: research must reuse persisted raw vectors.')
    encoder._compute_from_arrays = refuse_recompute
    wait_cached(encoder,training_paths,metadata,args.wait_cache)
    candidates = [('source_noaux',baseline,json.loads(baseline.with_suffix('.training.json').read_text(encoding='utf8')))]
    experiments = [('mean_warm_17',1e-4,17,baseline),('mean_warm_29',3e-5,29,baseline),('mean_scratch_41',3e-4,41,None)]
    for name,lr,seed,initial in experiments:
        output = folder/f'{name}.pt'
        meta = output.with_suffix('.training.json')
        if meta.exists() and output.exists():
            result = json.loads(meta.read_text(encoding='utf8'))
        else:
            result = train_catalog_head(encoder,splits,root,output,epochs=args.epochs,
                episodes_per_epoch=args.episodes,lr=lr,seed=seed,n_way=17,init_head=initial)
            save(meta,result)
            save(output.with_suffix('.splits.json'),split_manifest(splits))
        candidates.append((name,output,result))
    # Regularize the shared residual toward the frozen backbone. This keeps
    # the architecture/support budget unchanged and is selected only on dev/val.
    baseline_payload = torch.load(baseline,map_location='cpu',weights_only=True)
    baseline_result = json.loads(baseline.with_suffix('.training.json').read_text(encoding='utf8'))
    for strength in (.25,.5,.75,.9):
        name = f'mean_residual_{strength:g}'
        output = folder/f'{name}.pt'
        payload = dict(baseline_payload)
        state = {k:v.clone() for k,v in baseline_payload['state_dict'].items()}
        residual = state['residual.weight']
        state['residual.weight'] = strength*residual + (1-strength)*torch.eye(*residual.shape)
        state['net.3.weight'] *= strength
        state['net.3.bias'] *= strength
        payload.update(state_dict=state,training_config=dict(payload.get('training_config',{}),
            adaptation='Residual shrinkage toward frozen identity',residual_strength=strength,
            prototype_geometry='normalized_support_mean'))
        torch.save(payload,output)
        result = dict(baseline_result,save_path=str(output),training_config=payload['training_config'])
        save(output.with_suffix('.training.json'),result)
        save(output.with_suffix('.splits.json'),split_manifest(splits))
        candidates.append((name,output,result))
    classes = sorted(splits.support)
    roles = {k:[p for c in classes for p in mapping[c]] for k,mapping in
             [('support',splits.support),('cal',splits.calib),('val',splits.validation)]}
    roles['extra_cal'] = [str((root/r['path']).resolve()) for r in risk['calibration']]
    roles['extra_val'] = [str((root/r['path']).resolve()) for r in risk['selection']]
    selection_paths = list(dict.fromkeys([p for ps in roles.values() for p in ps]+[p for ps in novel_support.values() for p in ps]+novel_query))
    wait_cached(encoder,selection_paths,metadata,args.wait_cache)
    raw = encoder.backbone_features(selection_paths)
    index = {p:i for i,p in enumerate(selection_paths)}
    caltruth = np.array([j for j,c in enumerate(classes) for p in splits.calib[c]])
    valtruth = np.array([j for j,c in enumerate(classes) for p in splits.validation[c]])
    results = []
    representation = {}
    for name,path,training in candidates:
        Z = project(raw,path)
        for count in (1,2):
            candidate_name = name if count==1 else name+'_p2'
            supports = [Z[[index[p] for p in splits.support[c]]] for c in classes]
            nsupports = [Z[[index[p] for p in novel_support[c]]] for c in sorted(novel_support)]
            def distances(queries, groups):
                centers = np.concatenate([l2_normalize(v.mean(0,keepdims=True)) if count==1 else v for v in groups])
                pairs = pairwise_distances(queries,centers)
                return pairs if count==1 else pairs.reshape(len(queries),len(groups),2).min(2)
            D = {k:distances(Z[[index[p] for p in ps]],supports) for k,ps in roles.items() if k!='support'}
            _,_,ntruth = novel_development(root,partition,len(classes))
            nd = distances(Z[[index[p] for p in novel_query]],supports+nsupports)
            nacc = float((nd.argmin(1)==ntruth).mean())
            representation[candidate_name] = dict(known_val_accuracy=float((D['val'].argmin(1)==valtruth).mean()),
                                        novel_development_2shot_accuracy=nacc,path=str(path),centroids_per_sku=count)
            print('[representation]',candidate_name,representation[candidate_name],flush=True)
            profiles = select_profiles(candidate_name,D['cal'],caltruth,D['val'],valtruth,D['extra_cal'],D['extra_val'])
            for p in profiles:
                p['centroids_per_sku'] = count
            results.extend(profiles)
            save(folder/'validation_profiles.json',dict(representations=representation,profiles=results))
    base = representation['source_noaux']
    allowed_heads = {name for name,r in representation.items() if r['known_val_accuracy']>=base['known_val_accuracy']
                     and r['novel_development_2shot_accuracy']>=base['novel_development_2shot_accuracy']-.01}
    feasible = [p for p in results if p['head'] in allowed_heads and
                p['validation']['known_risk'] is not None and p['validation']['known_risk']<.05 and
                p['validation']['balanced_mixture_risk']<.05 and p['validation']['natural_mixture_risk']<.05]
    if not feasible:
        save(folder/'selection.json',dict(state='no_feasible_profile',representations=representation))
        raise RuntimeError('No validation operating point meets <5%; baseline retained.')
    winner = max(feasible,key=lambda p:(p['validation']['known_coverage'],
                    representation[p['head']]['known_val_accuracy'], -p['validation']['balanced_mixture_risk']))
    baseline_profiles = [p for p in feasible if p['head']=='source_noaux' and p['decision_rule']=='conformal']
    best_baseline = max(baseline_profiles,key=lambda p:p['validation']['known_coverage']) if baseline_profiles else None
    selection = dict(state='selected',winner=winner,best_baseline=best_baseline,representations=representation,
                     n_profiles=len(results),risk_partition_fingerprint=risk['fingerprint'],
                     selection_rule='Max known validation coverage; known, balanced 50:50 mixture and natural mixture risks <5%; no test used.',
                     test_used=False)
    save(folder/'selection.json',selection)
    print(json.dumps(selection,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
