"""Group-separated episodic head training with fixed catalog/novel validation.

Only known train receives gradients. Extra-development selects the checkpoint;
main unknown and extra-holdout are never loaded here.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
from rpv.data.extra import extra_partition, novel_development
from rpv.data.managed import split_manifest
from rpv.util import log
from rpv.vision.encoders import l2_normalize
from rpv.vision.projection import ProjectionHead, _require_torch
from rpv.vision.prototypes import pairwise_distances
from rpv import config as app_config

def train_catalog_head(encoder, splits, data_dir, save_path, *, epochs=15,
                       episodes_per_epoch=250, lr=.0005, seed=0, out_dim=None,
                       n_way=17, q_query=8, init_head=None, patience=5, centroids_per_sku=None):
    torch, nn, F = _require_torch()
    centers_per_class = app_config.settings.vision.centroids_per_sku if centroids_per_sku is None else centroids_per_sku
    if centers_per_class not in (1,2):
        raise ValueError('Two-shot catalog training supports one mean or two exemplar prototypes.')
    splits.assert_disjoint()
    if epochs<1 or episodes_per_epoch<1 or q_query<1 or n_way<2 or lr<=0:
        raise ValueError('Invalid catalog training parameters.')
    classes = sorted(splits.train)
    if len(classes)<2 or any(len(splits.train[c])<10 for c in classes):
        raise ValueError('Catalog protocol needs >=2 classes and >=10 train images/class.')
    partition = extra_partition(data_dir)
    if (set(classes) | set(splits.novel_classes)) & {r['label'] for r in partition['development']}:
        raise ValueError('Novel development overlaps main classes.')
    novel_support, novel_query, novel_truth = novel_development(data_dir, partition, len(classes))
    paths = list(dict.fromkeys([p for c in classes for p in splits.train[c]] +
               [p for c in classes for p in splits.support[c]] +
               [p for c in classes for p in splits.validation[c]] +
               [p for ps in novel_support.values() for p in ps] + novel_query))
    index = {p:i for i,p in enumerate(paths)}
    raw = encoder.backbone_features(paths)
    dim = raw.shape[1]; out_dim = dim if out_dim is None else out_dim
    trainmap = {c:np.array([index[p] for p in splits.train[c]],int) for c in classes}
    groups = []
    for c in classes:
        g = defaultdict(list)
        for p in splits.train[c]:
            g[splits.groups[p]].append(index[p])
        groups.append([np.array(v,int) for v in g.values()])
    base_support = [np.array([index[p] for p in splits.support[c]],int) for c in classes]
    val = np.array([index[p] for c in classes for p in splits.validation[c]],int)
    val_truth = np.array([j for j,c in enumerate(classes) for p in splits.validation[c]],int)
    ns = [np.array([index[p] for p in novel_support[c]],int) for c in sorted(novel_support)]
    nq = np.array([index[p] for p in novel_query],int)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    device = encoder.device
    model = ProjectionHead(dim,out_dim,.15).to(device)
    if init_head is not None:
        checkpoint = torch.load(init_head, map_location=device, weights_only=True)
        if (checkpoint.get('backbone_tag') != encoder.backbone_tag
                or checkpoint.get('split_fingerprint') != split_manifest(splits)['fingerprint']):
            raise ValueError('Warm-start checkpoint uses a different backbone or split.')
        model.load_state_dict(checkpoint['state_dict'])
    X = torch.from_numpy(raw).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    best, state, history, stale, best_epoch = -1., None, [], 0, 0
    def project():
        model.eval()
        with torch.inference_mode():
            return l2_normalize(np.concatenate([model(X[i:i+2048]).cpu().numpy() for i in range(0,len(X),2048)]))
    def distances(Z, queries, prototypes):
        if centers_per_class == 1:
            centers = l2_normalize(np.stack([p.mean(0) for p in prototypes]))
            return pairwise_distances(Z[queries], centers)
        pairs = pairwise_distances(Z[queries],np.concatenate(prototypes))
        return pairs.reshape(len(queries),len(prototypes),2).min(2)

    def validate(epoch, losses=(), accuracies=()):
        Z = project()
        base = [Z[ids] for ids in base_support]
        kacc = float((distances(Z,val,base).argmin(1)==val_truth).mean())
        nacc = float((distances(Z,nq,base+[Z[ids] for ids in ns]).argmin(1)==novel_truth).mean())
        score = .5*kacc + .5*nacc
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)) if len(losses) else None,
                   train_acc=float(np.mean(accuracies)) if len(accuracies) else None,
                   val_acc=score, known_catalog_val_acc=kacc, unseen_class_dev_acc_2shot=nacc,
                   selection_score=score, lr=float(optimizer.param_groups[0]['lr']))
        history.append(row)
        log(f'[catalog-train] {row}')
        return score

    # Include epoch zero: fine-tuning must beat the initial representation.
    best = validate(0)
    state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for epoch in range(1,epochs+1):
        model.train(); losses=[]; accuracies=[]
        for _ in range(episodes_per_epoch):
            chosen = rng.choice(len(classes),min(n_way,len(classes)),replace=False)
            supports, queries = [], []
            for j in chosen:
                valid = [g for g in range(len(groups[j])) if len(groups[j][g])>=2]
                if len(groups[j])>=2 and valid:
                    if len(groups[j]) >= 3:
                        a, b, q = rng.choice(len(groups[j]), 3, replace=False)
                        supports.append(np.array([rng.choice(groups[j][a]), rng.choice(groups[j][b])]))
                    else:
                        a=rng.choice(valid)
                        q=rng.choice([g for g in range(len(groups[j])) if g!=a])
                        supports.append(rng.choice(groups[j][a],2,replace=False))
                    queries.append(rng.choice(groups[j][q],min(q_query,len(groups[j][q])),replace=False))
                else:
                    picked=rng.choice(trainmap[classes[j]],2+q_query,replace=False)
                    supports.append(picked[:2]); queries.append(picked[2:])
            S=X[np.concatenate(supports)]; Q=X[np.concatenate(queries)]
            S=F.normalize(S+torch.randn_like(S)*.01,dim=-1)
            Q=F.normalize(Q+torch.randn_like(Q)*.01,dim=-1)
            sp=model(S).reshape(len(chosen),2,out_dim); qp=model(Q)
            if centers_per_class == 1:
                prototypes = F.normalize(sp.mean(1), dim=-1)
                d = torch.cdist(qp, prototypes)
            else:
                d = torch.cdist(qp,sp.reshape(-1,out_dim)).reshape(len(qp),len(chosen),2).min(2).values
            targets=torch.cat([torch.full((len(q),),j,dtype=torch.long,device=device) for j,q in enumerate(queries)])
            weights=torch.cat([torch.full((len(q),),1/len(q),device=device) for q in queries])
            loss=(F.cross_entropy(-d*10,targets,reduction='none')*weights).sum()/len(chosen)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite catalog loss; checkpoint not saved.')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            losses.append(float(loss.item()));accuracies.append(float((d.argmin(1)==targets).float().mean().item()))
        scheduler.step()
        score = validate(epoch, losses, accuracies)
        if score>best:
            best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0;best_epoch=epoch
        else:stale+=1
        if stale>=patience:break
    manifest=split_manifest(splits)
    result=dict(history=history,best_val_acc=best,best_epoch=best_epoch,n_train_classes=len(classes),
                n_train_images=sum(len(v) for v in trainmap.values()),n_val_classes=len(classes),
                n_novel_development_classes=len(ns),n_novel_development_queries=len(nq),
                in_dim=dim,out_dim=out_dim,backbone_tag=encoder.backbone_tag,
                dataset_fingerprint=splits.dataset_fingerprint,split_fingerprint=manifest['fingerprint'],
                development_fingerprint=partition['fingerprint'],save_path=str(save_path),
                training_config=dict(protocol='catalog_grouped',seed=seed,n_way=n_way,k_shot=2,q_query=q_query,
                    epochs=epochs,episodes_per_epoch=episodes_per_epoch,lr=lr,weight_decay=.001,noise_std=.01,
                    head_dropout=.15,temperature=10,patience=patience,device=device,
                    init_head=str(init_head) if init_head is not None else None,
                    prototype_geometry='normalized_support_mean' if centers_per_class==1 else 'nearest_of_two_support_prototypes',
                    centroids_per_sku=centers_per_class,
                    support_sampling='Two distinct capture groups when >=3 train groups; separate query group.',
                    checkpoint_selection='Mean of known validation and novel-development 2-shot accuracy; epoch zero included; identical prototype geometry in training and inference.'),
                versions=dict(torch=str(torch.__version__),numpy=np.__version__))
    path=Path(save_path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict=state,in_dim=dim,out_dim=out_dim,backbone_tag=encoder.backbone_tag,
                    dataset_fingerprint=splits.dataset_fingerprint,split_fingerprint=manifest['fingerprint'],
                    development_fingerprint=partition['fingerprint'],training_config=result['training_config']),path)
    return result
