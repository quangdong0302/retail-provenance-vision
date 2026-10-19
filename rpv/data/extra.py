"""Frozen class partition for extra data; main unknown never enters development."""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from rpv.data.managed import fingerprint, sha256

def extra_partition(root, *, verify=True):
    root = Path(root).resolve()
    metadata = root / 'metadata.csv'
    if not metadata.is_file():
        raise ValueError('joint_extra/catalog_grouped requires metadata.csv and _extra_unknown.')
    with metadata.open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['pool'] == '_extra_unknown']
    labels = sorted({r['label'] for r in rows})
    if len(labels) < 25:
        raise ValueError('Extra protocol requires >=25 disjoint classes: 12 reserved + 12 development + holdout.')
    np.random.default_rng(20261012).shuffle(labels)
    def check_record(r):
        p = (root / r['path']).resolve()
        if root not in p.parents or p.relative_to(root).parts[:2] != ('_extra_unknown', r['label']):
            raise ValueError('Extra image path escapes its declared class.')
        if not p.is_file() or (verify and sha256(p) != r['sha256']):
            raise ValueError(f'Extra image missing/changed: {p}')
        return p
    # Keep full verification; overlap filesystem reads instead of skipping hashes.
    with ThreadPoolExecutor(max_workers=8) as executor:
        checked_paths = list(executor.map(check_record, rows))
    groups, hashes, paths = {}, set(), set()
    for r,p in zip(rows,checked_paths):
        if str(p) in paths or r['sha256'] in hashes:
            raise ValueError('Duplicate extra path/content; prepare a new dataset version.')
        paths.add(str(p)); hashes.add(r['sha256'])
        group = r['group_id']
        if not group or (group in groups and groups[group] != r['label']):
            raise ValueError('Extra capture group must belong to exactly one class.')
        groups[group] = r['label']
    result = {}
    for role, subset in [('reserved', labels[:12]), ('development', labels[12:24]), ('holdout', labels[24:])]:
        chosen = [r for r in rows if r['label'] in subset]
        result[role] = chosen
    descriptor = {'seed': 20261012, 'classes': {k: sorted({r['label'] for r in v}) for k,v in result.items()},
                  'records': [{k:r[k] for k in ('path','label','group_id','sha256')} for r in rows]}
    result['fingerprint'] = fingerprint(descriptor)
    return result

def novel_development(root, partition, known_count):
    groups = defaultdict(lambda: defaultdict(list))
    for row in partition['development']:
        groups[row['label']][row['group_id']].append(str((Path(root) / row['path']).resolve()))
    labels = sorted(c for c,g in groups.items() if len(g)>=3)
    if len(labels)<2:
        raise ValueError('Need >=2 novel development classes with >=3 capture groups.')
    support, query, truth = {}, [], []
    rng = np.random.default_rng(314159)
    for j,c in enumerate(labels):
        gs = sorted(groups[c]); rng.shuffle(gs)
        support[c] = [str(rng.choice(groups[c][g])) for g in gs[:2]]
        ids = [p for g in gs[2:] for p in groups[c][g]]
        query.extend(ids); truth.extend([known_count+j]*len(ids))
    return support, query, np.asarray(truth, dtype=int)


def risk_extra_partition(partition):
    """Split reserved classes; head-development classes never calibrate risk."""
    labels = sorted({r['label'] for r in partition['reserved']})
    np.random.default_rng(20261013).shuffle(labels)
    result = {'calibration': [r for r in partition['reserved'] if r['label'] in labels[:6]],
              'selection': [r for r in partition['reserved'] if r['label'] in labels[6:]],
              'reserved': partition['reserved'], 'holdout': partition['holdout']}
    result['fingerprint'] = fingerprint({'parent': partition['fingerprint'], 'seed': 20261013,
        'pool': 'reserved', 'calibration_classes': labels[:6], 'selection_classes': labels[6:]})
    return result
