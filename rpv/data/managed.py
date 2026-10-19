"""Versioned 70/30 class partitions and independent image roles for a thesis.

known/unknown are class pools, not binary visual labels. Unknown images never
participate in head training, model selection or calibration.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import uuid

from PIL import Image

from rpv.data.dataset import DatasetIndex, DatasetSplits, IMAGE_EXT, index_dataset
from rpv.util import log


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


@contextmanager
def staging_directory(parent):
    # A normal directory inherits the workspace ACL. TemporaryDirectory's
    # private Windows ACL can remain attached after publishing via rename.
    parent = Path(parent).resolve()
    path = parent / ('rpv_prepare_' + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        if path.exists():
            if path.resolve().parent != parent:
                raise RuntimeError('Staging directory escaped its parent.')
            shutil.rmtree(path)


def _publish_dataset(staging, destination):
    """Publish once; tolerate short Windows sharing locks without replacing data."""
    import time
    staging, destination = Path(staging).resolve(), Path(destination).resolve()
    if staging.parent.parent != destination.parent or not staging.parent.name.startswith('rpv_prepare_'):
        raise ValueError('Dataset staging must stay inside the destination parent.')
    for attempt in range(5):
        if destination.exists():
            raise ValueError(f'Đích đã tồn tại: {destination}. Không ghi đè dataset.')
        try:
            staging.rename(destination)
            return
        except PermissionError as exc:
            # Windows scanners can briefly hold new files open. A real ACL
            # failure is still raised after a bounded 0.75-second retry window.
            if getattr(exc, 'winerror', None) not in {5, 32, 33} or attempt == 4:
                raise
            time.sleep(.05 * 2 ** attempt)


def prepare_dataset(source, destination, known_ratio=0.7, seed=1, groups_csv=None, workers=8):
    """Copy into a new directory; source files are never moved or overwritten.

    Optional CSV columns: path (relative to source), group_id (capture object /
    session). Without metadata, identical content is grouped, not near-duplicates.
    Legacy unknown_* classes are kept unknown within the total 30% budget.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not 0 < known_ratio < 1:
        raise ValueError('known_ratio phải nằm trong (0, 1).')
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Nguồn và đích phải là hai thư mục tách biệt, không lồng nhau.')
    if destination.exists():
        raise ValueError(f'Đích đã tồn tại: {destination}. Chọn phiên bản dataset mới để giữ split cũ.')
    original_index = index_dataset(source)
    aliases, index, original_owner = defaultdict(list), defaultdict(list), {}
    forced = set()
    for original, paths in original_index.items():
        normalized = re.sub(r'[\s_]+', '_', original.removeprefix('unknown_').casefold()).strip('_')
        if not normalized or normalized in {'.', '..'}:
            raise ValueError(f'Tên lớp không hợp lệ: {original}')
        aliases[normalized].append(original)
        index[normalized].extend(paths)
        for path in paths:
            original_owner[path] = original
        if original.startswith('unknown_') or original in getattr(original_index, 'unknown_classes', ()):
            forced.add(normalized)
    if len(index) < 3:
        raise ValueError('Cần ít nhất 3 lớp: tối thiểu 2 known và 1 unknown.')
    n_known = max(2, min(len(index) - 1, math.floor(len(index) * known_ratio + 0.5)))
    candidates = sorted(set(index) - forced)
    if len(candidates) < n_known:
        raise ValueError('Số lớp unknown định sẵn vượt ngân sách 30%; hãy đổi tỷ lệ hoặc nguồn.')
    random.Random(seed).shuffle(candidates)
    known = set(candidates[:n_known])
    metadata = {}
    if groups_csv:
        with Path(groups_csv).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                key = Path(row['path']).as_posix()
                if key in metadata or not row['group_id'].strip():
                    raise ValueError(f'Metadata trùng hoặc group_id rỗng: {key}')
                metadata[key] = row['group_id'].strip()
    records, owners, class_counts, duplicates = [], {}, {}, []
    def inspect(path):
        with Image.open(path) as image:
            image.verify()
        return path, sha256(path)
    for label, paths in sorted(index.items()):
        class_counts[label] = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            inspected = list(executor.map(inspect, paths))
        for path, content in inspected:
            path = Path(path)
            relative = path.relative_to(source).as_posix()
            if content in owners:
                previous_label, previous_path = owners[content]
                if previous_label != label:
                    raise ValueError(f'Ảnh trùng nhưng khác nhãn: {relative} và {previous_path}. Cần sửa nhãn trước.')
                duplicates.append({'excluded': relative, 'kept_source': previous_path, 'sha256': content})
                continue
            owners[content] = (label, relative)
            if groups_csv and relative not in metadata:
                raise ValueError(f'Metadata thiếu ảnh: {relative}')
            pool = 'known' if label in known else 'unknown'
            # Names and nested capture directories are preserved within the class.
            original = original_owner[str(path)]
            base = source / original
            if isinstance(original_index, DatasetIndex):
                base = source / ('unknown' if original in original_index.unknown_classes else 'known') / original
            dest_relative = (Path(pool) / label / original / path.relative_to(base)).as_posix()
            records.append({'path': dest_relative, 'source_path': relative, 'label': label,
                            'pool': pool, 'sha256': content,
                            'group_id': metadata.get(relative, content)})
            class_counts[label] += 1
        log(f'[prepare] {label}: {class_counts[label]} ảnh hợp lệ')
    manifest = {'schema_version': 1, 'protocol': 'known_unknown_70_30_v1',
                'source': str(source), 'seed': seed, 'requested_known_ratio': known_ratio,
                'actual_known_ratio': len(known) / len(index), 'split_unit': 'class',
                'known_classes': sorted(known), 'unknown_classes': sorted(set(index) - known),
                'class_counts': class_counts, 'label_aliases': dict(aliases),
                'duplicates_excluded': duplicates,
                'grouping': 'metadata' if groups_csv else 'content_hash',
                'records': records}
    manifest['fingerprint'] = fingerprint(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with staging_directory(destination.parent) as temp:
        staging = Path(temp) / 'dataset'
        for pool in ('known', 'unknown'):
            (staging / pool).mkdir(parents=True)
        def copy_record(record):
            target = staging / record['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / record['source_path'], target)
            if sha256(target) != record['sha256']:
                raise ValueError('Nguồn ảnh thay đổi trong lúc sao chép.')
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for count, _ in enumerate(executor.map(copy_record, records), 1):
                if count % 5000 == 0:
                    log(f'[prepare] copied {count}/{len(records)}')
        (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        _publish_dataset(staging, destination)
    return {k: v for k, v in manifest.items() if k not in {'records', 'duplicates_excluded'}} | {
        'destination': str(destination), 'n_images': len(records), 'n_duplicates_excluded': len(duplicates)}


def read_managed_dataset(root):
    root = Path(root).resolve()
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'Thiếu {manifest_path}. Dùng prepare-data để kiểm tra và khóa dataset.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = manifest.pop('fingerprint')
    if manifest.get('schema_version') != 1 or fingerprint(manifest) != expected:
        raise ValueError('Manifest sai phiên bản hoặc đã thay đổi; tạo một dataset version mới.')
    known = set(manifest['known_classes'])
    unknown = set(manifest['unknown_classes'])
    if (not known or not unknown or known & unknown
            or len(known) != len(manifest['known_classes'])
            or len(unknown) != len(manifest['unknown_classes'])):
        raise ValueError('Danh sách lớp known/unknown phải không rỗng, không trùng và tách biệt.')
    result = DatasetIndex(unknown_classes=manifest['unknown_classes'], fingerprint=expected,
                          grouping=manifest['grouping'])
    def check_record(record):
        path = (root / record['path']).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f'Ảnh thiếu hoặc ngoài dataset: {record["path"]}')
        if sha256(path) != record['sha256']:
            raise ValueError(f'Ảnh đã thay đổi: {path}. Tạo dataset version mới.')
        return path
    with ThreadPoolExecutor(max_workers=8) as executor:
        checked_paths = list(executor.map(check_record, manifest['records']))
    seen, groups = set(), {}
    for record, path in zip(manifest['records'], checked_paths):
        if root not in path.parents or path.suffix.lower() not in IMAGE_EXT:
            raise ValueError(f'Đường dẫn ảnh không hợp lệ: {record["path"]}')
        if record['sha256'] in seen:
            raise ValueError('Manifest có ảnh trùng nội dung.')
        seen.add(record['sha256'])
        label = record['label']
        if label not in known | unknown:
            raise ValueError('Record có nhãn ngoài danh sách lớp của manifest.')
        pool = 'unknown' if label in result.unknown_classes else 'known'
        if record.get('pool') != pool:
            raise ValueError('Pool trong record không khớp danh sách lớp known/unknown.')
        if path.relative_to(root).parts[:2] != (pool, label):
            raise ValueError('Pool hoặc nhãn không khớp đường dẫn trong manifest.')
        # A group shared across classes would need a joint grouped splitter;
        # reject instead of silently allowing leakage across those classes.
        group = record['group_id']
        if group in groups and groups[group] != label:
            raise ValueError('group_id phải định danh vật thể/phiên của một lớp, không dùng chung giữa các lớp.')
        groups[group] = label
        result.setdefault(label, []).append(str(path))
        result.records[str(path)] = record
    actual = {p.resolve() for pool in ('known', 'unknown') for p in (root / pool).rglob('*')
              if p.is_file() and p.suffix.lower() in IMAGE_EXT}
    if actual != {Path(p) for ps in result.values() for p in ps}:
        raise ValueError('Có ảnh ngoài manifest. Tạo dataset version mới để cập nhật dữ liệu.')
    if set(result) != known | unknown:
        raise ValueError('Manifest có lớp không chứa ảnh hoặc thiếu record cho lớp đã khai báo.')
    return result


def build_managed_splits(index, k_shot, class_seed, split_seed):
    if k_shot < 1:
        raise ValueError('k_shot phải >= 1.')
    out = DatasetSplits(protocol='known_unknown_70_30_v1', dataset_fingerprint=index.fingerprint)
    out.grouping = index.grouping
    out.groups = {p: r['group_id'] for p, r in index.records.items()}
    out.relative_paths = {p: r['path'] for p, r in index.records.items()}
    out.train_classes = sorted(set(index) - index.unknown_classes)
    out.novel_classes = sorted(index.unknown_classes)
    for label, paths in sorted(index.items()):
        grouped = defaultdict(list)
        for path in paths:
            grouped[out.groups[path]].append(path)
        groups = [sorted(ps) for _, ps in sorted(grouped.items())]
        # Separate seeds: hold validation/calib/test fixed as k changes.
        rng = random.Random(f'{split_seed}:{label}')
        rng.shuffle(groups)
        if label in index.unknown_classes:
            if len(groups) < k_shot + 1:
                raise ValueError(f'{label}: cần ít nhất {k_shot + 1} nhóm độc lập cho novel support/query.')
            out.unknown_test.extend(p for g in groups for p in g)
            reserve = min(max(5, k_shot, round(len(groups) * .20)), len(groups) - 1)
            enrollment_pool = groups[:reserve]
            random.Random(f'{class_seed}:{label}').shuffle(enrollment_pool)
            out.novel_support[label] = [g[0] for g in enrollment_pool[:max(5, k_shot)]]
            out.novel_test[label] = [p for g in groups[reserve:] for p in g]
            continue
        n = len(groups)
        nv, nc, nt = max(2, round(n * .15)), max(1, round(n * .15)), max(1, round(n * .20))
        development = groups[nv + nc + nt:]
        if len(development) < k_shot + 2:
            raise ValueError(f'{label}: {n} nhóm không đủ cho train/validation/calib/test và {k_shot} support. Thêm ảnh/nhóm hoặc giảm k.')
        out.validation[label] = [p for g in groups[:nv] for p in g]
        out.calib[label] = [p for g in groups[nv:nv + nc] for p in g]
        out.test[label] = [p for g in groups[nv + nc:nv + nc + nt] for p in g]
        random.Random(f'{class_seed}:{label}').shuffle(development)
        out.support[label] = [g[0] for g in development[:k_shot]]
        out.train[label] = [p for g in development[k_shot:] for p in g]
    out.assert_disjoint()
    return out


def split_manifest(splits):
    payload = {'protocol': splits.protocol, 'dataset_fingerprint': splits.dataset_fingerprint}
    for role in ('train', 'validation', 'support', 'calib', 'test', 'novel_support', 'novel_test'):
        payload[role] = {label: [splits.relative_paths.get(p, p) for p in paths]
                         for label, paths in getattr(splits, role).items()}
    payload['unknown_test'] = [splits.relative_paths.get(p, p) for p in splits.unknown_test]
    payload['fingerprint'] = fingerprint(payload)
    return payload
