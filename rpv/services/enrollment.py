"""Persistent, assisted product enrollment. Validation is development data, not a test benchmark."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

import numpy as np
from PIL import Image, ImageOps

from rpv.vision.encoders import l2_normalize
from rpv.vision.prototypes import PrototypeBank, PrototypeEntry, kmeans, pairwise_distances


@dataclass(frozen=True)
class EnrollmentCriteria:
    min_support: int = 3
    min_validation: int = 10
    min_negative: int = 10
    min_groups: int = 2
    min_accuracy: float = .90
    max_negative_assignment: float = .05
    min_margin: float = .05
    min_side: int = 96
    min_sharpness: float = 8.0
    max_per_role: int = 40


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def inspect_image(path, criteria):
    """Deterministic image heuristics; does not claim object detection or SKU correctness."""
    path = Path(path)
    if path.stat().st_size > 15 * 1024 * 1024:
        raise ValueError('Ảnh vượt quá 15 MB.')
    with Image.open(path) as source:
        if source.width * source.height > 24_000_000:
            raise ValueError('Ảnh vượt quá 24 triệu pixel; giảm kích thước ảnh trước khi tải.')
        im = ImageOps.exif_transpose(source).convert('RGB')
        im.load()
    gray = np.asarray(im.resize((128, 128)).convert('L'), dtype=float)
    lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    brightness, sharpness = float(gray.mean()), float(lap.var())
    thumb = np.asarray(im.resize((16, 16)), dtype=np.uint8)
    pixel_hash = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
    issues = []
    if min(im.size) < criteria.min_side:
        issues.append('Ảnh quá nhỏ; chụp gần hơn và dùng ảnh gốc.')
    if brightness < 15 or brightness > 245:
        issues.append('Ảnh quá tối hoặc quá sáng; điều chỉnh ánh sáng.')
    if sharpness < criteria.min_sharpness:
        issues.append('Ảnh có ít chi tiết hoặc có thể bị mờ; kiểm tra và chụp lại rõ nét.')
    return im, {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'pixel_hash': pixel_hash,
                'thumbnail': thumb.flatten().tolist(), 'width': im.width, 'height': im.height,
                'brightness': brightness, 'sharpness': sharpness, 'issues': issues}


def near_duplicate(a, b):
    return (a['pixel_hash'] == b['pixel_hash'] or
            float(np.abs(np.asarray(a['thumbnail'], float) - np.asarray(b['thumbnail'], float)).mean()) < 2.0)


class EnrollmentService:
    def __init__(self, encoder, bank, root, criteria=None):
        self.encoder, self.bank = encoder, bank
        self.root = Path(root).resolve()
        self.criteria = criteria or EnrollmentCriteria()

    @contextmanager
    def database(self):
        self.root.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.root / 'enrollment.sqlite', timeout=60)
        try:
            db.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _read(self, db, session_id):
        row = db.execute('SELECT payload FROM sessions WHERE id=?', (session_id,)).fetchone()
        if not row:
            raise ValueError('Không tìm thấy hồ sơ đăng ký.')
        d = json.loads(row[0])
        if d['encoder_tag'] != self.encoder.tag:
            raise ValueError('Encoder đã đổi. Tạo hồ sơ mới để không trộn đặc trưng khác phiên bản.')
        return d

    @staticmethod
    def _save(db, d):
        d['updated_at'] = now()
        db.execute('INSERT OR REPLACE INTO sessions VALUES (?,?)',
                   (d['id'], json.dumps(d, ensure_ascii=False, allow_nan=False)))

    @staticmethod
    def public(d):
        return {**{k: v for k, v in d.items() if k != 'retired_samples'},
                'samples': [{k: v for k, v in s.items() if k not in ('vector', 'thumbnail', 'stored_path')}
                            for s in d['samples']]}

    def create(self, sku_id, label='', operator=''):
        sku_id = sku_id.strip()
        if not sku_id or len(sku_id) > 120:
            raise ValueError('Mã mặt hàng cần 1..120 ký tự.')
        d = {'id': uuid.uuid4().hex, 'sku_id': sku_id, 'label': label.strip() or sku_id,
             'operator': operator, 'encoder_tag': self.encoder.tag, 'created_at': now(),
             'revision': 0, 'samples': [], 'history': [], 'events': [], 'report': None,
             'published': False, 'criteria': asdict(self.criteria)}
        with self.database() as db:
            self._save(db, d)
        return self.public(d)

    def get(self, session_id):
        with self.database() as db:
            d = self._read(db, session_id)
        result = self.public(d)
        result['assessment_current'] = bool(d['report'] and
            d['report']['revision'] == d['revision'] and d['report']['bank_fingerprint'] == self.bank.fingerprint())
        return result

    def list_sessions(self):
        with self.database() as db:
            rows = db.execute('SELECT payload FROM sessions ORDER BY rowid DESC').fetchall()
        return [{k: d[k] for k in ('id', 'sku_id', 'label', 'updated_at', 'revision', 'published', 'encoder_tag')}
                for d in (json.loads(row[0]) for row in rows)]

    def add_images(self, session_id, role, paths, group_id, negative_sku='', filenames=None):
        if role not in ('support', 'validation', 'negative'):
            raise ValueError('Vai trò phải là support, validation hoặc negative.')
        group_id = group_id.strip()
        if not group_id or len(group_id) > 160:
            raise ValueError('Nhập mã vật thể/phiên chụp (tối đa 160 ký tự).')
        if not 1 <= len(paths) <= self.criteria.max_per_role:
            raise ValueError('Mỗi lượt cần 1..40 ảnh.')
        if filenames is not None and len(filenames) != len(paths):
            raise ValueError('Tên ảnh không khớp danh sách file.')
        with self.database() as db:
            d = self._read(db, session_id)
            if d['published']:
                raise ValueError('Hồ sơ đã xuất bản. Tạo hồ sơ mới để thay đổi mẫu.')
            c = EnrollmentCriteria(**d['criteria'])
            reserved = d['samples'] + d.get('retired_samples', [])
            if role == 'negative' and (negative_sku not in self.bank.entries or negative_sku == d['sku_id']):
                raise ValueError('Ảnh hàng khác cần mã SKU đúng, đã có trong danh mục và khác mặt hàng đang đăng ký.')
            if any(s['group_id'] == group_id and s.get('negative_sku', '') != (negative_sku if role == 'negative' else '') for s in reserved):
                raise ValueError('Một nhóm vật thể không được khai thành nhiều SKU.')
            if any(s['group_id'] == group_id and s['role'] != role for s in reserved):
                raise ValueError('Một vật thể/phiên chụp không được dùng ở nhiều vai trò.')
            if sum(s['role'] == role for s in d['samples']) + len(paths) > c.max_per_role:
                raise ValueError('Tối đa 40 ảnh mỗi vai trò; bỏ ảnh support không phù hợp trước khi bổ sung.')
            accepted, rejected = [], []
            bank_hashes = {h for e in self.bank.entries.values() for h in e.image_hashes}
            for i, path in enumerate(paths):
                filename = Path(filenames[i] if filenames else path).name
                try:
                    im, info = inspect_image(path, c)
                    if info['issues']:
                        raise ValueError(' '.join(info['issues']))
                    if any(near_duplicate(info, s) for s in d['samples'] + d.get('retired_samples', [])):
                        raise ValueError('Ảnh trùng hoặc gần giống ảnh đã lưu; dùng ảnh chụp mới, không đổi tên/cắt lại ảnh cũ.')
                    if role != 'support' and info['sha256'] in bank_hashes:
                        raise ValueError('Ảnh đã dùng dựng danh mục; cần ảnh kiểm tra độc lập.')
                    vector = np.asarray(self.encoder.embed_path(path), np.float32).reshape(-1)
                    if len(vector) != self.encoder.dim or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-8:
                        raise ValueError('Encoder không trả về đặc trưng hợp lệ.')
                    sample_id = uuid.uuid4().hex
                    folder = self.root / 'images' / d['id']
                    folder.mkdir(parents=True, exist_ok=True)
                    target = folder / (sample_id + '.png')
                    # Persist decoded pixels, so temporary API/UI uploads can be removed.
                    im.save(target)
                    sample = {**info, 'id': sample_id, 'role': role, 'group_id': group_id,
                              'negative_sku': negative_sku if role == 'negative' else '',
                              'filename': filename, 'stored_path': target.relative_to(self.root).as_posix(),
                              'stored_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                              'vector': l2_normalize(vector).tolist()}
                    d['samples'].append(sample)
                    accepted.append(sample_id)
                except (ValueError, OSError) as exc:
                    rejected.append({'filename': filename, 'reason': str(exc)})
            if accepted:
                d['revision'] += 1
            d['events'].append({'at': now(), 'action': 'add_images', 'role': role, 'group_id': group_id,
                                'accepted': accepted, 'rejected': rejected})
            self._save(db, d)
        return {'accepted': accepted, 'rejected': rejected, 'session': self.public(d)}

    def remove_support(self, session_id, sample_id):
        with self.database() as db:
            d = self._read(db, session_id)
            if d['published']:
                raise ValueError('Hồ sơ đã xuất bản.')
            selected = next((s for s in d['samples'] if s['id'] == sample_id), None)
            if not selected or selected['role'] != 'support':
                raise ValueError('Chỉ được bỏ ảnh support; không chọn lọc xóa ảnh kiểm tra khó.')
            d['samples'].remove(selected)
            d.setdefault('retired_samples', []).append(selected)
            d['revision'] += 1
            d['events'].append({'at': now(), 'action': 'remove_support', 'sample_id': sample_id})
            self._save(db, d)
        return self.public(d)

    def _entry(self, d):
        samples = [s for s in d['samples'] if s['role'] == 'support']
        if not samples:
            raise ValueError('Cần ít nhất một ảnh support đạt kiểm tra chất lượng.')
        for s in d['samples']:
            path = (self.root / s['stored_path']).resolve()
            if self.root not in path.parents or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != s['stored_sha256']:
                raise ValueError('Ảnh hồ sơ đã bị sửa hoặc thiếu; không thể dùng kết quả kiểm tra cũ.')
        vectors = np.asarray([s['vector'] for s in samples], dtype=np.float32)
        centers = kmeans(vectors, self.bank.centroids_per_sku)
        return PrototypeEntry(d['sku_id'], centers, len(samples), label=d['label'],
                              image_hashes=[s['sha256'] for s in samples],
                              intra_spread=float(pairwise_distances(vectors, centers).min(axis=1).mean()),
                              requires_operator=True, enrollment_id=d['id'])

    def assess(self, session_id):
        with self.database() as db:
            d = self._read(db, session_id)
            c = EnrollmentCriteria(**d['criteria'])
            candidate = self._entry(d)
            trial = PrototypeBank(self.encoder, self.bank.centroids_per_sku)
            trial.entries = dict(self.bank.entries)
            trial.entries[d['sku_id']] = candidate
            support = [s for s in d['samples'] if s['role'] == 'support']
            rivals = PrototypeBank(self.encoder, self.bank.centroids_per_sku)
            rivals.entries = {k: v for k, v in self.bank.entries.items() if k != d['sku_id']}
            nearest = []
            if len(rivals):
                distances, labels = rivals.distances_batch(np.asarray([s['vector'] for s in support]))
                means = distances.mean(axis=0)
                nearest = [{'sku_id': labels[i], 'label': rivals.entries[labels[i]].label or labels[i],
                            'mean_distance': float(means[i]),
                            'n_support_nearest': int((distances.argmin(axis=1) == i).sum())}
                           for i in np.argsort(means)[:3]]
            predictions = []
            for s in d['samples']:
                if s['role'] == 'support':
                    continue
                distances = trial.distances(np.asarray(s['vector']))
                predicted = min(distances, key=distances.get)
                other = min((v for k, v in distances.items() if k != d['sku_id']), default=None)
                margin = None if other is None else other - distances[d['sku_id']]
                predictions.append({'sample_id': s['id'], 'role': s['role'], 'group_id': s['group_id'],
                                    'predicted': predicted, 'target_distance': distances[d['sku_id']],
                                    'margin': margin, 'correct': (predicted == d['sku_id']) == (s['role'] == 'validation')})
            pos = [r for r in predictions if r['role'] == 'validation']
            neg = [r for r in predictions if r['role'] == 'negative']
            accuracy = sum(r['correct'] for r in pos) / len(pos) if pos else None
            negative_rate = sum(not r['correct'] for r in neg) / len(neg) if neg else None
            clear_rate = sum(r['margin'] is not None and r['margin'] >= c.min_margin for r in pos) / len(pos) if pos else None
            counts = {role: sum(s['role'] == role for s in d['samples']) for role in ('support', 'validation', 'negative')}
            groups = {role: len({s['group_id'] for s in d['samples'] if s['role'] == role}) for role in counts}
            guidance = []
            for role, minimum, title in [('support', c.min_support, 'ảnh mẫu'), ('validation', c.min_validation, 'ảnh kiểm tra đúng mặt hàng'),
                                          ('negative', c.min_negative, 'ảnh kiểm tra mặt hàng khác')]:
                if counts[role] < minimum:
                    guidance.append(f'Bổ sung {minimum-counts[role]} {title}; chụp ảnh mới và ghi đúng nhóm vật thể/phiên.')
                if groups[role] < c.min_groups:
                    guidance.append(f'{title.capitalize()} cần ít nhất {c.min_groups} nhóm vật thể/phiên riêng.')
            sufficient = all(counts[r] >= m and groups[r] >= c.min_groups for r, m in
                             [('support', c.min_support), ('validation', c.min_validation), ('negative', c.min_negative)])
            passed = sufficient and len(rivals) > 0 and accuracy >= c.min_accuracy and negative_rate <= c.max_negative_assignment and clear_rate >= c.min_accuracy
            if nearest:
                guidance.append('Đối chiếu với: ' + ', '.join(r['label'] for r in nearest) + '. Chụp mặt bên, đầu/cuống và một vật thể khác; ưu tiên khác biệt nhìn thấy được.')
            else:
                guidance.append('Danh mục chưa có mặt hàng đối chứng; chưa thể kiểm tra nhầm lẫn giữa SKU.')
            tested_rivals = {s.get('negative_sku') for s in d['samples'] if s['role'] == 'negative'}
            missing_rivals = [r['sku_id'] for r in nearest if r['sku_id'] not in tested_rivals]
            if missing_rivals:
                passed = False
                guidance.append('Bổ sung ảnh kiểm tra của các SKU dễ nhầm chưa có mẫu: ' + ', '.join(missing_rivals) + '.')
            if sufficient and not passed:
                guidance.append('Còn nhầm lẫn hoặc khoảng cách phân biệt nhỏ. Rà nhãn, ảnh mẫu và giữ chế độ nhân viên xác nhận.')
            eval_ids = sorted(s['id'] for s in d['samples'] if s['role'] != 'support')
            evaluation_fingerprint = hashlib.sha256(json.dumps(eval_ids).encode()).hexdigest()
            previous = d['history'][-1] if d['history'] else None
            comparable = bool(previous and previous['evaluation_fingerprint'] == evaluation_fingerprint and
                              previous['bank_fingerprint'] == self.bank.fingerprint())
            report = {'at': now(), 'revision': d['revision'], 'encoder_tag': self.encoder.tag,
                      'bank_fingerprint': self.bank.fingerprint(), 'evaluation_fingerprint': evaluation_fingerprint,
                      'status': 'validation_passed' if passed else ('needs_review' if sufficient else 'needs_samples'),
                      'operating_mode': 'operator_required', 'criteria': d['criteria'], 'counts': counts, 'groups': groups,
                      'accuracy': accuracy, 'negative_assignment_rate': negative_rate, 'clear_margin_rate': clear_rate,
                      'nearest_products': nearest, 'missing_rival_checks': missing_rivals,
                      'predictions': predictions, 'guidance': guidance,
                      'comparison': {'comparable': comparable,
                                     'accuracy_delta': accuracy - previous['accuracy'] if comparable and accuracy is not None and previous['accuracy'] is not None else None,
                                     'negative_assignment_delta': negative_rate - previous['negative_assignment_rate'] if comparable and negative_rate is not None and previous['negative_assignment_rate'] is not None else None},
                      'limitations': ['Đây là kiểm tra phát triển nội bộ; xem rồi sửa support không tạo kết quả test độc lập.',
                                      'Các ngưỡng chất lượng/margin là heuristic, chưa được xác nhận cho mọi điều kiện chụp.',
                                      'Nhóm chụp do người dùng khai; kiểm tra trùng ảnh không chứng minh vật thể độc lập.',
                                      'Negative assignment là nhầm top-1 vào SKU mới, không phải FAR của hệ thống đã hiệu chỉnh.',
                                      'Đạt kiểm tra không cấp bảo đảm thống kê và không bật bán tự động.']}
            d['report'] = report
            d['history'].append(report)
            self._save(db, d)
        return self.public(d)

    def publish(self, session_id, publisher):
        """Publish through checkout's guarded writer, always requiring an operator."""
        with self.database() as db:
            d = self._read(db, session_id)
            if d['published']:
                raise ValueError('Hồ sơ đã xuất bản; không đăng ký lại cùng phiên.')
            report = d['report']
            if not report or report['revision'] != d['revision'] or report['bank_fingerprint'] != self.bank.fingerprint():
                raise ValueError('Ảnh hoặc danh mục đã đổi; chạy kiểm tra lại trước khi đăng ký.')
            if report['counts']['support'] < EnrollmentCriteria(**d['criteria']).min_support:
                raise ValueError('Cần đủ ảnh mẫu đạt chất lượng trước khi đăng ký.')
            entry = self._entry(d)
            publication = publisher(entry, report)
            d['published'] = True
            d['publication'] = publication
            d['events'].append({'at': now(), 'action': 'published_operator_required', 'revision': d['revision']})
            self._save(db, d)
        return self.public(d)

    def sample_path(self, session_id, sample_id):
        with self.database() as db:
            d = self._read(db, session_id)
            s = next((s for s in d['samples'] if s['id'] == sample_id), None)
            if not s:
                raise ValueError('Không tìm thấy ảnh.')
            path = (self.root / s['stored_path']).resolve()
            if self.root not in path.parents or not path.is_file():
                raise ValueError('Ảnh lưu trữ không hợp lệ.')
            return path

    def reference_images(self, session_id):
        """Resolve reference photos only from owned sessions or the configured manifest."""
        from rpv.config import settings
        d = self.get(session_id)
        if not d['report']:
            return []
        dataset_root = Path(settings.paths.data_dir).resolve()
        manifest = dataset_root / 'manifest.json'
        if not hasattr(self, '_references'):
            self._references = {}
            if manifest.is_file():
                records = json.loads(manifest.read_text(encoding='utf-8'))['records']
                for record in records:
                    self._references[record['sha256']] = record['path']
        photos = []
        for rival in d['report']['nearest_products']:
            entry = self.bank.entries.get(rival['sku_id'])
            if not entry:
                continue
            resolved = [(dataset_root / self._references[h]).resolve()
                        for h in entry.image_hashes if h in self._references]
            paths = [p for p in resolved if dataset_root in p.parents]
            if entry.enrollment_id:
                saved = self.get(entry.enrollment_id)
                paths.extend(self.sample_path(saved['id'], s['id']) for s in saved['samples'] if s['role'] == 'support')
            photos.extend((str(p), rival['label']) for p in paths[:1] if p.is_file())
        return photos
