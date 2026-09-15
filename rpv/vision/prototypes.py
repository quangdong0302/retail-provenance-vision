"""Prototype bank — "danh mục thị giác" của hệ thống.

Khác biệt then chốt so với bản trước của dự án: prototype KHÔNG còn khoá theo
tên thư mục ảnh (``"Apple 10"``) mà khoá theo ``sku_id`` — đối tượng có thật
trong nghiệp vụ, liên kết được tới lô hàng, nhà cung cấp và chứng từ. Nhờ đó
câu hỏi "cái này là gì" (thị giác) và "nó từ đâu" (hồ sơ) mới nối được với nhau.

Hai tính chất kỹ thuật quan trọng:

* **Multi-centroid.** Một SKU có thể có nhiều hình thái (chuối xanh/chín, có
  hoặc không túi nilon). Ép về một tâm duy nhất làm trung bình rơi vào vùng
  không giống bất kỳ ảnh thật nào. Bank hỗ trợ k tâm/SKU (k-means nhẹ).
* **Có phiên bản.** Bank ghi kèm ``encoder_tag``/``dim``; ``load()`` từ chối
  bank sinh bởi encoder khác thay vì trả về khoảng cách vô nghĩa. Với DINOv2,
  ``encoder_tag`` gồm cả variant, kích thước ảnh và projection head đang dùng.
"""
from __future__ import annotations

import json
import hashlib
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from rpv.util import log
from rpv.file_lock import file_lock
from rpv.vision.encoders import BaseEncoder, image_sha256, l2_normalize


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Khoảng cách Euclid ``[len(a), len(b)]`` tính bằng phép nhân ma trận.

    Vì sao không dùng ``norm(a[:, None] - b[None])``: cách đó cấp phát tensor
    trung gian ``[N, M, D]``. Với DINOv2 (D=384..1536) và vài nghìn ảnh đánh giá,
    đó là hàng GB RAM cho một phép tính đáng ra chỉ cần một GEMM. Công thức dưới
    đây dùng ``|a-b|^2 = |a|^2 + |b|^2 - 2a·b``, chính xác kể cả khi vector chưa
    normalize, và ``maximum(0)`` chặn sai số âm nhỏ trước khi lấy căn.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    d2 = (np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :]
          - 2.0 * (a @ b.T))
    return np.sqrt(np.maximum(d2, 0.0)).astype(np.float32)


def kmeans(vectors: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    """k-means++ nhẹ trên embedding đã normalize. Không kéo thêm sklearn vào
    đường chạy chính (bank phải load được ở thiết bị quầy tối giản)."""
    vectors = np.asarray(vectors, dtype=np.float32)
    n = vectors.shape[0]
    k = max(1, min(int(k), n))
    if k == 1:
        return l2_normalize(vectors.mean(axis=0, keepdims=True))
    rng = np.random.default_rng(seed)
    centers = [vectors[rng.integers(n)]]
    for _ in range(k - 1):
        d = pairwise_distances(vectors, np.asarray(centers)).min(axis=1).astype(np.float64) ** 2
        total = d.sum()
        centers.append(vectors[rng.integers(n)] if total <= 0 else vectors[rng.choice(n, p=d / total)])
    C = np.asarray(centers, dtype=np.float32)
    for _ in range(iters):
        assign = np.argmin(pairwise_distances(vectors, C), axis=1)
        newC = l2_normalize(np.stack([
            vectors[assign == j].mean(axis=0) if np.any(assign == j) else C[j]
            for j in range(k)
        ]))
        if np.allclose(newC, C, atol=1e-6):
            C = newC
            break
        C = newC
    return C.astype(np.float32)


@dataclass
class PrototypeEntry:
    """Một SKU trong bank."""
    sku_id: str
    centroids: np.ndarray                      # [n_centroids, dim]
    n_shots: int
    label: str = ""
    product_id: str = ""
    image_hashes: list[str] = field(default_factory=list)
    registered_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    # Khoảng cách nội-lớp trung bình quan sát được lúc đăng ký. Đây là số liệu
    # CHẨN ĐOÁN, không tham gia quyết định: SKU có intra_spread cao bất thường
    # nghĩa là ảnh mẫu không nhất quán (lẫn nền, lẫn mặt hàng khác) và cần chụp
    # lại. Việc chuẩn hoá score theo độ rộng từng lớp do calibration làm, bằng
    # ngưỡng riêng theo lớp đo trên tập calib — không phải bằng con số này.
    intra_spread: float = 0.0
    requires_operator: bool = False
    enrollment_id: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["centroids"] = self.centroids.astype(np.float32).tolist()
        return d

    @staticmethod
    def from_json(d: dict) -> "PrototypeEntry":
        """Bỏ qua khoá không nhận biết được thay vì ném ``TypeError``.

        Với format được hỗ trợ, bỏ qua field chẩn đoán bổ sung và giữ các
        trường bắt buộc. PrototypeBank.load vẫn từ chối format mới hơn để
        tránh bỏ sót các cờ làm thay đổi hành vi vận hành.
        """
        if not isinstance(d, dict) or not {'sku_id', 'centroids', 'n_shots'} <= d.keys():
            raise BankIncompatibleError('Prototype thiếu trường bắt buộc hoặc không phải object JSON.')
        known = {f.name for f in dataclass_fields(PrototypeEntry)}
        unknown = sorted(set(d) - known)
        if unknown:
            log(f"[bank] bỏ qua trường không nhận biết trong prototype: {unknown}")
        data = {k: v for k, v in d.items() if k in known}
        try:
            data["centroids"] = np.asarray(data["centroids"], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise BankIncompatibleError('Prototype chứa vector không hợp lệ.') from exc
        if (data['centroids'].ndim != 2 or not data['centroids'].size
                or not np.isfinite(data['centroids']).all()
                or not isinstance(data['sku_id'], str) or not data['sku_id']
                or type(data['n_shots']) is not int or data['n_shots'] < 1
                or type(data.get('requires_operator', False)) is not bool):
            raise BankIncompatibleError('Prototype rỗng, không hữu hạn hoặc có metadata sai kiểu.')
        return PrototypeEntry(**data)


class BankIncompatibleError(RuntimeError):
    """Bank được tạo bởi encoder khác encoder đang chạy."""


class PrototypeBank:
    """Bộ nhớ danh mục thị giác. Thêm SKU = trung bình vài vector, O(vài ảnh)."""

    # v3 persists an operator-only gate; old readers must not silently drop it.
    FORMAT_VERSION = 3

    def __init__(self, encoder: BaseEncoder, centroids_per_sku: int | None = None) -> None:
        from rpv.config import settings
        self.encoder = encoder
        self.centroids_per_sku = centroids_per_sku or settings.vision.centroids_per_sku
        self.entries: dict[str, PrototypeEntry] = {}
        self._lock = threading.RLock()

    # -- ghi ---------------------------------------------------------------
    def register(self, sku_id: str, image_paths: Sequence[str | Path], label: str = "",
                 product_id: str = "", max_images: int = 40) -> PrototypeEntry:
        """Đăng ký (hoặc ghi đè) một SKU từ 1..max_images ảnh mẫu.

        Không train lại gì cả — đây chính là cơ chế đáp ứng "cửa hàng nhập hàng
        mới, chụp 2-3 ảnh là bán được ngay".
        """
        paths = [str(p) for p in image_paths]
        if not paths:
            raise ValueError(f"SKU {sku_id}: cần ít nhất 1 ảnh mẫu.")
        if len(paths) > max_images:
            raise ValueError(f"SKU {sku_id}: tối đa {max_images} ảnh mỗi lần đăng ký.")
        vecs, ok_paths, failed = [], [], []
        for p in paths:
            try:
                vecs.append(self.encoder.embed_path(p))
                ok_paths.append(p)
            except Exception as exc:  # ảnh lỗi KHÔNG được lặng lẽ thành vector 0
                failed.append((p, repr(exc)))
        if failed:
            log(f"[bank] SKU {sku_id}: bỏ qua {len(failed)} ảnh lỗi:")
            for p, err in failed[:5]:
                log(f"        - {p}: {err}")
        if not vecs:
            raise RuntimeError(f"SKU {sku_id}: tất cả ảnh đều lỗi, không đăng ký được.")
        V = l2_normalize(np.asarray(vecs, dtype=np.float32))
        centroids = kmeans(V, self.centroids_per_sku)
        spread = float(np.mean(np.min(np.linalg.norm(V[:, None, :] - centroids[None], axis=2), axis=1)))
        with self._lock:
            existing = self.entries.get(sku_id)
            entry = PrototypeEntry(
                sku_id=sku_id, centroids=centroids, n_shots=len(ok_paths),
                label=label or (existing.label if existing else ""),
                product_id=product_id or (existing.product_id if existing else ""),
                image_hashes=[image_sha256(p) for p in ok_paths],
                registered_at=existing.registered_at if existing else _now(),
                updated_at=_now(), intra_spread=spread,
                requires_operator=existing.requires_operator if existing else False,
                enrollment_id=existing.enrollment_id if existing else "",
            )
            self.entries[sku_id] = entry
        return entry

    def remove(self, sku_id: str) -> bool:
        return self.entries.pop(sku_id, None) is not None

    def fingerprint(self) -> str:
        """Bind thresholds to exact prototype geometry, independent of timestamps."""
        with self._lock:
            payload = {'encoder': self.encoder.tag, 'centroids': {
                s: self.entries[s].centroids.tolist() for s in self.sku_ids}}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    # -- đọc ---------------------------------------------------------------
    @property
    def sku_ids(self) -> list[str]:
        return sorted(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, sku_id: object) -> bool:
        return sku_id in self.entries

    def _matrix(self) -> tuple[np.ndarray, list[str]]:
        rows, owners = [], []
        for sku in self.sku_ids:
            for c in self.entries[sku].centroids:
                rows.append(c)
                owners.append(sku)
        if not rows:
            return np.zeros((0, self.encoder.dim), dtype=np.float32), []
        return np.asarray(rows, dtype=np.float32), owners

    def distances(self, embedding: np.ndarray) -> dict[str, float]:
        """Khoảng cách Euclid tới tâm gần nhất của từng SKU (embedding đã
        normalize nên tương đương cosine distance)."""
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        D, skus = self.distances_batch(embedding)
        return {s: float(D[0, i]) for i, s in enumerate(skus)}

    def distances_batch(self, embeddings: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """``[N, n_sku]`` khoảng cách tới tâm gần nhất của từng SKU + danh sách
        sku theo cột. Đây là đường chạy nóng của cả phần đánh giá."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        with self._lock:
            M, owners = self._matrix()
            skus = self.sku_ids
        if M.shape[0] == 0 or embeddings.shape[0] == 0:
            return np.zeros((embeddings.shape[0], len(skus) if M.shape[0] else 0),
                            dtype=np.float32), (skus if M.shape[0] else [])
        if embeddings.shape[1] != M.shape[1]:
            raise ValueError(
                f"Embedding có dim={embeddings.shape[1]} nhưng prototype trong bank có "
                f"dim={M.shape[1]}. Đây là dấu hiệu bank thuộc encoder khác — hãy đăng ký "
                "lại danh mục bằng encoder hiện tại."
            )
        D = pairwise_distances(embeddings, M)
        col = {s: i for i, s in enumerate(skus)}
        out = np.full((embeddings.shape[0], len(skus)), np.inf, dtype=np.float32)
        for j, owner in enumerate(owners):
            i = col[owner]
            np.minimum(out[:, i], D[:, j], out=out[:, i])
        return out, skus

    # -- bền hoá -----------------------------------------------------------
    @contextmanager
    def storage_lock(self, path: str | Path):
        """Hold across reload/check, mutation and atomic save, including other processes."""
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, file_lock(path.with_name(path.name + '.lock')):
            yield

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.storage_lock(path):
            payload = {
                "_meta": {
                    "format_version": self.FORMAT_VERSION,
                    "encoder_tag": self.encoder.tag,
                    "dim": int(self.encoder.dim),
                    "centroids_per_sku": self.centroids_per_sku,
                    "saved_at": _now(),
                    "n_skus": len(self.entries),
                },
                "entries": {k: v.to_json() for k, v in self.entries.items()},
            }
            tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        return path

    def load(self, path: str | Path, strict: bool = True) -> "PrototypeBank":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(payload, dict) or not isinstance(payload.get('_meta'), dict)
                or not isinstance(payload.get('entries'), dict)):
            raise BankIncompatibleError('Bank phải chứa _meta và entries dạng object JSON.')
        meta = payload.get("_meta", {})
        version = meta.get("format_version", 0)
        if type(version) is not int or version < 2:
            raise BankIncompatibleError('Bank có format_version không được hỗ trợ.')
        if version > self.FORMAT_VERSION:
            raise BankIncompatibleError(
                f"Bank tại '{path}' dùng format_version={version}, phiên bản code này chỉ đọc "
                f"tới {self.FORMAT_VERSION}. Hãy cập nhật code thay vì đọc nửa vời."
            )
        tag, dim = meta.get("encoder_tag"), meta.get("dim")
        if tag != self.encoder.tag or dim != int(self.encoder.dim):
            msg = (
                f"Bank tại '{path}' được tạo bởi encoder {tag!r} (dim={dim}), "
                f"khác encoder hiện tại {self.encoder.tag!r} (dim={self.encoder.dim}). "
                "Khoảng cách embedding giữa 2 không gian khác nhau là VÔ NGHĨA. "
                "Hãy đăng ký lại danh mục bằng encoder hiện tại."
            )
            if strict:
                raise BankIncompatibleError(msg)
            log(f"[bank] CẢNH BÁO: {msg}")
        entries = {k: PrototypeEntry.from_json(v) for k, v in payload.get("entries", {}).items()}
        if any(k != entry.sku_id for k, entry in entries.items()):
            raise BankIncompatibleError('Khóa danh mục không khớp sku_id của prototype.')
        centroids_per_sku = meta.get("centroids_per_sku", self.centroids_per_sku)
        if type(centroids_per_sku) is not int or centroids_per_sku < 1:
            raise BankIncompatibleError('centroids_per_sku phải là số nguyên dương.')
        bad = {k: e.centroids.shape for k, e in entries.items()
               if e.centroids.ndim != 2 or e.centroids.shape[1] != int(self.encoder.dim)}
        if bad and strict:
            raise BankIncompatibleError(
                f"Bank tại '{path}': {len(bad)} SKU có prototype sai số chiều so với encoder "
                f"(dim={self.encoder.dim}). Ví dụ: {dict(list(bad.items())[:3])}."
            )
        with self._lock:
            self.entries = entries
            self.centroids_per_sku = centroids_per_sku
        return self

    def summary(self) -> dict:
        return {
            "n_skus": len(self.entries),
            "encoder_tag": self.encoder.tag,
            "dim": int(self.encoder.dim),
            "centroids_per_sku": self.centroids_per_sku,
            "total_shots": int(sum(e.n_shots for e in self.entries.values())),
            "mean_intra_spread": (float(np.mean([e.intra_spread for e in self.entries.values()]))
                                  if self.entries else 0.0),
            # SKU có ảnh mẫu kém nhất quán nhất: danh sách việc cần chụp lại,
            # không phải số trang trí.
            "most_inconsistent_skus": [
                {"sku_id": e.sku_id, "intra_spread": round(float(e.intra_spread), 4), "n_shots": e.n_shots}
                for e in sorted(self.entries.values(), key=lambda x: -x.intra_spread)[:5]
            ],
        }
