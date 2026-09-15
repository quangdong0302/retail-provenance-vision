"""Cache embedding theo NỘI DUNG ảnh.

Vì sao bắt buộc phải có khi chuyển sang DINOv2: ``calibrate``, ``evaluate`` và
``train-head`` đều đọc lại cùng một bộ ảnh, và một lượt forward ViT trên CPU đắt
hơn descriptor thủ công vài trăm lần. Không có cache, mỗi lần chỉnh alpha là
phải chờ lại toàn bộ backbone — đủ lâu để người ta ngừng đánh giá lại cho tử tế,
và đó mới là thiệt hại thật.

Khoá cache = (tag encoder) + (sha256 nội dung ảnh) + (có TTA hay không). Đổi
backbone, đổi kích thước ảnh, đổi TTA hay sửa một pixel trong ảnh đều sinh khoá
khác, nên cache KHÔNG THỂ trả về vector của một cấu hình khác. Đây là lý do khoá
theo hash nội dung chứ không theo đường dẫn file.
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path

import numpy as np

from rpv.util import log, stable_hash_hex


def cache_key(image_sha256: str, use_tta: bool) -> str:
    return f"{image_sha256}:{'tta1' if use_tta else 'tta0'}"


class EmbeddingCache:
    """Lưu ``[N, dim]`` float32 trong một file ``.npz`` cho mỗi tag encoder."""

    def __init__(self, tag: str, dim: int, cache_dir: str | Path, max_entries: int = 200_000) -> None:
        self.tag = tag
        self.dim = int(dim)
        self.max_entries = int(max_entries)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)[:60]
        self.path = Path(cache_dir) / f"{slug}-{stable_hash_hex(tag, dim, length=10)}.npz"
        self._lock = threading.RLock()
        self._data: dict[str, np.ndarray] = {}
        self._dirty = False
        self._loaded = False

    # -- vòng đời ----------------------------------------------------------
    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            with np.load(self.path, allow_pickle=False) as z:
                keys = z["keys"]
                vecs = z["vecs"]
                meta_dim = int(z["dim"][0]) if "dim" in z else self.dim
            if meta_dim != self.dim or vecs.ndim != 2 or vecs.shape[1] != self.dim:
                # Cache của cấu hình khác: bỏ, không cố "chữa".
                self._data = {}
                return
            self._data = {str(k): np.asarray(v, dtype=np.float32)
                          for k, v in zip(keys.tolist(), vecs)}
        except Exception:
            # Cache hỏng không bao giờ được làm chết đường chạy chính: nó chỉ là
            # tối ưu tốc độ, không phải nguồn dữ liệu.
            self._data = {}

    def flush(self) -> None:
        with self._lock:
            if not self._dirty or not self._data:
                return
            items = list(self._data.items())
            if len(items) > self.max_entries:       # giữ phần mới nhất
                items = items[-self.max_entries:]
                self._data = dict(items)
            keys = np.array([k for k, _ in items], dtype=object).astype("U80")
            vecs = np.stack([v for _, v in items]).astype(np.float32)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Ghi qua file tạm rồi os.replace: cache bị ngắt giữa lúc ghi không
            # để lại file .npz hỏng. Truyền file object cho savez vì nếu truyền
            # đường dẫn, numpy TỰ THÊM ".npz" và tên file tạm sẽ khác dự kiến.
            tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
            try:
                with open(tmp, "wb") as fh:
                    np.savez(fh, keys=keys, vecs=vecs, dim=np.array([self.dim], dtype=np.int64))
                os.replace(tmp, self.path)
                self._dirty = False
            except OSError as exc:
                Path(tmp).unlink(missing_ok=True)
                log(f"[cache] không ghi được {self.path}: {exc} (bỏ qua, chỉ mất tốc độ)")

    # -- truy cập ----------------------------------------------------------
    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            self._load()
            v = self._data.get(key)
            return None if v is None else v.copy()

    def put(self, key: str, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"Cache {self.tag}: kỳ vọng vector dim={self.dim}, nhận {vec.shape}.")
        with self._lock:
            self._load()
            self._data[key] = vec
            self._dirty = True

    def __len__(self) -> int:
        with self._lock:
            self._load()
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data = {}
            self._loaded = True
            self._dirty = False
            self.path.unlink(missing_ok=True)

    def info(self) -> dict:
        return {"tag": self.tag, "dim": self.dim, "n_entries": len(self), "path": str(self.path)}
