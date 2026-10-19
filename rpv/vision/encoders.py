"""Encoder ảnh -> vector đặc trưng L2-normalized.

**DINOv2 là encoder chính của hệ thống.** ``ClassicalEncoder`` vẫn ở đây nhưng
chỉ với hai vai trò rõ ràng, và luận văn nên nói đúng như vậy:

  * ``DinoV2Encoder`` — DINOv2 frozen (self-supervised, pretrained) + tùy chọn
    ``ProjectionHead`` fine-tune bằng episodic training. Mặc định của hệ thống
    và là encoder chính trong thực nghiệm; classical được báo riêng làm baseline.
  * ``ClassicalEncoder`` — descriptor thủ công (numpy + Pillow). Baseline trung
    thực để đo backbone pretrained đóng góp bao nhiêu, và là encoder chạy được
    trên máy không có PyTorch nên bộ test tự động không cần GPU/mạng.

Quy tắc bất di bất dịch: encoder LUÔN công bố ``tag`` (định danh phiên bản).
Prototype bank và mô hình calibration đều ghi kèm tag này; nếu tag lệch, hệ
thống báo lỗi thay vì so sánh khoảng cách giữa hai không gian embedding khác
nhau — lỗi này im lặng và rất khó phát hiện khi đã lên production.
"""
from __future__ import annotations

import hashlib
import json
import ssl
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageOps

from rpv import config as app_config
from rpv.util import log
from rpv.vision.cache import EmbeddingCache, cache_key

# Trung bình/độ lệch chuẩn ImageNet — DINOv2 được pretrain với đúng bộ số này.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def image_sha256(path: str | Path) -> str:
    """Hash nội dung ảnh. Dùng cho audit trail VÀ làm khoá cache: cùng 1 hash =
    cùng 1 ảnh đầu vào, không phụ thuộc tên file hay đường dẫn."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rgb(path: str | Path) -> np.ndarray:
    """Đọc ảnh -> RGB uint8 [H,W,3], đã xử lý xoay theo EXIF.

    Bỏ bước EXIF nghĩa là ảnh chụp bằng điện thoại dọc sẽ vào model ở tư thế
    nằm ngang — một trong những nguyên nhân "model chạy tốt trên dataset, sai ở
    quầy" kinh điển.
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


class BaseEncoder(ABC):
    """Interface cố định cho mọi encoder."""

    dim: int
    tag: str
    #: tag dùng làm khoá cache. Với DINOv2 đây là tag của BACKBONE (không gồm
    #: projection head) để đổi head không phải tính lại toàn bộ backbone.
    cache_tag: str

    @abstractmethod
    def embed_arrays(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """images: list ảnh RGB uint8 [H,W,3] -> [N, dim] đã L2-normalize."""

    # -- tiện ích chung, không cần override ---------------------------------
    def _load(self, path: str | Path) -> np.ndarray:
        return load_rgb(path)

    def _tta_views(self, arrays: Sequence[np.ndarray]) -> list[list[np.ndarray]]:
        """Các "góc nhìn" dùng cho test-time augmentation: gốc + lật ngang.

        Chỉ lật ngang, có chủ đích: mặt hàng ở quầy không có hướng cố định nên
        lật ngang là phép biến đổi bảo toàn nhãn; còn lật dọc hay quay lớn thì
        không (chữ trên bao bì, cuống trái cây). Biến thể đa tỉ lệ chưa được
        triển khai ở đây; muốn so sánh cần thí nghiệm riêng trên validation.
        """
        flipped = [np.ascontiguousarray(a[:, ::-1]) for a in arrays]
        return [list(arrays), flipped]

    def _embed_arrays_tta(self, arrays: Sequence[np.ndarray], use_tta: bool,
                          fn: Callable[[Sequence[np.ndarray]], np.ndarray]) -> np.ndarray:
        if not use_tta:
            return fn(arrays)
        stack = np.stack([fn(v) for v in self._tta_views(arrays)], axis=0)
        return l2_normalize(stack.mean(axis=0))

    # -- cache -------------------------------------------------------------
    def _cache(self) -> EmbeddingCache | None:
        if not app_config.settings.vision.use_cache:
            return None
        got = getattr(self, "_cache_obj", None)
        if got is None:
            got = EmbeddingCache(getattr(self, "cache_tag", self.tag), self._cache_dim(),
                                 app_config.settings.paths.cache_dir, app_config.settings.vision.cache_max_entries)
            self._cache_obj = got
        return got

    def _cache_dim(self) -> int:
        return int(self.dim)

    def _compute_from_arrays(self, arrays: Sequence[np.ndarray], use_tta: bool) -> np.ndarray:
        """Phần được cache. DINOv2 override để cache đặc trưng BACKBONE."""
        return self._embed_arrays_tta(arrays, use_tta, self.embed_arrays)

    def _postprocess(self, vectors: np.ndarray) -> np.ndarray:
        """Áp dụng sau khi lấy từ cache (DINOv2 dùng để chạy projection head)."""
        return vectors

    def _cached_features(self, paths: Sequence[str | Path], use_tta: bool | None = None,
                         batch_size: int | None = None) -> np.ndarray:
        """Nhúng danh sách ảnh, có cache và chia lô.

        Chia lô KHÔNG phải tối ưu cho vui: bản trước nạp toàn bộ ảnh vào RAM rồi
        đẩy một batch duy nhất qua ViT, nên chỉ cần vài trăm ảnh là hết bộ nhớ
        GPU. Đánh giá trên bộ dữ liệu thật luôn có vài nghìn ảnh.
        """
        paths = list(paths)
        if not paths:
            return np.zeros((0, self._cache_dim()), dtype=np.float32)
        use_tta = getattr(self, 'use_tta', app_config.settings.vision.use_tta) if use_tta is None else bool(use_tta)
        bs = app_config.settings.vision.batch_size if batch_size is None else batch_size
        if type(bs) is not int or bs < 1:
            raise ValueError('Embedding batch_size phải là số nguyên dương.')
        cache = self._cache()

        out = np.zeros((len(paths), self._cache_dim()), dtype=np.float32)
        todo: list[int] = []
        keys: list[str | None] = [None] * len(paths)
        if cache is not None:
            for i, p in enumerate(paths):
                try:
                    keys[i] = cache_key(image_sha256(p), use_tta)
                except OSError as exc:
                    raise OSError(f"Không đọc được ảnh {p}: {exc}") from exc
                hit = cache.get(keys[i])          # type: ignore[arg-type]
                if hit is None:
                    todo.append(i)
                else:
                    out[i] = hit
        else:
            todo = list(range(len(paths)))

        try:
            for start in range(0, len(todo), bs):
                idx = todo[start:start + bs]
                arrays = [self._load(paths[i]) for i in idx]
                vecs = self._compute_from_arrays(arrays, use_tta)
                vecs = np.asarray(vecs, dtype=np.float32)
                if (vecs.shape != (len(idx), self._cache_dim()) or not np.isfinite(vecs).all()
                        or np.any(np.linalg.norm(vecs, axis=1) < 1e-8)):
                    raise ValueError('Encoder trả về đặc trưng rỗng, sai số chiều hoặc không hữu hạn.')
                for j, i in enumerate(idx):
                    out[i] = vecs[j]
                    if cache is not None and keys[i] is not None:
                        cache.put(keys[i], vecs[j])    # type: ignore[arg-type]
                # Persist every 256 images so an interrupted notebook can resume.
                if cache is not None and (start + len(idx)) // 256 > start // 256:
                    cache.flush()
        finally:
            if cache is not None and todo:
                cache.flush()
        return out

    def embed_paths(self, paths: Sequence[str | Path], use_tta: bool | None = None,
                    batch_size: int | None = None) -> np.ndarray:
        if hasattr(self, 'validate_artifacts'):
            self.validate_artifacts()
        paths = list(paths)
        if not paths:
            return np.zeros((0, self.dim), dtype=np.float32)
        result = np.asarray(self._postprocess(self._cached_features(paths, use_tta, batch_size)), dtype=np.float32)
        if (result.shape != (len(paths), self.dim) or not np.isfinite(result).all()
                or np.any(np.linalg.norm(result, axis=1) < 1e-8)):
            raise ValueError('Head/encoder trả về đặc trưng không hợp lệ; dừng nhận diện.')
        return result

    def embed_path(self, path: str | Path, use_tta: bool | None = None) -> np.ndarray:
        return self.embed_paths([path], use_tta=use_tta)[0]

    def describe(self) -> dict:
        return {"tag": self.tag, "dim": int(self.dim), "cache_tag": getattr(self, "cache_tag", self.tag)}


class ClassicalEncoder(BaseEncoder):
    """Descriptor thủ công, bám vật thể thay vì bám nền.

    Vì sao vẫn giữ trong một khóa luận đã dùng DINOv2: nó là baseline trung thực
    để chứng minh backbone pretrained đóng góp bao nhiêu. Rất nhiều báo cáo bỏ
    qua bước này rồi kết luận "deep learning tốt hơn" mà không có số so sánh nào.
    Nó cũng là encoder duy nhất chạy được khi không có PyTorch, nên CI kiểm
    chứng được toàn bộ pipeline mà không cần GPU hay mạng.

    Ba quyết định thiết kế đáng bảo vệ trước hội đồng, mỗi cái sửa một lỗi cụ
    thể của bản histogram toàn ảnh:

    * **Tách nền trước khi mô tả.** Ở quầy thanh toán, nền là mặt bàn: nó chiếm
      phần lớn khung hình và giống nhau giữa mọi mặt hàng. Histogram toàn ảnh
      vì thế mô tả cái bàn, không mô tả quả xoài.
    * **Bất biến ánh sáng bằng chromaticity.** Dùng r/(r+g+b), g/(r+g+b) thay
      cho RGB thô: cường độ sáng bị chia đi nên đèn quầy sáng/tối không làm
      dịch chuyển descriptor.
    * **Bất biến quay của hình dạng.** Chữ ký bán kính theo góc được đưa qua
      biên độ FFT: quả xoài đặt xoay kiểu nào cũng cho cùng mô tả hình dạng.
    """

    def __init__(self, size: int = 128, chroma_bins: int = 8, hue_bins: int = 16,
                 grid: int = 3, radial_sectors: int = 32, n_fft: int = 8) -> None:
        for name, value, lo in (("size", size, 32), ("chroma_bins", chroma_bins, 2),
                                ("hue_bins", hue_bins, 2), ("grid", grid, 1),
                                ("radial_sectors", radial_sectors, 4), ("n_fft", n_fft, 2)):
            if int(value) < lo:
                raise ValueError(f"ClassicalEncoder: {name} phải >= {lo} (nhận {value}).")
        if n_fft > radial_sectors // 2 + 1:
            raise ValueError(
                f"n_fft={n_fft} lớn hơn số hệ số rfft có được từ {radial_sectors} cung "
                f"(tối đa {radial_sectors // 2 + 1}); phần dư sẽ luôn bằng 0."
            )
        self.size = int(size)
        self.chroma_bins = int(chroma_bins)
        self.hue_bins = int(hue_bins)
        self.grid = int(grid)
        self.radial_sectors = int(radial_sectors)
        self.n_fft = int(n_fft)
        self.use_tta = bool(app_config.settings.vision.use_tta)
        self.dim = (
            chroma_bins * chroma_bins      # histogram chromaticity 2D của foreground
            + hue_bins                     # hue có trọng số saturation
            + 8 + 8                        # histogram saturation, value
            + n_fft                        # hình dạng bất biến quay
            + 6                            # moment/hình học: area, extent, aspect, eccentricity, solidity, radius std
            + 9                            # kết cấu: gradient + phương sai cục bộ
            + grid * grid * 3              # chroma trung bình theo lưới (bố cục màu bên trong vật thể)
        )
        self.cache_tag = (f"classical2_s{self.size}_c{self.chroma_bins}_h{self.hue_bins}"
                    f"_g{self.grid}_r{self.radial_sectors}_f{self.n_fft}")
        self.tag = f'{self.cache_tag}_tta{int(self.use_tta)}'

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
        diff = mx - mn
        hue = np.zeros_like(mx)
        m = diff > 1e-6
        rmax = m & (mx == r)
        gmax = m & (mx == g) & ~rmax
        bmax = m & (mx == b) & ~rmax & ~gmax
        safe = np.where(m, diff, 1.0)          # tránh chia 0 thay vì bịt cảnh báo
        hue[rmax] = ((g[rmax] - b[rmax]) / safe[rmax]) % 6
        hue[gmax] = ((b[gmax] - r[gmax]) / safe[gmax]) + 2
        hue[bmax] = ((r[bmax] - g[bmax]) / safe[bmax]) + 4
        sat = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)
        return np.stack([hue / 6.0, sat, mx], axis=-1)

    def _foreground_mask(self, rgb: np.ndarray) -> np.ndarray:
        """Nền = median viền ảnh; foreground = vùng lệch xa nền.

        Ngưỡng lấy theo phân vị của chính bức ảnh (không hằng số cứng) nên tự
        thích ứng với độ tương phản khác nhau giữa các mặt hàng.
        """
        h, w, _ = rgb.shape
        band = max(2, h // 16)
        border = np.concatenate([
            rgb[:band].reshape(-1, 3), rgb[-band:].reshape(-1, 3),
            rgb[:, :band].reshape(-1, 3), rgb[:, -band:].reshape(-1, 3),
        ])
        bg = np.median(border, axis=0)
        dist = np.linalg.norm(rgb - bg[None, None, :], axis=-1)
        thr = max(float(np.quantile(dist, 0.55)), 0.10)
        mask = dist > thr
        if mask.mean() < 0.03:                       # tương phản quá thấp: nới ngưỡng
            mask = dist > max(float(np.quantile(dist, 0.35)), 0.04)
        if mask.mean() < 0.02:                       # vẫn không tách được: lấy vùng trung tâm
            yy, xx = np.mgrid[0:h, 0:w]
            mask = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (min(h, w) * 0.32) ** 2
        return mask

    @staticmethod
    def _norm(block: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(block))
        return block / n if n > 1e-8 else block

    def _hist(self, values: np.ndarray, bins: int, weights: np.ndarray | None = None) -> np.ndarray:
        idx = np.clip((values * bins).astype(np.int32), 0, bins - 1)
        return np.bincount(idx.ravel(), weights=None if weights is None else weights.ravel(),
                           minlength=bins).astype(np.float32)

    def _shape_features(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ys, xs = np.nonzero(mask)
        if ys.size < 8:
            return np.zeros(self.n_fft, np.float32), np.zeros(6, np.float32)
        cy, cx = ys.mean(), xs.mean()
        dy, dx = ys - cy, xs - cx
        r = np.sqrt(dy * dy + dx * dx)
        theta = np.arctan2(dy, dx)
        sector = np.clip(((theta + np.pi) / (2 * np.pi) * self.radial_sectors).astype(np.int32),
                         0, self.radial_sectors - 1)
        prof = np.zeros(self.radial_sectors, np.float32)
        np.maximum.at(prof, sector, r.astype(np.float32))
        scale = max(float(prof.max()), 1e-6)
        prof = prof / scale
        # Biên độ FFT của chữ ký bán kính -> bất biến với phép quay trong mặt phẳng.
        spec = np.abs(np.fft.rfft(prof))[: self.n_fft].astype(np.float32)

        # Moment bậc hai -> độ dẹt (eccentricity)
        mu20, mu02 = float((dx ** 2).mean()), float((dy ** 2).mean())
        mu11 = float((dx * dy).mean())
        common = np.sqrt(max((mu20 - mu02) ** 2 + 4 * mu11 ** 2, 0.0))
        lam1, lam2 = (mu20 + mu02 + common) / 2, max((mu20 + mu02 - common) / 2, 1e-9)
        bbox_h, bbox_w = int(ys.max() - ys.min()) + 1, int(xs.max() - xs.min()) + 1
        geom = np.array([
            float(mask.mean()),                                            # area
            float(mask.sum()) / max(float(bbox_h * bbox_w), 1.0),          # extent
            float(bbox_w) / max(float(bbox_h), 1.0),                       # aspect ratio
            float(np.sqrt(max(1 - lam2 / max(lam1, 1e-9), 0.0))),          # eccentricity
            float(mask.sum()) / max(float(np.pi * (r.max() ** 2)), 1.0),   # solidity xấp xỉ
            float(r.std() / max(r.mean(), 1e-6)),                          # độ gồ ghề biên
        ], dtype=np.float32)
        return spec, geom

    def embed_arrays(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        for i, arr in enumerate(images):
            out[i] = self._descriptor(np.asarray(arr, dtype=np.uint8))
        return l2_normalize(out)

    def _descriptor(self, arr: np.ndarray) -> np.ndarray:
        img = Image.fromarray(arr).resize((self.size, self.size), Image.BILINEAR)
        rgb = np.asarray(img, dtype=np.float32) / 255.0
        mask = self._foreground_mask(rgb)

        total = rgb.sum(axis=-1) + 1e-6
        chroma_r, chroma_g = rgb[..., 0] / total, rgb[..., 1] / total   # bất biến cường độ sáng
        hsv = self._rgb_to_hsv(rgb)

        fg = mask
        if fg.sum() < 16:
            fg = np.ones_like(mask)

        # 1) histogram chromaticity 2D (chỉ foreground)
        cb = self.chroma_bins
        ri = np.clip((chroma_r[fg] * cb).astype(np.int32), 0, cb - 1)
        gi = np.clip((chroma_g[fg] * cb).astype(np.int32), 0, cb - 1)
        h2d = np.bincount(ri * cb + gi, minlength=cb * cb).astype(np.float32)

        # 2) hue (trọng số saturation: hue của pixel bạc màu là nhiễu)
        hue_h = self._hist(hsv[..., 0][fg], self.hue_bins, weights=hsv[..., 1][fg])
        sat_h = self._hist(hsv[..., 1][fg], 8)
        val_h = self._hist(hsv[..., 2][fg], 8)

        # 3) hình dạng
        spec, geom = self._shape_features(mask)

        # 4) kết cấu trên foreground
        gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        mag = np.sqrt(gx * gx + gy * gy)
        k = 5
        pad = np.pad(gray, k // 2, mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(pad, (k, k))
        local_var = win.var(axis=(-1, -2))
        tex = np.concatenate([
            np.array([float(mag[fg].mean()), float(mag[fg].std()),
                      float((mag[fg] > 0.08).mean())], dtype=np.float32),
            self._hist(np.clip(local_var[fg] * 40.0, 0, 0.999), 6),
        ])

        # 5) bố cục màu bên trong vật thể (lưới thô trên bbox foreground)
        grid_feats = np.zeros(self.grid * self.grid * 3, dtype=np.float32)
        ys, xs = np.nonzero(mask)
        if ys.size >= 16:
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            sh, sw = max(1, (y1 - y0) // self.grid), max(1, (x1 - x0) // self.grid)
            for gy_ in range(self.grid):
                for gx_ in range(self.grid):
                    sl = (slice(y0 + gy_ * sh, min(y1, y0 + (gy_ + 1) * sh)),
                          slice(x0 + gx_ * sw, min(x1, x0 + (gx_ + 1) * sw)))
                    cell = mask[sl]
                    base = (gy_ * self.grid + gx_) * 3
                    if cell.sum() > 4:
                        grid_feats[base] = float(chroma_r[sl][cell].mean())
                        grid_feats[base + 1] = float(chroma_g[sl][cell].mean())
                        grid_feats[base + 2] = float(hsv[..., 1][sl][cell].mean())

        parts = [
            self._norm(h2d) * 1.6,      # màu là tín hiệu mạnh nhất với nông sản
            self._norm(hue_h) * 1.3,
            self._norm(sat_h) * 0.7,
            self._norm(val_h) * 0.4,    # độ sáng: trọng số thấp vì phụ thuộc đèn
            self._norm(spec) * 1.2,     # hình dạng
            self._norm(geom) * 1.0,
            self._norm(tex) * 0.9,
            self._norm(grid_feats) * 1.0,
        ]
        return np.concatenate(parts).astype(np.float32)


def preprocess_for_vit(arr: np.ndarray, image_size: int, center_crop: bool = True,
                       crop_ratio: float = 0.875, resize_mode: str | None = None) -> np.ndarray:
    """RGB uint8 [H,W,3] -> float32 [3,S,S] đã chuẩn hoá theo ImageNet.

    Tự viết bằng PIL + numpy thay vì ``torchvision.transforms``: bớt một phụ
    thuộc nặng, và quan trọng hơn là bước tiền xử lý trở nên đọc được — nó ảnh
    hưởng tới số liệu ngang với model nên không nên nằm sau lớp trừu tượng của
    thư viện khác.

    ``center_crop=True`` lặp lại transform đánh giá của DINOv2: resize CẠNH NGẮN
    rồi cắt giữa, nên tỉ lệ khung ảnh được giữ nguyên. Resize thẳng về hình vuông
    (``center_crop=False``) làm méo vật thể — quả dưa leo dài thành quả bí — và đó
    là biến dạng mà backbone chưa từng thấy khi pretrain.

    Là hàm module-level có chủ đích: nhờ vậy bước tiền xử lý kiểm thử được trên
    máy không có PyTorch, đúng phần dễ sai nhất và im lặng nhất.
    """
    if image_size <= 0:
        raise ValueError(f"image_size phải > 0 (nhận {image_size}).")
    img = Image.fromarray(np.asarray(arr, dtype=np.uint8))
    s = int(image_size)
    mode = resize_mode or ('crop' if center_crop else 'stretch')
    if mode not in {'pad', 'crop', 'stretch'}:
        raise ValueError('resize_mode phải là pad, crop hoặc stretch.')
    if mode == 'pad':
        w, h = img.size
        scale = s / max(w, h)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        resized = img.resize(size, Image.BICUBIC)
        # Neutral padding matches the normalization mean, so padded pixels ~0.
        img = Image.new('RGB', (s, s), tuple(round(v * 255) for v in IMAGENET_MEAN))
        img.paste(resized, ((s - size[0]) // 2, (s - size[1]) // 2))
    elif mode == 'crop':
        resize_to = max(s, int(round(s / max(crop_ratio, 1e-6))))
        w, h = img.size
        if w <= h:
            nw, nh = resize_to, max(s, int(round(h * resize_to / w)))
        else:
            nw, nh = max(s, int(round(w * resize_to / h))), resize_to
        img = img.resize((nw, nh), Image.BICUBIC)
        left, top = (nw - s) // 2, (nh - s) // 2
        img = img.crop((left, top, left + s, top + s))
    else:
        img = img.resize((s, s), Image.BICUBIC)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


@contextmanager
def _certifi_ssl_context():
    """Tạm dùng CA bundle của certifi khi tải trọng số qua HTTPS.

    KHÔNG tắt xác thực chứng chỉ, và KHÔNG để lại thay đổi toàn cục sau khi tải
    xong: bản trước ghi đè ``ssl._create_default_https_context`` vĩnh viễn, làm
    mọi lệnh gọi HTTPS khác trong tiến trình đổi hành vi một cách vô hình.
    """
    original = getattr(ssl, "_create_default_https_context", None)
    try:
        import certifi
        ssl._create_default_https_context = lambda *a, **kw: ssl.create_default_context(  # type: ignore[attr-defined]
            cafile=certifi.where())
    except ImportError:
        pass
    try:
        yield
    finally:
        if original is not None:
            ssl._create_default_https_context = original  # type: ignore[attr-defined]


class DinoV2Encoder(BaseEncoder):
    """DINOv2 frozen (+ ProjectionHead tùy chọn). Encoder chính của hệ thống.

    Backbone luôn ở chế độ eval và không có gradient: thêm sản phẩm mới chỉ là
    phép trung bình vài vector, không train lại gì cả — đúng yêu cầu nghiệp vụ
    "2-3 ảnh là dùng được ngay".

    Ba điều được xử lý tường minh vì đều đã từng làm hệ thống sai hoặc chết:

    * **Kích thước ảnh phải chia hết cho patch size (14).** ViT-*/14 chia ảnh
      thành ô 14x14; đưa 256 vào sẽ hoặc lỗi hoặc bị cắt ngầm. Ở đây kiểm tra
      trước và báo lỗi kèm danh sách giá trị hợp lệ.
    * **Chia lô khi suy luận.** Không chia lô thì chỉ vài trăm ảnh là hết VRAM.
    * **Cache theo backbone.** Khoá cache dùng tag backbone, không gồm head, nên
      train lại projection head không làm mất cache đặc trưng đã tính.
    """

    _OUT_DIM = {"vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536}
    PATCH = 14

    def __init__(self, variant: str | None = None, checkpoint: str | Path | None = None,
                 device: str | None = None, image_size: int | None = None,
                 repo: str | Path | None = None, weights: str | Path | None = None) -> None:
        cfg = app_config.settings.vision

        self.variant = (variant or cfg.dinov2_variant).lower()
        if self.variant not in self._OUT_DIM:
            raise ValueError(
                f"variant DINOv2 không hợp lệ: {self.variant!r}. Chọn một trong {list(self._OUT_DIM)}."
            )
        self.image_size = int(image_size or cfg.image_size)
        if self.image_size % self.PATCH != 0:
            valid = [self.PATCH * n for n in range(8, 25)]
            raise ValueError(
                f"RPV_IMAGE_SIZE={self.image_size} không chia hết cho patch size {self.PATCH} của "
                f"ViT-*/{self.PATCH}. Dùng một trong: {valid}."
            )
        self.center_crop = bool(cfg.center_crop)
        self.crop_ratio = float(cfg.crop_ratio)
        self.resize_mode = cfg.resize_mode
        self.use_tta = bool(cfg.use_tta)
        if self.resize_mode not in {'pad', 'crop', 'stretch'}:
            raise ValueError('RPV_RESIZE_MODE phải là pad, crop hoặc stretch.')
        if not 0.5 <= self.crop_ratio <= 1.0:
            raise ValueError(f"RPV_CROP_RATIO={self.crop_ratio} phải trong [0.5, 1.0].")

        _require_torch()
        self.device = str(device or cfg.resolved_device())
        self.backbone_dim = self._OUT_DIM[self.variant]
        self._lock = threading.RLock()

        self.backbone = self._load_backbone(repo, weights)
        weight_hash = hashlib.sha256()
        for name, tensor in sorted(self.backbone.state_dict().items()):
            weight_hash.update(name.encode())
            weight_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        self.weights_sha256 = weight_hash.hexdigest()
        transform = {'version': 2, 'variant': self.variant, 'image_size': self.image_size,
                     'center_crop': self.center_crop, 'crop_ratio': self.crop_ratio,
                     'resize_mode': self.resize_mode,
                     'weights_sha256': self.weights_sha256}
        signature = hashlib.sha256(json.dumps(transform, sort_keys=True).encode()).hexdigest()[:16]
        self.backbone_tag = f'dinov2_{self.variant}_s{self.image_size}_{signature}'
        self.cache_tag = self.backbone_tag
        self.backbone.eval().to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.head = None
        self.dim = self.backbone_dim
        self.head_path = ""
        self.head_metadata = {}
        ckpt = None if checkpoint is False else (checkpoint if checkpoint not in (None, "") else (cfg.checkpoint or None))
        if ckpt:
            self._load_head(Path(ckpt))
        head_tag = "raw" if self.head is None else f"proj{self.dim}_{self._head_sha256[:16]}"
        self.tag = f"{self.backbone_tag}_{head_tag}_tta{int(self.use_tta)}"

    # -- nạp model ---------------------------------------------------------
    def _load_backbone(self, repo: str | Path | None, weights: str | Path | None):
        """Nạp backbone, ưu tiên các đường offline trước khi nghĩ tới mạng.

        Thứ tự: repo local do người dùng chỉ định -> cache torch.hub đã có sẵn
        -> tải từ GitHub. Máy dùng để bảo vệ khóa luận thường không có mạng, nên
        đường offline phải là đường chính, không phải phương án chữa cháy.
        """
        import torch

        name = f"dinov2_{self.variant}"
        cfg = app_config.settings.vision
        repo_dir = Path(repo or cfg.dinov2_repo) if (repo or cfg.dinov2_repo) else None
        weights_file = Path(weights or cfg.dinov2_weights) if (weights or cfg.dinov2_weights) else None

        if weights_file is not None and not weights_file.exists():
            raise FileNotFoundError(f"Không tìm thấy file trọng số DINOv2: {weights_file}")
        if repo_dir is not None:
            if not (repo_dir / 'hubconf.py').is_file():
                raise FileNotFoundError(f'Repo DINOv2 chỉ định không hợp lệ: {repo_dir}')
            try:
                model = torch.hub.load(str(repo_dir), name, source='local', pretrained=weights_file is None)
                if weights_file is not None:
                    self._load_state_into(model, weights_file)
                return model
            except Exception as exc:
                raise RuntimeError(f'Không nạp được repo DINOv2 đã chỉ định {repo_dir}; không đổi sang repo khác.') from exc

        local_dirs = [d for d in (repo_dir, Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main")
                      if d is not None and d.exists()]
        errors: list[str] = []
        for d in local_dirs:
            try:
                model = torch.hub.load(str(d), name, source="local",
                                       pretrained=weights_file is None)
                if weights_file is not None:
                    self._load_state_into(model, weights_file)
                return model
            except Exception as exc:                     # pragma: no cover - phụ thuộc môi trường
                errors.append(f"{d}: {type(exc).__name__}: {exc}")

        try:
            with _certifi_ssl_context():
                model = torch.hub.load("facebookresearch/dinov2", name, trust_repo=True,
                                       pretrained=weights_file is None)
                if weights_file is not None:
                    self._load_state_into(model, weights_file)
                return model
        except Exception as exc:                         # pragma: no cover - phụ thuộc mạng
            errors.append(f"github: {type(exc).__name__}: {exc}")
            raise RuntimeError(
                "Không nạp được backbone DINOv2. Các cách đã thử:\n  - "
                + "\n  - ".join(errors)
                + "\n\nChạy offline: clone https://github.com/facebookresearch/dinov2 rồi đặt\n"
                  "  RPV_DINOV2_REPO=/duong/dan/dinov2\n"
                  f"  RPV_DINOV2_WEIGHTS=/duong/dan/{name}_pretrain.pth\n"
                  "Hoặc tạm dùng baseline không cần mạng: RPV_ENCODER=classical."
            ) from exc

    @staticmethod
    def _load_state_into(model, weights_file: Path) -> None:
        import torch

        state = torch.load(weights_file, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"Trọng số {weights_file.name} thiếu {len(missing)} tham số của backbone "
                f"(ví dụ: {list(missing)[:3]}). Rất có thể sai variant."
            )
        if unexpected:
            log(f"[dinov2] bỏ qua {len(unexpected)} tham số lạ trong {weights_file.name}")

    def _load_head(self, path: Path) -> None:
        """Nạp ProjectionHead và TỪ CHỐI nếu nó thuộc backbone khác.

        Ghép head train trên ViT-S vào ViT-B sẽ không crash nếu chỉ so chiều
        (có khi trùng), nhưng khoảng cách sinh ra vô nghĩa. Vì vậy checkpoint
        lưu kèm ``backbone_tag`` và ở đây so khớp tường minh.
        """
        import torch

        if not path.exists():
            raise FileNotFoundError(
                f"RPV_PROJECTION_CKPT trỏ tới file không tồn tại: {path}. "
                "Bỏ biến này để dùng đặc trưng backbone thô, hoặc chạy "
                "`python -m rpv.cli train-head` để tạo."
            )
        from rpv.vision.projection import ProjectionHead

        head_hash = image_sha256(path)
        state = torch.load(path, map_location=self.device, weights_only=True)
        if image_sha256(path) != head_hash:
            raise RuntimeError('Head thay đổi trong lúc nạp; dừng và nạp lại checkpoint hoàn chỉnh.')
        self.head_metadata = {key: state.get(key, '') for key in
                              ('dataset_fingerprint', 'split_fingerprint', 'development_fingerprint')}
        in_dim = int(state.get("in_dim", self.backbone_dim))
        out_dim = int(state["out_dim"])
        ckpt_backbone = str(state.get("backbone_tag", ""))
        if in_dim != self.backbone_dim:
            raise RuntimeError(
                f"ProjectionHead tại '{path}' nhận đầu vào dim={in_dim} nhưng backbone "
                f"{self.variant} cho dim={self.backbone_dim}. Head này thuộc backbone khác."
            )
        if ckpt_backbone and ckpt_backbone != self.backbone_tag:
            raise RuntimeError(
                f"ProjectionHead tại '{path}' được train trên backbone {ckpt_backbone!r}, "
                f"khác backbone hiện tại {self.backbone_tag!r}. Từ chối nạp để tránh so sánh "
                "khoảng cách giữa hai không gian embedding khác nhau."
            )
        head = ProjectionHead(self.backbone_dim, out_dim).to(self.device)
        head.load_state_dict(state["state_dict"])
        head.eval()
        for p in head.parameters():
            p.requires_grad_(False)
        self.head = head
        self.dim = out_dim
        self.head_path = str(path)
        self._head_sha256 = head_hash
        info = path.stat()
        self._head_stamp = (info.st_mtime_ns, info.st_size)

    def validate_artifacts(self):
        if not self.head_path:
            return
        path = Path(self.head_path)
        if not path.is_file():
            raise RuntimeError('Head đang dùng đã bị xóa. Khởi động lại demo/API với checkpoint đúng.')
        info = path.stat()
        stamp = (info.st_mtime_ns, info.st_size)
        if stamp != self._head_stamp:
            if image_sha256(path) != self._head_sha256:
                raise RuntimeError('Head trên đĩa đã đổi. Khởi động lại demo/API để không chạy head cũ.')
            self._head_stamp = stamp

    # -- tiền xử lý --------------------------------------------------------
    def _preprocess(self, arr: np.ndarray) -> np.ndarray:
        return preprocess_for_vit(arr, self.image_size, self.center_crop, self.crop_ratio, self.resize_mode)

    # -- suy luận ----------------------------------------------------------
    def _backbone_arrays(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Đặc trưng backbone thô (chưa qua head), đã L2-normalize."""
        import torch

        if not images:
            return np.zeros((0, self.backbone_dim), dtype=np.float32)
        batch = np.stack([self._preprocess(np.asarray(a, dtype=np.uint8)) for a in images])
        with self._lock, torch.inference_mode():
            t = torch.from_numpy(batch).to(self.device)
            feats = self.backbone(t)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            if feats.ndim == 3:                     # [B, tokens, D] -> trung bình token
                feats = feats.mean(dim=1)
            out = feats.float().cpu().numpy()
        return l2_normalize(out.astype(np.float32))

    def _apply_head(self, features: np.ndarray) -> np.ndarray:
        import torch

        if self.head is None or features.shape[0] == 0:
            return l2_normalize(np.asarray(features, dtype=np.float32))
        with self._lock, torch.inference_mode():
            t = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(self.device)
            out = self.head(t).float().cpu().numpy()
        return l2_normalize(out.astype(np.float32))

    def embed_arrays(self, images: Sequence[np.ndarray]) -> np.ndarray:
        return self._apply_head(self._backbone_arrays(images))

    # Cache lưu đặc trưng BACKBONE; head chạy sau khi lấy từ cache.
    def _cache_dim(self) -> int:
        return int(self.backbone_dim)

    def _compute_from_arrays(self, arrays: Sequence[np.ndarray], use_tta: bool) -> np.ndarray:
        return self._embed_arrays_tta(arrays, use_tta, self._backbone_arrays)

    def _postprocess(self, vectors: np.ndarray) -> np.ndarray:
        return self._apply_head(vectors)

    def backbone_features(self, paths: Sequence[str | Path], use_tta: bool | None = None) -> np.ndarray:
        """Đặc trưng backbone (bỏ qua head) — dùng để train ProjectionHead."""
        # Keep a shared head attached while retrieving raw cached backbone vectors.
        return l2_normalize(self._cached_features(paths, use_tta=use_tta))

    def describe(self) -> dict:
        return {
            "tag": self.tag, "dim": int(self.dim), "cache_tag": self.cache_tag,
            "variant": self.variant, "backbone_dim": int(self.backbone_dim),
            "image_size": self.image_size, "center_crop": self.center_crop,
            "crop_ratio": self.crop_ratio, 'weights_sha256': self.weights_sha256,
            'resize_mode': self.resize_mode,
            'use_tta': self.use_tta, 'head_metadata': self.head_metadata,
            "device": self.device, "projection_head": self.head_path or None,
        }


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise RuntimeError(
            "Encoder DINOv2 cần PyTorch. Cài bằng:\n"
            "    pip install -r requirements.txt\n"
            "Hoặc tạm dùng baseline không cần torch: RPV_ENCODER=classical "
            "(chỉ dùng để kiểm chứng pipeline, KHÔNG dùng cho số liệu báo cáo)."
        ) from exc


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def load_encoder(name: str | None = None, **kwargs) -> BaseEncoder:
    """Factory. Nếu chọn dinov2 mà môi trường thiếu torch, KHÔNG âm thầm đổi
    encoder: raise lỗi rõ ràng, vì đổi encoder ngầm sẽ làm mọi số liệu so sánh
    trong luận văn mất ý nghĩa."""
    name = (name or app_config.settings.vision.encoder).strip().lower()
    if name in {"dinov2", "dino"}:
        encoder = DinoV2Encoder(**kwargs)
        weight = app_config.settings.vision.prototype_color_weight
        if weight > 0 and kwargs.get('checkpoint') is not False:
            from rpv.vision.color import PrototypeColorEncoder
            return PrototypeColorEncoder(encoder,weight)
        return encoder
    if name in {"classical", "handcrafted", "baseline"}:
        return ClassicalEncoder(**kwargs)
    raise ValueError(f"Encoder không hỗ trợ: {name!r}. Dùng 'dinov2' (mặc định) hoặc 'classical'.")
