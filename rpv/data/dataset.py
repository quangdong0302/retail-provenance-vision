"""Chia dữ liệu — phần dễ làm sai nhất và làm hỏng toàn bộ số liệu.

Ba tầng tách biệt, bắt buộc phải có đủ cả ba:

1. **Theo lớp**: một phần SKU được giữ lại hoàn toàn khỏi quá trình học
   (``novel``), mô phỏng "hàng cửa hàng vừa nhập". Không có tầng này, con số
   "few-shot accuracy" chỉ đo lại những gì model đã thấy.
2. **Theo ảnh trong lớp**: ảnh support (tạo prototype) và ảnh query (để đo)
   không bao giờ trùng nhau.
3. **Theo mục đích**: query lại chia calib (chọn ngưỡng) / test (báo cáo).
   Chọn ngưỡng và báo cáo trên cùng một tập là rò rỉ dữ liệu.

Ảnh gốc được nhập bằng prepare-data. Chỉ dataset có manifest known/unknown
được dùng cho thực nghiệm; unknown không tham gia train hoặc calibration.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class DatasetIndex(dict):
    """Class index with explicit pool membership from known/unknown folders."""

    def __init__(self, *args, unknown_classes=(), records=None, fingerprint="", grouping='content_hash', **kwargs):
        super().__init__(*args, **kwargs)
        self.unknown_classes = set(unknown_classes)
        self.records = records or {}
        self.fingerprint = fingerprint
        self.grouping = grouping


def index_dataset(root: str | Path) -> dict[str, list[str]]:
    """``root/<class_name>/*.jpg`` -> {class_name: [paths]} (đã sort, deterministic)."""
    root = Path(root)
    if (root / "known").is_dir() or (root / "unknown").is_dir():
        from rpv.data.managed import read_managed_dataset
        return read_managed_dataset(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {root}. "
            "Dùng `python -m rpv.cli prepare-data --source <ảnh thật> "
            "--destination data/dataset_v2`, rồi trỏ RPV_DATA_DIR tới dataset đã chuẩn bị."
        )
    out: dict[str, list[str]] = defaultdict(list)
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(cdir.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_EXT:
                out[cdir.name].append(str(f))
    empty = [c for c, v in out.items() if not v]
    for c in empty:
        out.pop(c)
    if not out:
        raise ValueError(f"Thư mục {root} không có lớp nào chứa ảnh hợp lệ.")
    return dict(out)


def split_known_unknown(class_to_paths: DatasetIndex) -> tuple[dict, dict]:
    if not isinstance(class_to_paths, DatasetIndex):
        raise ValueError("Dataset cần manifest known/unknown. Chạy prepare-data trước.")
    unknown = class_to_paths.unknown_classes
    return ({c: list(p) for c, p in class_to_paths.items() if c not in unknown},
            {c: list(p) for c, p in class_to_paths.items() if c in unknown})


@dataclass
class DatasetSplits:
    train: dict[str, list[str]] = field(default_factory=dict)
    validation: dict[str, list[str]] = field(default_factory=dict)
    support: dict[str, list[str]] = field(default_factory=dict)
    calib: dict[str, list[str]] = field(default_factory=dict)
    test: dict[str, list[str]] = field(default_factory=dict)
    unknown_calib: list[str] = field(default_factory=list)
    unknown_test: list[str] = field(default_factory=list)
    train_classes: list[str] = field(default_factory=list)
    novel_classes: list[str] = field(default_factory=list)
    novel_support: dict[str, list[str]] = field(default_factory=dict)
    novel_test: dict[str, list[str]] = field(default_factory=dict)
    protocol: str = "known_unknown_70_30_v1"
    dataset_fingerprint: str = ""
    groups: dict[str, str] = field(default_factory=dict)
    relative_paths: dict[str, str] = field(default_factory=dict)
    grouping: str = 'path'

    def summary(self) -> dict:
        return {
            "protocol": self.protocol,
            "n_train_images": sum(map(len, self.train.values())),
            "n_validation_images": sum(map(len, self.validation.values())),
            "n_known_classes": len(self.support),
            "n_train_classes": len(self.train_classes),
            "n_novel_classes": len(self.novel_classes),
            "n_support_images": sum(len(v) for v in self.support.values()),
            "n_calib_images": sum(len(v) for v in self.calib.values()),
            "n_test_images": sum(len(v) for v in self.test.values()),
            "n_unknown_calib": len(self.unknown_calib),
            "n_unknown_test": len(self.unknown_test),
            "n_novel_support": sum(map(len, self.novel_support.values())),
            "n_novel_query": sum(map(len, self.novel_test.values())),
        }

    def assert_disjoint(self) -> None:
        """Kiểm tra bất biến: không một ảnh nào xuất hiện ở hai vai trò.

        Gọi hàm này trong test tự động. Rò rỉ dữ liệu là loại lỗi không bao giờ
        làm chương trình crash — nó chỉ làm số liệu đẹp lên và làm kết luận sai.
        """
        if set(self.train_classes) & set(self.novel_classes):
            raise AssertionError('Train và novel phải tách biệt theo lớp.')
        if self.protocol != 'known_unknown_70_30_v1':
            raise ValueError("Protocol không hỗ trợ. Chạy prepare-data để tạo dataset known/unknown.")
        allowed = set(self.train_classes)
        for role in ('train', 'validation', 'support', 'calib', 'test'):
            if set(getattr(self, role)) - allowed:
                raise AssertionError(f'{role} chứa lớp ngoài known.')
        if self.unknown_calib:
            raise AssertionError('Protocol không dùng unknown để hiệu chỉnh.')
        buckets = {
            "train": {p for v in self.train.values() for p in v},
            "validation": {p for v in self.validation.values() for p in v},
            "support": {p for v in self.support.values() for p in v},
            "calib": {p for v in self.calib.values() for p in v},
            "test": {p for v in self.test.values() for p in v},
            "unknown_calib": set(self.unknown_calib),
            "unknown_test": set(self.unknown_test),
        }
        names = list(buckets)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = buckets[a] & buckets[b]
                if overlap:
                    raise AssertionError(
                        f"RÒ RỈ DỮ LIỆU: {len(overlap)} ảnh xuất hiện ở cả '{a}' và '{b}'. "
                        f"Ví dụ: {sorted(overlap)[:3]}"
                    )
                if self.groups:
                    ga = {self.groups[p] for p in buckets[a]}
                    gb = {self.groups[p] for p in buckets[b]}
                    if ga & gb:
                        raise AssertionError(f"Nhóm chụp trùng giữa {a} và {b}.")
        # Novel evaluation deliberately reuses the unknown pool in a SECOND
        # experiment, after enrollment. Its support/query must remain disjoint.
        ns = {p for ps in self.novel_support.values() for p in ps}
        nt = {p for ps in self.novel_test.values() for p in ps}
        if ns & nt:
            raise AssertionError("Novel support/query overlap.")
        if self.groups and {self.groups[p] for p in ns} & {self.groups[p] for p in nt}:
            raise AssertionError("Novel support/query share capture groups.")


def build_splits(class_to_paths: DatasetIndex, k_shot: int,
                 class_seed: int, calib_test_seed: int) -> DatasetSplits:
    if not isinstance(class_to_paths, DatasetIndex):
        raise ValueError(
            "Train/calibrate/evaluate cần dataset có manifest known/unknown. "
            "Chạy python -m rpv.cli prepare-data --source <ảnh gốc> "
            "--destination data/dataset_v2, rồi cập nhật RPV_DATA_DIR."
        )
    from rpv.data.managed import build_managed_splits
    return build_managed_splits(class_to_paths, k_shot, class_seed, calib_test_seed)
