"""Cấu hình dùng chung của dự án.

Encoder chính là DINOv2 đóng băng. ``classical`` là baseline descriptor thủ
công để so sánh đóng góp của backbone và chạy test khi máy không có PyTorch.
Thực nghiệm chính sử dụng DINOv2.
"""
from __future__ import annotations

import os
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _strip_inline_comment(value: str) -> str:
    """Bỏ comment của dotenv nhưng giữ dấu # nằm trong chuỗi có quote."""
    quote = ""
    for i, char in enumerate(value):
        if char in {"'", '"'}:
            if not quote:
                quote = char
            elif quote == char:
                quote = ""
        elif char == "#" and not quote and (i == 0 or value[i - 1].isspace()):
            return value[:i].rstrip()
    return value.strip()


def load_env_file(path: str | Path = ROOT / ".env", *, override: bool = False) -> int:
    """Nạp file ``.env`` không cần dependency ngoài.

    Biến đã có trong môi trường luôn thắng mặc định. Parser giữ nguyên dấu
    gạch chéo ngược trong đường dẫn Windows và đọc được file có UTF-8 BOM.
    """
    path = Path(path)
    if not path.is_file():
        return 0
    loaded = 0
    preceding_comment = ''
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if line.startswith('#'):
            preceding_comment = line
            continue
        if not line:
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"{path}:{lineno}: dòng .env thiếu dấu '='.")
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"{path}:{lineno}: tên biến .env không hợp lệ: {key!r}.")
        value = _strip_inline_comment(value.strip())
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
            loaded += 1
        elif ((not key.endswith(('_DIR', '_PATH', '_REPO', '_WEIGHTS', '_CKPT', '_OUT',
                                 '_DB', '_BANK', '_CALIB', '_AUDIT', '_REPORTS'))
               or (not value and os.environ[key].strip() == preceding_comment))
              and os.environ[key] != value and _strip_inline_comment(os.environ[key].strip()) == value):
            # Some launchers pre-load .env without removing inline comments.
            # Repair only an inherited value matching this file after comment
            # removal; keep genuine environment overrides and literal # paths.
            os.environ[key] = value
            loaded += 1
        preceding_comment = ''
    return loaded


# README hướng dẫn sao chép .env.example thành .env, vì vậy phải nạp nó thật.
# Trước đây file này bị bỏ qua hoàn toàn, đặc biệt gây nhầm lẫn trên Windows.
load_env_file()


def _env(key: str, default):
    """Đọc biến môi trường theo kiểu của giá trị mặc định.
    Giá trị sai kiểu gây lỗi cấu hình, không được thay bằng mặc định.
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    raw = raw.strip()
    try:
        if isinstance(default, bool):
            if raw.lower() in {"1", "true", "yes", "on"}:
                return True
            if raw.lower() in {"0", "false", "no", "off", ""}:
                return False
            raise ValueError("cần 1/0/true/false")
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Biến môi trường {key}={raw!r} không hợp lệ (kỳ vọng "
            f"{type(default).__name__}): {exc}"
        ) from exc
    return raw


@dataclass(frozen=True)
class VisionConfig:
    # "dinov2"    = DINOv2 frozen (mặc định, dùng cho mọi số liệu báo cáo).
    # "classical" = descriptor numpy thuần; baseline + chạy được khi không có torch.
    encoder: str = field(default_factory=lambda: _env("RPV_ENCODER", "dinov2"))
    dinov2_variant: str = field(default_factory=lambda: _env("RPV_DINOV2_VARIANT", "vits14"))
    # Chạy hoàn toàn offline: trỏ tới bản clone facebookresearch/dinov2 và/hoặc
    # file trọng số .pth đã tải sẵn. Các đường dẫn này hỗ trợ chạy khi không có mạng.
    dinov2_repo: str = field(default_factory=lambda: _env("RPV_DINOV2_REPO", ""))
    dinov2_weights: str = field(default_factory=lambda: _env("RPV_DINOV2_WEIGHTS", ""))
    device: str = field(default_factory=lambda: _env("RPV_DEVICE", ""))       # "" = tự chọn
    batch_size: int = field(default_factory=lambda: _env("RPV_BATCH_SIZE", 32))
    image_size: int = field(default_factory=lambda: _env("RPV_IMAGE_SIZE", 224))
    # Giữ cờ center_crop cho API cũ; encoder v2 chọn phép biến đổi bằng resize_mode.
    center_crop: bool = field(default_factory=lambda: _env("RPV_CENTER_CROP", True))
    crop_ratio: float = field(default_factory=lambda: _env("RPV_CROP_RATIO", 0.875))
    # pad preserves the complete object and aspect ratio, including long produce.
    resize_mode: str = field(default_factory=lambda: _env('RPV_RESIZE_MODE', 'pad'))
    # Số tâm (centroid) mỗi SKU. >1 cho phép 1 SKU có nhiều "hình thái"
    # (độ chín, có/không túi nilon) mà không cần tách thành nhiều lớp.
    centroids_per_sku: int = field(default_factory=lambda: _env("RPV_CENTROIDS", 2))
    top_k: int = field(default_factory=lambda: _env("RPV_TOP_K", 5))
    # Test-time augmentation: lật ngang rồi trung bình embedding.
    use_tta: bool = field(default_factory=lambda: _env("RPV_TTA", True))
    # Cache embedding theo nội dung ảnh: bắt buộc trong thực tế với DINOv2, vì
    # `calibrate`/`evaluate`/`train-head` đọc lại cùng bộ ảnh nhiều lần.
    use_cache: bool = field(default_factory=lambda: _env("RPV_EMBED_CACHE", True))
    cache_max_entries: int = field(default_factory=lambda: _env("RPV_CACHE_MAX", 200_000))
    checkpoint: str = field(default_factory=lambda: _env("RPV_PROJECTION_CKPT", ""))
    catalog_backend: str = field(default_factory=lambda: _env('RPV_CATALOG_BACKEND', 'prototype'))
    prototype_color_weight: float = field(default_factory=lambda: _env('RPV_PROTOTYPE_COLOR_WEIGHT',0.0))

    def __post_init__(self):
        if not math.isfinite(self.prototype_color_weight) or self.prototype_color_weight < 0:
            raise ValueError('RPV_PROTOTYPE_COLOR_WEIGHT must be finite and nonnegative.')
        if self.catalog_backend != 'prototype':
            raise ValueError('RPV_CATALOG_BACKEND must be prototype; restart an old kernel after changing .env.')
        for name in ('batch_size', 'image_size', 'centroids_per_sku', 'cache_max_entries'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'VisionConfig.{name} phải là số nguyên dương, nhận {value!r}.')
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError('RPV_TOP_K phải là số nguyên >= 0.')
        if not math.isfinite(self.crop_ratio) or not .5 <= self.crop_ratio <= 1:
            raise ValueError('RPV_CROP_RATIO phải hữu hạn, trong [0.5, 1].')
        if self.resize_mode not in {'pad', 'crop', 'stretch'}:
            raise ValueError('RPV_RESIZE_MODE phải là pad, crop hoặc stretch.')

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


@dataclass(frozen=True)
class CalibrationConfig:
    decision_rule: str = field(default_factory=lambda: _env('RPV_DECISION_RULE', 'conformal'))
    conformal_score: str = field(default_factory=lambda: _env('RPV_CONFORMAL_SCORE', 'distance'))
    confidence_method: str = field(default_factory=lambda: _env('RPV_CONFIDENCE_METHOD', 'energy'))
    risk_calibration: str = field(default_factory=lambda: _env('RPV_RISK_CALIBRATION', 'known'))
    ood_method: str = field(default_factory=lambda: _env('RPV_OOD_METHOD', 'energy'))
    energy_temperature: float = field(default_factory=lambda: _env('RPV_ENERGY_TEMPERATURE', 0.05))
    target_selective_risk: float = field(default_factory=lambda: _env('RPV_TARGET_RISK', 0.01))
    # alpha = mức rủi ro mục tiêu của conformal prediction (1 - coverage).
    alpha: float = field(default_factory=lambda: _env("RPV_ALPHA", 0.10))
    # Ràng buộc nghiệp vụ khi hiệu chỉnh ngưỡng open-set.
    target_known_recall: float = field(default_factory=lambda: _env("RPV_TARGET_KNOWN_RECALL", 0.90))
    # Số mẫu calib tối thiểu để cho phép ngưỡng riêng theo lớp; dưới mức này
    # lớp đó dùng ngưỡng toàn cục (tránh ngưỡng "học" từ 2 điểm dữ liệu).
    min_calib_per_class: int = field(default_factory=lambda: _env("RPV_MIN_CALIB", 8))
    class_split_seed: int = field(default_factory=lambda: _env("RPV_CLASS_SPLIT_SEED", 1))
    calib_test_seed: int = field(default_factory=lambda: _env("RPV_CALIB_TEST_SEED", 123))
    k_shot: int = field(default_factory=lambda: _env("RPV_K_SHOT", 3))

    def __post_init__(self):
        if self.decision_rule not in {'conformal', 'prototype_selective'}:
            raise ValueError('RPV_DECISION_RULE must be conformal or prototype_selective.')
        if self.conformal_score not in {'distance', 'relative'}:
            raise ValueError('RPV_CONFORMAL_SCORE must be distance or relative.')
        if self.confidence_method not in {'energy', 'nearest_ratio'}:
            raise ValueError('Unsupported confidence method.')
        if self.risk_calibration not in {'known', 'joint_extra', 'reserved_extra', 'six_extra_cal'}:
            raise ValueError('Unsupported risk calibration protocol.')
        if not math.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError('RPV_ALPHA phải hữu hạn, trong (0, 1).')
        if not math.isfinite(self.target_known_recall) or not 0 < self.target_known_recall <= 1:
            raise ValueError('RPV_TARGET_KNOWN_RECALL phải hữu hạn, trong (0, 1].')
        if not math.isfinite(self.target_selective_risk) or not 0 <= self.target_selective_risk < 1:
            raise ValueError('RPV_TARGET_RISK phải hữu hạn, trong [0, 1).')
        if not math.isfinite(self.energy_temperature) or self.energy_temperature <= 0:
            raise ValueError('RPV_ENERGY_TEMPERATURE phải hữu hạn và > 0.')
        for name in ('k_shot', 'min_calib_per_class'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'CalibrationConfig.{name} phải là số nguyên dương.')
        if self.ood_method not in {'energy', 'distance'}:
            raise ValueError('RPV_OOD_METHOD phải là energy hoặc distance.')


@dataclass(frozen=True)
class PolicyConfig:
    # Giá trị giao dịch (VND) vượt mức này thì luôn cần người kiểm tra khi có
    # bất kỳ cảnh báo, kể cả cảnh báo nhẹ.
    high_value_threshold: float = field(default_factory=lambda: _env("RPV_HIGH_VALUE", 500_000.0))
    # Số ngày trước hạn sử dụng bắt đầu cảnh báo.
    expiry_warning_days: int = field(default_factory=lambda: _env("RPV_EXPIRY_WARN_DAYS", 3))
    # Lời khai vẫn nằm trong bán kính chấp nhận NHƯNG có SKU khác gần hơn quá
    # mức này (khoảng cách embedding) thì bắt buộc gọi người: đây là dạng đổi mã
    # "hợp lệ về ngưỡng" nhưng đáng ngờ về hành vi.
    verify_margin_assist: float = field(default_factory=lambda: _env("RPV_VERIFY_MARGIN_ASSIST", 0.15))
    # Chứng từ bắt buộc theo nhóm hàng. Thiếu -> PROVENANCE_INCOMPLETE.
    required_evidence: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "fresh_produce": ("delivery_note", "supplier_declaration"),
            "imported": ("delivery_note", "import_customs_doc", "phytosanitary_cert"),
            "packaged": ("delivery_note",),
            "default": ("delivery_note",),
        }
    )

    def __post_init__(self):
        if not math.isfinite(self.high_value_threshold) or self.high_value_threshold < 0:
            raise ValueError('RPV_HIGH_VALUE phải hữu hạn và >= 0.')
        if type(self.expiry_warning_days) is not int or self.expiry_warning_days < 0:
            raise ValueError('RPV_EXPIRY_WARN_DAYS phải là số nguyên >= 0.')
        if not math.isfinite(self.verify_margin_assist) or self.verify_margin_assist < 0:
            raise ValueError('RPV_VERIFY_MARGIN_ASSIST phải hữu hạn và >= 0.')


@dataclass(frozen=True)
class Paths:
    data_dir: Path = field(default_factory=lambda: Path(_env("RPV_DATA_DIR", str(ROOT / "data" / "dataset_v2"))))
    db_path: Path = field(default_factory=lambda: Path(_env("RPV_DB", str(ROOT / "var" / "thesis" / "rpv.db"))))
    bank_path: Path = field(default_factory=lambda: Path(_env("RPV_BANK", str(ROOT / "var" / "thesis" / "prototypes.json"))))
    calib_path: Path = field(default_factory=lambda: Path(_env("RPV_CALIB", str(ROOT / "var" / "thesis" / "calibration.json"))))
    audit_path: Path = field(default_factory=lambda: Path(_env("RPV_AUDIT", str(ROOT / "var" / "thesis" / "audit.jsonl"))))
    reports_dir: Path = field(default_factory=lambda: Path(_env("RPV_REPORTS", str(ROOT / "var" / "thesis" / "reports"))))
    cache_dir: Path = field(default_factory=lambda: Path(_env("RPV_CACHE_DIR", str(ROOT / "var" / "embed_cache"))))
    head_path: Path = field(default_factory=lambda: Path(_env("RPV_HEAD_OUT", str(ROOT / "var" / "thesis" / "projection_head.pt"))))

    def ensure(self) -> None:
        for p in (self.db_path, self.bank_path, self.calib_path, self.audit_path, self.head_path):
            p.parent.mkdir(parents=True, exist_ok=True)
        for d in (self.reports_dir, self.cache_dir, self.data_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    vision: VisionConfig = field(default_factory=VisionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    paths: Paths = field(default_factory=Paths)
    # Adapter Internet tắt mặc định để chạy offline và tái lập kết quả,
    # và dữ liệu cộng đồng không thay thế hồ sơ lô (xem docs/ARCHITECTURE.md).
    enable_external_adapters: bool = field(default_factory=lambda: _env("RPV_EXTERNAL_ADAPTERS", False))
    training_protocol: str = field(default_factory=lambda: _env('RPV_TRAINING_PROTOCOL', 'episodic'))


settings = Settings()


def reload_settings() -> Settings:
    """Nạp lại cấu hình sau khi thay biến môi trường.

    ``settings`` được tạo lúc import. Thay ``os.environ`` trong notebook hoặc
    test cần gọi hàm này để cập nhật cấu hình đang sử dụng.
    """
    global settings
    settings = Settings()
    return settings
