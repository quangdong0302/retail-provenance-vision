"""Tiện ích dùng chung — hash ổn định và JSON an toàn.

Hai vấn đề nhỏ nhưng đã gây lỗi thật trong dự án này, nên được đưa về một chỗ:

* ``hash()`` của Python được RANDOM HOÁ theo từng tiến trình (PYTHONHASHSEED).
  Dùng nó để chọn mẫu đánh giá hay để sinh ID nghĩa là chạy hai lần cho hai kết
  quả khác nhau — đúng loại lỗi phá tính tái lập mà không bao giờ crash.
  ``stable_hash_int`` / ``stable_hash_hex`` thay thế, dựa trên BLAKE2b.

* ``json.dumps`` mặc định cho phép ghi ``NaN`` / ``Infinity``. Đây KHÔNG phải
  JSON hợp lệ: ``json.loads`` của Python đọc được, nhưng client khác
  (JavaScript, Go, jq nghiêm ngặt) sẽ lỗi. Hệ thống này sinh ``inf`` một cách
  hợp lý (khoảng cách khi danh mục rỗng, ngưỡng khi chưa hiệu chỉnh), nên phải
  làm sạch ở biên đầu ra thay vì cấm ``inf`` trong tính toán.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from typing import Any

_SEP = "\x1f"


def log(*parts: Any) -> None:
    """In tiến trình ra **stderr**.

    stdout của CLI phải là JSON thuần để nối được với ``jq`` hay đọc bằng script.
    Trộn log tiến trình vào stdout làm mọi lệnh dạng
    ``python -m rpv.cli calibrate | jq .closed_set`` thất bại — một chi tiết nhỏ
    nhưng là khác biệt giữa "có CLI" và "CLI dùng được".
    """
    print(*parts, file=sys.stderr, flush=True)


def stable_hash_bytes(*parts: Any) -> bytes:
    blob = _SEP.join(str(p) for p in parts).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=32).digest()


def stable_hash_int(*parts: Any) -> int:
    """Số nguyên không âm, ổn định giữa các lần chạy và giữa các máy."""
    return int.from_bytes(stable_hash_bytes(*parts)[:8], "big")


def stable_hash_hex(*parts: Any, length: int = 64) -> str:
    """Hex ổn định (mặc định 64 ký tự, cùng độ dài sha256)."""
    return stable_hash_bytes(*parts).hex()[:length]


def json_safe(obj: Any) -> Any:
    """Đệ quy thay ``NaN``/``±Infinity`` bằng ``None`` để JSON luôn hợp lệ.

    Giữ nguyên mọi giá trị khác: không làm tròn, không đổi kiểu — báo cáo khoa
    học không được sửa số liệu ở bước ghi file.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    return obj


def json_dumps(obj: Any, *, indent: int | None = 2, sort_keys: bool = False) -> str:
    """``json.dumps`` đã làm sạch + ``allow_nan=False``: không bao giờ im lặng
    ghi ra JSON không hợp lệ."""
    return json.dumps(json_safe(obj), indent=indent, ensure_ascii=False,
                      allow_nan=False, sort_keys=sort_keys, default=str)
