"""Hash ổn định và xử lý JSON.

``hash()`` của Python thay đổi theo PYTHONHASHSEED giữa các tiến trình.
``stable_hash_int`` và ``stable_hash_hex`` dùng BLAKE2b để giữ kết quả chọn
mẫu và ID ổn định giữa các lượt chạy.

``json.dumps`` mặc định cho phép NaN và Infinity, trong khi đây không phải
giá trị JSON hợp lệ. Python đọc được nhưng các client khác có thể báo lỗi.
Khoảng cách khi danh mục rỗng hoặc ngưỡng chưa hiệu chỉnh có thể là inf;
các giá trị này được xử lý lúc xuất JSON, giữ nguyên trong tính toán.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from typing import Any

_SEP = "\x1f"


def log(*parts: Any) -> None:
    """Ghi tiến trình ra stderr, giữ stdout của CLI là JSON.
    Các lệnh như ``python -m rpv.cli calibrate | jq .closed_set``
    có thể đọc trực tiếp kết quả mà không lẫn log.
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
    """Thay đệ quy NaN và ±Infinity bằng None để xuất JSON hợp lệ.
    Các giá trị khác giữ nguyên, không làm tròn hay đổi kiểu khi ghi báo cáo.
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
