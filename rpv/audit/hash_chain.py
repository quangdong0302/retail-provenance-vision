"""Audit trail dạng hash chain (append-only).

Vì sao cần trong một hệ thống kiểm chứng nguồn gốc: nếu ai cũng có thể sửa lịch
sử quyết định, thì "hệ thống kiểm chứng" chỉ là lời hứa. Mỗi bản ghi chứa hash
của bản ghi trước, nên sửa/xoá một dòng ở giữa sẽ làm đứt chuỗi và bị
``verify_chain`` phát hiện ngay.

Đây KHÔNG phải blockchain và luận văn không nên gọi nó như vậy: không có đồng
thuận phân tán, chỉ có tính toàn vẹn tuần tự trong phạm vi một cửa hàng — đúng
mức cần thiết cho bài toán này.

Hai chi tiết hiện thực đáng nói:

* ``append`` không đọc lại toàn bộ file để tìm bản ghi cuối. Bản trước làm vậy
  nên ghi N bản ghi tốn O(N^2) lượt đọc — chạy vài nghìn giao dịch là thấy ngay.
  Ở đây phần đuôi file được đọc một lần và cache lại, kèm kiểm tra kích thước
  file để phát hiện tiến trình khác vừa ghi thêm.
* ``append`` giữ khoá thread và khoá liên tiến trình từ trước khi đọc đuôi
  đến sau ``fsync``. API và demo phải dùng chung file log và file ``.lock``
  trên cùng volume. Không xoá file khoá khi còn tiến trình đang sử dụng.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict, fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from rpv.file_lock import file_lock as _process_lock, path_lock as _path_lock

GENESIS = "0" * 64
_TAIL_BYTES = 1 << 16
def _canonical(payload: Any) -> str:
    """Chuẩn hoá JSON trước khi hash: sort_keys + separators cố định. Không có
    bước này, cùng một dữ liệu hash ra hai kết quả khác nhau tuỳ thứ tự khoá."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_hash(index: int, timestamp: str, prev_hash: str, event_type: str, payload: Any) -> str:
    blob = _canonical({"i": index, "ts": timestamp, "prev": prev_hash, "type": event_type, "payload": payload})
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class AuditEntry:
    index: int
    timestamp: str
    event_type: str
    payload: dict
    prev_hash: str
    entry_hash: str

    def to_json(self) -> dict:
        return asdict(self)

    def recompute(self) -> str:
        return compute_hash(self.index, self.timestamp, self.prev_hash, self.event_type, self.payload)

    @staticmethod
    def from_json(d: dict) -> "AuditEntry":
        known = {f.name for f in dataclass_fields(AuditEntry)}
        missing = known - set(d)
        if missing:
            raise ValueError(f"Bản ghi audit thiếu trường bắt buộc: {sorted(missing)}")
        return AuditEntry(**{k: v for k, v in d.items() if k in known})


class AuditLog:
    """Log JSONL append-only. Mỗi dòng là 1 ``AuditEntry``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        # Mọi instance trong cùng process dùng chung khoá theo đường dẫn. API có
        # service log và status-change log riêng, nên khoá theo instance là sai.
        self._lock = _path_lock(self.path)
        self._process_lock_path = self.path.with_name(self.path.name + ".lock")
        self._head: Optional[AuditEntry] = None
        self._head_size: int = -1

    def __len__(self) -> int:
        return sum(1 for _ in self.entries())

    def entries(self) -> Iterator[AuditEntry]:
        # Snapshot under the same process lock; never parse a half-written line
        # or hold an OS lock while a caller pauses iteration.
        with self._lock, _process_lock(self._process_lock_path):
            with open(self.path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        for lineno, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield AuditEntry.from_json(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Chuỗi audit '{self.path}' hỏng ở dòng {lineno}: {exc}"
                ) from exc

    def _read_tail_entry(self) -> Optional[AuditEntry]:
        """Đọc bản ghi cuối bằng cách chỉ lấy phần đuôi file."""
        size = self.path.stat().st_size
        if size == 0:
            return None
        with open(self.path, "rb") as fh:
            window = _TAIL_BYTES
            while True:
                start = max(0, size - window)
                fh.seek(start)
                lines = fh.read().split(b"\n")
                if start:
                    # The first fragment may start inside JSON or a UTF-8 codepoint.
                    lines = lines[1:]
                for line in reversed(lines):
                    if line.strip():
                        return AuditEntry.from_json(json.loads(line.decode("utf-8")))
                if start == 0:
                    return None
                window *= 2  # Also supports records larger than 64 KiB.

    def _last_unlocked(self) -> Optional[AuditEntry]:
        """Caller must hold both locks, including while consulting the cache."""
        size = self.path.stat().st_size
        if self._head is not None and size == self._head_size:
            return self._head
        self._head = self._read_tail_entry()
        self._head_size = size
        return self._head

    def last(self) -> Optional[AuditEntry]:
        with self._lock, _process_lock(self._process_lock_path):
            return self._last_unlocked()

    def append(self, event_type: str, payload: dict) -> AuditEntry:
        with self._lock, _process_lock(self._process_lock_path):
            prev = self._last_unlocked()
            index = 0 if prev is None else prev.index + 1
            prev_hash = GENESIS if prev is None else prev.entry_hash
            ts = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            entry = AuditEntry(
                index=index, timestamp=ts, event_type=event_type, payload=payload,
                prev_hash=prev_hash,
                entry_hash=compute_hash(index, ts, prev_hash, event_type, payload),
            )
            line = json.dumps(entry.to_json(), ensure_ascii=False, sort_keys=True) + "\n"
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            self._head = entry
            self._head_size = self.path.stat().st_size
            return entry

    def verify(self) -> dict:
        try:
            return verify_chain(list(self.entries()))
        except ValueError as exc:
            return {"valid": False, "n_entries": 0, "problems": [str(exc)], "head": GENESIS}


def verify_chain(entries: list[AuditEntry]) -> dict:
    """Kiểm tra tính toàn vẹn: chỉ số liên tục, hash khớp, liên kết prev đúng."""
    problems: list[str] = []
    prev_hash = GENESIS
    for i, e in enumerate(entries):
        if e.index != i:
            problems.append(f"Bản ghi #{i}: index={e.index}, kỳ vọng {i} (có dòng bị xoá hoặc chèn).")
        if e.prev_hash != prev_hash:
            problems.append(f"Bản ghi #{i}: prev_hash không khớp bản ghi trước (chuỗi bị đứt).")
        if e.recompute() != e.entry_hash:
            problems.append(f"Bản ghi #{i}: nội dung đã bị sửa sau khi ghi (hash không khớp).")
        prev_hash = e.entry_hash
    return {"valid": not problems, "n_entries": len(entries), "problems": problems, "head": prev_hash}
