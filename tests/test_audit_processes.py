"""Exercise the real JSONL writer from independent Python processes."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from rpv.audit.hash_chain import AuditLog


ROOT = Path(__file__).resolve().parents[1]
WORKER = r'''
import sys
import time
from rpv.audit.hash_chain import AuditLog

log = AuditLog(sys.argv[1])
if sys.argv[3] == "delayed":
    read_tail = log._read_tail_entry
    def delayed_read():
        entry = read_tail()
        # Force overlap after reading the head, before writing the next entry.
        time.sleep(0.01)
        return entry
    log._read_tail_entry = delayed_read
sys.stdin.readline()
for item in range(20):
    log.append("process_test", {"worker": int(sys.argv[2]), "item": item})
    time.sleep(0.001)
'''


class AuditProcessTests(unittest.TestCase):
    def _concurrent_appends(self, mode: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            # Prime a different object's cache before the children update the file.
            log = AuditLog(path)
            log.append("seed", {})
            processes = []
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
            try:
                for worker in range(4):
                    processes.append(subprocess.Popen(
                        [sys.executable, "-c", WORKER, str(path), str(worker), mode],
                        cwd=ROOT, env=env, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                        encoding="utf-8",
                    ))
                for process in processes:
                    process.stdin.write("start\n")
                    process.stdin.flush()
                for process in processes:
                    stdout, stderr = process.communicate(timeout=45)
                    self.assertEqual(process.returncode, 0, stdout + stderr)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.communicate()

            report = log.verify()
            self.assertEqual(report["n_entries"], 81,
                             f'valid={report["valid"]}, errors={len(report["problems"])}')
            self.assertTrue(report["valid"],
                            f'{len(report["problems"])} errors: {report["problems"][:4]}')
            self.assertEqual(log.last().index, 80)
            observed = {(e.payload["worker"], e.payload["item"])
                        for e in log.entries() if e.event_type == "process_test"}
            self.assertEqual(observed, {(w, i) for w in range(4) for i in range(20)})

    def test_chain_stays_valid_across_processes(self):
        self._concurrent_appends("normal")

    def test_process_lock_covers_read_head_through_append(self):
        self._concurrent_appends("delayed")

    def test_large_unicode_tail_can_be_read_by_a_new_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            log = AuditLog(path)
            payload = {"description": "Hàng hóa 🍊 " * 20000}
            log.append("large", payload)
            other = AuditLog(path)
            self.assertEqual(other.last().payload, payload)
            other.append("next", {})
            self.assertTrue(log.verify()["valid"])
            self.assertEqual(log.last().index, 1)

    def test_failed_append_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            log = AuditLog(path)
            with self.assertRaises(TypeError):
                log.append("invalid", {"not_json": object()})
            AuditLog(path).append("valid", {})
            self.assertTrue(log.verify()["valid"])
            self.assertEqual(len(log), 1)

    def test_process_exit_releases_lock(self):
        holder_code = r'''
import sys
from pathlib import Path
from rpv.audit.hash_chain import AuditLog, _process_lock
log = AuditLog(sys.argv[1])
with _process_lock(log._process_lock_path):
    Path(sys.argv[2]).touch()
    sys.stdin.readline()
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            ready = Path(directory) / "ready"
            holder = subprocess.Popen(
                [sys.executable, "-c", holder_code, str(path), str(ready)],
                cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(os.environ, PYTHONPATH=str(ROOT)),
            )
            try:
                deadline = time.monotonic() + 15
                while not ready.exists():
                    if holder.poll() is not None or time.monotonic() > deadline:
                        self.fail("Child did not acquire the audit lock")
                    time.sleep(0.01)
                holder.kill()  # No Python finally block: the OS must release it.
                holder.communicate(timeout=10)
                log = AuditLog(path)
                log.append("after_crash", {})
                self.assertTrue(log.verify()["valid"])
            finally:
                if holder.poll() is None:
                    holder.kill()
                holder.communicate()


if __name__ == "__main__":
    unittest.main()
