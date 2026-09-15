"""Reentrant locks shared by cooperating threads and processes on a local volume."""
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import threading
import time


_GUARD = threading.Lock()
_LOCKS = {}
_HELD = threading.local()


def path_lock(path: Path):
    key = os.path.normcase(str(path.resolve()))
    with _GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


@contextmanager
def file_lock(path: Path):
    """Lock a stable sidecar; never delete it while any writer is running.

    A thread may reenter the same lock (transaction -> save). Other threads
    and processes still wait until the outermost transaction has finished.
    The OS releases the lock even if its owning process crashes.
    """
    path = Path(path).resolve()
    key = os.path.normcase(str(path))
    with path_lock(path):
        held = getattr(_HELD, 'paths', None)
        if held is None:
            held = _HELD.paths = set()
        if key in held:
            yield
            return
        with open(path, 'a+b') as handle:
            if os.name == 'nt':
                import msvcrt
                deadline = time.monotonic() + 30
                while True:
                    try:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as exc:
                        if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                            raise
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f'Timed out acquiring file lock: {path}') from exc
                        time.sleep(.01)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            held.add(key)
            try:
                yield
            finally:
                held.remove(key)
                if os.name == 'nt':
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
