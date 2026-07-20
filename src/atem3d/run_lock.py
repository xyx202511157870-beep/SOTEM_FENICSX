"""Dependency-free cross-process writer lock with same-thread reentrancy."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import threading


_THREAD_STATE = threading.local()


@contextmanager
def run_lock(run_dir: str | Path):
    """Hold a non-blocking writer lock for one run directory."""

    run_path = Path(run_dir).expanduser().resolve()
    lock_path = run_path.parent / f".{run_path.name}.writer.lock"
    key = str(lock_path)
    held = getattr(_THREAD_STATE, "held", None)
    if held is None:
        held = {}
        _THREAD_STATE.held = held
    if key in held:
        held[key] += 1
        try:
            yield
        finally:
            held[key] -= 1
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"another writer holds the run lock: {run_path}") from exc
        acquired = True
        held[key] = 1
        yield
    finally:
        if acquired:
            held.pop(key, None)
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
