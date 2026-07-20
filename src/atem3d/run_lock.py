"""Dependency-free cross-process writer lock with same-thread reentrancy."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import threading


_THREAD_STATE = threading.local()
_REGISTRY_GUARD = threading.RLock()
_OPEN_HANDLES: set[object] = set()


def _after_fork_in_child() -> None:
    """Close inherited descriptors and clear child-local reentrancy state."""

    global _THREAD_STATE, _REGISTRY_GUARD, _OPEN_HANDLES
    for handle in tuple(_OPEN_HANDLES):
        try:
            handle.close()
        except OSError:
            pass
    _OPEN_HANDLES = set()
    _THREAD_STATE = threading.local()
    _REGISTRY_GUARD = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


@contextmanager
def run_lock(run_dir: str | Path):
    """Hold a non-blocking writer lock for one run directory."""

    run_path = Path(run_dir).expanduser().resolve()
    lock_path = run_path.parent / f".{run_path.name}.writer.lock"
    key = str(lock_path)
    owner_pid = os.getpid()
    held = getattr(_THREAD_STATE, "held", None)
    if held is None:
        held = {}
        _THREAD_STATE.held = held
    entry = held.get(key)
    if entry is not None and int(entry["pid"]) == owner_pid:
        entry["depth"] += 1
        try:
            yield
        finally:
            entry["depth"] -= 1
        return
    if entry is not None:
        inherited_handle = entry.get("handle")
        if inherited_handle is not None:
            try:
                inherited_handle.close()
            except OSError:
                pass
            with _REGISTRY_GUARD:
                _OPEN_HANDLES.discard(inherited_handle)
        held.pop(key, None)

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    with _REGISTRY_GUARD:
        _OPEN_HANDLES.add(handle)
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
        held[key] = {"pid": owner_pid, "depth": 1, "handle": handle}
        yield
    finally:
        if acquired and os.getpid() == owner_pid:
            held.pop(key, None)
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        with _REGISTRY_GUARD:
            _OPEN_HANDLES.discard(handle)
        if not handle.closed:
            handle.close()
