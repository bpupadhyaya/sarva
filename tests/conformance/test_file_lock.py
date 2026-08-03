"""Conformance tests for sarva.file_lock — the small, shared cross-process
exclusive lock extracted from sarva.config/SessionStore's own
independently-verified implementations. Real proof it actually
serializes concurrent callers, not just that the context manager
enters/exits without raising."""

from __future__ import annotations

import threading
import time

from sarva.file_lock import exclusive_lock


def test_exclusive_lock_serializes_concurrent_threads(tmp_path):
    lock_path = tmp_path / "test.lock"
    order: list[str] = []

    def critical_section(name: str) -> None:
        with exclusive_lock(lock_path):
            order.append(f"{name}-start")
            time.sleep(0.05)  # long enough that an unlocked caller would interleave
            order.append(f"{name}-end")

    t1 = threading.Thread(target=critical_section, args=("a",))
    t2 = threading.Thread(target=critical_section, args=("b",))
    t1.start()
    time.sleep(0.01)  # ensure t1 acquires first
    t2.start()
    t1.join()
    t2.join()

    # Each thread's start/end must be adjacent -- never interleaved
    # (a-start, b-start, a-end, b-end would prove no real serialization).
    assert order in (
        ["a-start", "a-end", "b-start", "b-end"],
        ["b-start", "b-end", "a-start", "a-end"],
    )


def test_exclusive_lock_never_rewrites_the_lock_file_after_first_creation(tmp_path):
    # The real Windows bug this exact mechanism was already found and
    # fixed for once (sarva.config._exclusive_lock, SessionStore.locked):
    # a marker byte rewritten on every acquisition conflicts with
    # Windows' own mandatory msvcrt.locking(). mtime (not inode/content,
    # which stay identical either way since both versions write the same
    # byte) is what actually distinguishes "wrote again" from "never
    # touched after the first creation."
    lock_path = tmp_path / "test.lock"

    with exclusive_lock(lock_path):
        pass
    first_mtime = lock_path.stat().st_mtime_ns

    time.sleep(0.05)
    with exclusive_lock(lock_path):
        pass
    with exclusive_lock(lock_path):
        pass

    assert lock_path.stat().st_mtime_ns == first_mtime
