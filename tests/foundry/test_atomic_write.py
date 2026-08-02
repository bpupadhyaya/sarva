"""Conformance tests for sarva_foundry.atomic_write -- the mirrored copy
of sarva.atomic_write (core and sarva_foundry have no dependency on each
other, so the helper itself is duplicated on purpose; these tests are
duplicated too, rather than assuming the core-side tests cover this
module)."""

from __future__ import annotations

import os as os_module
import threading

import pytest
from sarva_foundry.atomic_write import atomic_write, atomic_write_text


def test_atomic_write_text_writes_the_content(tmp_path):
    path = tmp_path / "tokenizer.json"
    atomic_write_text(path, '{"merges": []}')
    assert path.read_text() == '{"merges": []}'


def test_atomic_write_cleans_up_the_tmp_file_if_write_fn_raises(tmp_path):
    # Mirrors the identical fix in sarva.atomic_write -- see that
    # module's own test for the live-confirmed leak: a write_fn failure
    # (e.g. a disk-full torch.save mid-checkpoint) used to leave a
    # permanently orphaned, potentially multi-GB temp file with nothing
    # anywhere ever cleaning it up.
    path = tmp_path / "model.pt"
    path.write_bytes(b"good weights")

    def failing_write(tmp_path_arg):
        tmp_path_arg.write_bytes(b"x" * 15000)
        raise OSError("simulated disk-full mid-checkpoint")

    with pytest.raises(OSError, match="simulated disk-full"):
        atomic_write(path, failing_write)

    assert path.read_bytes() == b"good weights"
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_atomic_write_does_not_destroy_the_previous_file_if_interrupted_mid_write(
    tmp_path, monkeypatch
):
    path = tmp_path / "tokenizer.json"
    atomic_write_text(path, "first, good content")

    real_replace = os_module.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("simulated crash during os.replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os_module, "replace", flaky_replace)

    with pytest.raises(OSError):
        atomic_write_text(path, "second write, interrupted")

    assert path.read_text() == "first, good content"


def test_atomic_write_generic_write_fn_receives_the_tmp_path(tmp_path):
    # Generic over write_fn on purpose: Trainer.save_checkpoint needs
    # torch.save's own serialization, not a plain bytes/text write.
    path = tmp_path / "model.pt"
    seen = {}

    def write_fn(tmp_path_arg):
        seen["tmp_path"] = tmp_path_arg
        tmp_path_arg.write_bytes(b"weights")

    atomic_write(path, write_fn)
    assert seen["tmp_path"] != path
    assert path.read_bytes() == b"weights"


def test_atomic_write_does_not_raise_when_two_real_threads_race_the_same_path(tmp_path):
    # Mirrors the identical fix in sarva.atomic_write -- a PID-only temp
    # filename collides across every thread of one process (a PID names
    # a process, not a thread), so two threads writing the same path
    # concurrently raced the same temp file. Confirmed live before this
    # fix: deterministically, 10/10 trials raised an uncaught
    # FileNotFoundError. Fixed by including threading.get_ident() too.
    path = tmp_path / "shared.bin"
    errors = []

    def writer(n):
        try:
            atomic_write_text(path, f"content-{n}")
        except OSError as e:
            errors.append(e)

    for _ in range(20):
        errors.clear()
        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
    assert path.read_text().startswith("content-")
