"""Conformance tests for sarva.atomic_write -- the shared helper that
replaced four independent, hand-rolled copies of the same
temp-file-then-os.replace() fix (sarva.config, sarva.memory.session,
WriteFileTool, sarva.distill.save_jsonl). The bar here: prove the one
property every call site actually depends on -- an interrupted write
never destroys whatever was already on disk -- directly against the
shared helper, not just indirectly through each caller's own tests."""

from __future__ import annotations

import os as os_module
import threading

import pytest
from sarva.atomic_write import atomic_write, atomic_write_bytes, atomic_write_text


def test_atomic_write_text_writes_the_content(tmp_path):
    path = tmp_path / "greeting.txt"
    atomic_write_text(path, "hello")
    assert path.read_text() == "hello"


def test_atomic_write_bytes_uses_the_requested_mode(tmp_path):
    import stat

    path = tmp_path / "secret.bin"
    atomic_write_bytes(path, b"shh", mode=0o600)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_atomic_write_leaves_no_tmp_file_behind_on_success(tmp_path):
    path = tmp_path / "data.txt"
    atomic_write_text(path, "v1")
    atomic_write_text(path, "v2")
    assert path.read_text() == "v2"
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_atomic_write_does_not_destroy_the_previous_file_if_interrupted_mid_write(
    tmp_path, monkeypatch
):
    # The exact bug this helper exists to close, confirmed live across
    # every real call site before this fix: a crash between the target
    # file being truncated and the new content landing destroys whatever
    # was there before. Simulated here by making os.replace() -- the
    # final, atomic commit step -- raise; the real file must still hold
    # the first, complete write's content afterward.
    path = tmp_path / "data.txt"
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


def test_atomic_write_generic_write_fn_is_called_with_the_tmp_path_not_the_real_path(tmp_path):
    # atomic_write is deliberately generic over how content is produced
    # (torch.save's own serialization, not just plain bytes/text) --
    # confirm write_fn genuinely receives a sibling temp path, not the
    # real target, so a caller that inspects/renames inside write_fn
    # can't accidentally clobber the real file before the rename.
    path = tmp_path / "model.bin"
    seen = {}

    def write_fn(tmp_path_arg):
        seen["tmp_path"] = tmp_path_arg
        seen["existed_at_call_time"] = path.exists()
        tmp_path_arg.write_bytes(b"weights")

    atomic_write(path, write_fn)
    assert seen["tmp_path"] != path
    assert seen["tmp_path"].parent == path.parent
    assert seen["existed_at_call_time"] is False
    assert path.read_bytes() == b"weights"


def test_atomic_write_does_not_raise_when_two_real_threads_race_the_same_path(tmp_path):
    # A real bug found by actually racing two real threading.Thread
    # instances against this exact function: the temp filename used to
    # be keyed only on os.getpid(), which is identical across every
    # thread of one process (a PID names a process, not a thread) -- two
    # threads calling atomic_write on the SAME path concurrently then
    # raced the SAME temp file, and whichever thread's os.replace() ran
    # second found the temp path already renamed away by the winner,
    # raising an uncaught FileNotFoundError. Confirmed live before this
    # fix: deterministically, 10/10 trials. Fixed by including
    # threading.get_ident() (unique among the threads alive at any one
    # moment, unlike a PID) in the temp filename too.
    path = tmp_path / "shared.bin"
    errors = []

    def writer(n):
        try:
            atomic_write_bytes(path, f"content-{n}".encode())
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
    # Whichever write landed last, the file must hold real, complete
    # content -- never garbled/partial from a lost race.
    assert path.read_text().startswith("content-")
