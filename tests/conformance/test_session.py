"""Conformance tests for sarva.memory.session — the file-based session store."""

from __future__ import annotations

import stat
import sys

import pytest
from sarva.memory.session import SessionStore
from sarva.multimodal.content import ImageBlock, Message, TextBlock

_posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.chmod's real per-user isolation is POSIX-only -- see sarva.config's docstring",
)


@pytest.fixture
def store(tmp_path):
    return SessionStore(root=tmp_path)


def test_load_missing_session_returns_empty(store):
    assert store.load("does-not-exist") == []


def test_save_then_load_round_trips(store):
    messages = [
        Message(role="user", content=[TextBlock(text="hi")]),
        Message(role="assistant", content=[TextBlock(text="hello there")]),
    ]
    store.save("greeting", messages)
    assert store.load("greeting") == messages


def test_round_trip_preserves_binary_content(store):
    messages = [
        Message(
            role="user",
            content=[
                TextBlock(text="what's this?"),
                ImageBlock(media_type="image/png", data=b"\x89PNG\r\n\x1a\n"),
            ],
        )
    ]
    store.save("with-image", messages)
    restored = store.load("with-image")
    assert restored == messages


@_posix_only
def test_save_writes_the_session_file_with_owner_only_permissions(store, tmp_path):
    # Real gap this pins, the same class already fixed in sarva.config:
    # a saved session can hold real tool-use output (file contents,
    # shell command output, anything the user typed) -- at least as
    # sensitive as an API key, and was left world-readable (0644 on
    # this machine's real umask) until this fix.
    store.save("greeting", [Message(role="user", content=[TextBlock(text="hi")])])

    path = tmp_path / "greeting.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@_posix_only
def test_sessions_directory_itself_is_owner_only(store, tmp_path):
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700


@_posix_only
def test_save_tightens_permissions_on_a_file_that_already_existed_insecurely(store, tmp_path):
    path = tmp_path / "greeting.json"
    path.write_bytes(b"[]")
    path.chmod(0o644)
    assert stat.S_IMODE(path.stat().st_mode) == 0o644  # sanity: the insecure state is real

    store.save("greeting", [Message(role="user", content=[TextBlock(text="hi")])])

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_save_does_not_destroy_the_previous_file_if_interrupted_mid_write(
    store, tmp_path, monkeypatch
):
    # A real bug found by actually simulating an interrupted write: the
    # previous implementation opened the real session file directly
    # with O_TRUNC, truncating it to 0 bytes immediately -- before a
    # single byte of new content was written. A crash (OOM-kill,
    # SIGKILL, power loss) between that open() and the write completing
    # destroyed a previously-good, real conversation history. Simulated
    # here by making os.replace() (the final, atomic commit step) raise
    # partway through a second save -- the real session file must still
    # hold the first, complete save's content afterward, not be left
    # empty or partially written.
    import os as os_module

    original = [Message(role="user", content=[TextBlock(text="the real, important history")])]
    store.save("s1", original)
    path = tmp_path / "s1.json"
    assert path.stat().st_size > 0

    def crash_before_replace(*args, **kwargs):
        raise SystemExit("simulated crash before the atomic rename")

    monkeypatch.setattr(os_module, "replace", crash_before_replace)
    with pytest.raises(SystemExit):
        store.save(
            "s1", [Message(role="user", content=[TextBlock(text="new content, never commits")])]
        )
    monkeypatch.undo()

    assert store.load("s1") == original


def test_clear_removes_the_session(store):
    store.save("temp", [Message(role="user", content=[TextBlock(text="x")])])
    assert store.load("temp") != []
    store.clear("temp")
    assert store.load("temp") == []


def test_clear_missing_session_does_not_raise(store):
    store.clear("never-existed")  # must not raise


def test_list_sessions(store):
    assert store.list_sessions() == []
    store.save("alpha", [])
    store.save("beta", [Message(role="user", content=[TextBlock(text="hi")])])
    assert store.list_sessions() == ["alpha", "beta"]


def test_session_name_traversal_is_rejected(store):
    with pytest.raises(ValueError, match="invalid session name"):
        store.load("../../etc/passwd")


def test_session_name_with_invalid_characters_is_rejected(store):
    # Reject rather than silently sanitize — silent stripping risks two
    # distinct names colliding onto the same file (e.g. "my session" and
    # "mysession" both stripping to the same thing).
    with pytest.raises(ValueError, match="invalid session name"):
        store.save("my session!", [])


def test_session_name_past_the_filesystem_length_limit_is_rejected(store):
    # A real bug found by a fresh-eyes sweep: _sanitize()'s character
    # check said nothing about length, so a session name past the
    # filesystem's max filename length reached os.open() (in `locked`'s
    # `_acquire`) and raised a raw OSError (ENAMETOOLONG) -- a
    # completely different exception type than the ValueError every real
    # call site (cli.py, server/app.py) catches specifically because
    # that's the only exception type _sanitize() was ever documented to
    # raise. Confirmed live before this fix: POST /chat with a
    # 300-character session name crashed with a raw 500. `save()` here
    # (not `locked()`) proves the fix at the same layer every OTHER
    # invalid-name test in this file already uses.
    with pytest.raises(ValueError, match="session name too long"):
        store.save("a" * 300, [])


async def test_locked_is_a_noop_for_a_session_less_turn(store):
    # A session-less turn has nothing to protect -- locking on a shared
    # "no session" key would needlessly serialize every anonymous
    # request against every other one.
    async with store.locked(None):
        store.save("real-session", [Message(role="user", content=[TextBlock(text="hi")])])
    assert store.load("real-session") != []


async def test_locked_rejects_an_invalid_session_name(store):
    # store.locked() computes the same sanitized path load()/save() do,
    # so an invalid name must fail the same way -- confirmed here since
    # every real caller (CLI, server) now depends on this exact behavior
    # to preserve their own existing clean-failure handling.
    with pytest.raises(ValueError, match="invalid session name"):
        async with store.locked("bad name!"):
            pass


async def test_locked_never_rewrites_the_lock_file_after_first_creation(store):
    # Mirrors the identical fix in sarva.config's own _exclusive_lock --
    # see that test's own docstring for the real Windows bug this closes
    # and why mtime, not inode/content, is the property that actually
    # distinguishes "never touched again" from "rewrote the same byte
    # value" (both versions write the identical single byte, so inode
    # and content alone can't tell them apart).
    import asyncio

    async with store.locked("shared"):
        pass
    lock_path = store._path("shared").with_suffix(".lock")
    first_mtime = lock_path.stat().st_mtime_ns

    for _ in range(5):
        await asyncio.sleep(0.05)
        async with store.locked("shared"):
            pass

    assert lock_path.stat().st_mtime_ns == first_mtime


async def test_locked_serializes_concurrent_asyncio_tasks_on_the_same_session(store):
    # The in-process half of the cross-process fix's own claim: two
    # asyncio tasks racing a load-sleep-save cycle on the SAME session
    # must not lose either one's message once both are wrapped in
    # store.locked(). Mirrors the exact shape sarva.server.app's /chat
    # and /ws/chat handlers now use.
    import asyncio

    store.save("shared", [Message(role="user", content=[TextBlock(text="seed")])])

    async def turn(label: str, delay: float) -> None:
        async with store.locked("shared"):
            history = store.load("shared")
            await asyncio.sleep(delay)
            history = history + [Message(role="user", content=[TextBlock(text=label)])]
            store.save("shared", history)

    await asyncio.gather(turn("turn-A", 0.05), turn("turn-B", 0.02))

    texts = {m.text() for m in store.load("shared")}
    assert texts == {"seed", "turn-A", "turn-B"}


def test_locked_serializes_two_genuine_os_processes_on_the_same_session(tmp_path):
    # A real bug found by actually racing two genuine OS processes (not
    # threads, not asyncio tasks within one process -- the in-process
    # case the test above already covers) against the same session:
    # `sarva chat --session default` running at the same moment someone
    # is chatting into `"default"` on a running `sarva serve` instance,
    # or two independent CLI invocations, is a real, concrete scenario
    # this project's own users can hit. Confirmed live before this fix:
    # two subprocess.Popen workers doing load-sleep-save on the same
    # session file raced every time, 10/10 trials, with the loser's
    # entire turn silently discarded.
    import subprocess
    import sys

    worker = tmp_path / "worker.py"
    worker.write_text(
        "import asyncio, sys\n"
        "from pathlib import Path\n"
        "from sarva.memory.session import SessionStore\n"
        "from sarva.multimodal.content import Message, TextBlock\n"
        "label, delay, root = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])\n"
        "async def main():\n"
        "    store = SessionStore(root=root)\n"
        "    async with store.locked('shared'):\n"
        "        history = store.load('shared')\n"
        "        await asyncio.sleep(delay)\n"
        "        history = history + [Message(role='user', content=[TextBlock(text=label)])]\n"
        "        store.save('shared', history)\n"
        "asyncio.run(main())\n"
    )
    root = tmp_path / "sessions"
    store = SessionStore(root=root)
    store.save("shared", [Message(role="user", content=[TextBlock(text="seed")])])

    for _ in range(5):
        p1 = subprocess.Popen([sys.executable, str(worker), "turn-A", "0.2", str(root)])
        p2 = subprocess.Popen([sys.executable, str(worker), "turn-B", "0.1", str(root)])
        assert p1.wait(timeout=10) == 0
        assert p2.wait(timeout=10) == 0

    saved = store.load("shared")
    user_texts = [m.text() for m in saved if m.role == "user"]
    assert user_texts.count("turn-A") == 5, f"expected 5 turn-A messages, got {user_texts}"
    assert user_texts.count("turn-B") == 5, f"expected 5 turn-B messages, got {user_texts}"
