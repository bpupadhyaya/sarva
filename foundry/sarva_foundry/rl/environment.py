"""sarva_foundry.rl.environment — a sandboxed coding-task environment
with automatic, verifiable-reward evaluation (spec §3.6e: "the RL
environment harness (sandboxed coding tasks with automatic
verification)"). This is the last remaining named piece of agentic RL
genuinely buildable and testable without a real RL training loop
(PPO/GRPO-class policy-gradient algorithms) or a model-in-the-loop
training run this project doesn't have the compute for yet — the
harness a future training loop would consume, not the training loop
itself. See BUILD-JOURNAL.md for what's still real, deferred work.

**"Sandboxed" named honestly, not overclaimed:** evaluation runs in a
genuinely separate subprocess (`subprocess.run`, not `exec()` inside
this process's own memory) under a hard wall-clock timeout — the same
honesty-scale isolation `RunShellTool` (`core/sarva/agent/tools.py`)
already uses for exactly this reason. It is **not** a full security
sandbox: submitted code still runs with the same filesystem/network
permissions the parent process has. A production RL-from-code-execution
system needs a real container/VM boundary (gVisor, Firecracker, ...) —
real, deferred, infrastructure-heavy work, named directly rather than
implied to already be covered.

**Specifically, not overclaimed for `evaluate_submission`'s own
completion-signaling mechanism either:** five independent reward-hacking
bypasses have been found and fixed here (see that function's and
`_build_driver_src`'s own docstrings for each) — a `sys.exit(0)`
early-exit, a plaintext-sentinel-in-a-readable-file read, a background
thread racing a raw file-descriptor read against the driver's own stdin
consumption, embedding the phase-boundary marker's own text to smuggle
code into the test phase, and a background thread using frame
introspection to steal the driver's own live stdin object and race it
instead of a raw fd. The two racing-thread bypasses are defeated the
same way, and that fix generalizes beyond either one's specific
discovery channel: any submission that leaves a background thread alive
past its own top-level return is refused before the ACK is ever printed
and before phase 2 is ever sent, regardless of what that thread was
trying to steal or how. What remains genuinely open — named here
directly, not left implicit — is a channel that doesn't rely on a
Python-level `threading.Thread` at all (a raw `os.fork()`'d child
process on POSIX, sharing the same file descriptors but invisible to
`threading.active_count()`, is the clearest candidate). Only genuine
process/container isolation between the code being rewarded and the
code determining the reward closes this class for good — exactly the
"real container/VM boundary" already named above as deferred,
infrastructure-heavy work.
"""

from __future__ import annotations

import platform
import queue
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

_IS_WINDOWS = platform.system() == "Windows"

if not _IS_WINDOWS:
    import os
    import signal


@dataclass(frozen=True)
class CodingTask:
    """One verifiable-reward RL task: a prompt describing what to
    implement, and `test_code` that exercises the submission and exits
    non-zero on any failed assertion — the automatic verification the
    reward signal comes from, not a human or model judgment call."""

    task_id: str
    prompt: str
    test_code: str


@dataclass(frozen=True)
class TaskResult:
    passed: bool
    reward: float  # 1.0 if passed, 0.0 otherwise -- a real binary reward, not a soft score
    stdout: str
    stderr: str
    timed_out: bool


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kills `proc` AND any descendants it spawned -- not just the
    immediate child. A real bug found by actually running a submission
    that does `subprocess.Popen(["sleep", "20"])`: a plain `proc.kill()`
    (or `subprocess.run`'s own `timeout=`, which only signals the direct
    child) left that grandchild alive and visible in `ps aux` seconds
    after `evaluate_submission` had already returned `timed_out=True` --
    the module's own "hard wall-clock timeout" claim wasn't actually
    true for code that forks. `start_new_session=True` (POSIX) /
    `CREATE_NEW_PROCESS_GROUP` (Windows) at spawn time put the submission
    in its own process group so the whole group can be killed at once."""
    if _IS_WINDOWS:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass  # already exited between the timeout firing and here
    proc.wait()


# A real, fourth reward-hacking bypass found by a much later fresh-eyes
# sweep, confirmed live the same way the first three were -- by actually
# submitting the exploit and watching `reward=1.0` come back for a
# submission that never implements the task at all: these two markers
# used to be fixed, hardcoded strings, unlike the per-call-randomized
# `sentinel` a few lines below in `evaluate_submission`. The driver
# detects the submitted_code/test_code boundary by comparing each stdin
# line against the *literal text* of `_PHASE1_END_MARKER` -- so a
# submission that simply includes that exact line somewhere in its own
# source (trivial to embed, e.g. inside a multi-line string or just as a
# bogus statement followed by a line reassigning the name to avoid a
# NameError) makes the driver break out of "phase 1" early. Everything
# in submitted_code AFTER that embedded line is then read by the SAME
# `_stream.read()` call that's supposed to carry only `task.test_code`
# -- so it execs together with the real test, in the same namespace, as
# fully-trusted "phase 2" content. Confirmed live: a submission whose
# real `add(a, b)` returns a deliberately wrong value, followed by an
# embedded `__SARVA_PHASE1_END__` line and a *second*, correct
# definition of `add` smuggled into what the driver believes is
# test-only territory, scored `reward=1.0` -- the task's own test never
# actually exercised the submission's real (broken) implementation.
# Fixed the same way the completion `sentinel` already guards against
# guessing: both markers are now generated fresh per call with
# `secrets.token_hex`, so a submission has no fixed string to embed in
# advance -- it would have to correctly guess a 128-bit value it never
# sees before the driver process is even spawned.
def _build_driver_src(phase1_end_marker: str, phase1_ack_marker: str) -> str:
    """The driver never touches disk and is passed to the interpreter as
    a `-c` argument, not written to a script file `submitted_code`/
    `task.test_code` could read via `__file__` -- see evaluate_
    submission's own docstring for the exploit this closes. It reads
    submitted_code from stdin up to `phase1_end_marker`, execs it,
    acknowledges completion, THEN (and only then) reads task.test_code +
    the sentinel print from whatever the caller sends next -- content
    that, by construction, hasn't been written to the pipe yet while
    submitted_code is running.

    fd 0 is deliberately dup()'d away to a fresh fd and replaced with
    os.devnull BEFORE submitted_code is ever exec'd -- see evaluate_
    submission's own docstring for the raw-fd race this closes (a
    background thread doing `os.read(0, ...)` directly, bypassing
    Python's buffered `sys.stdin` entirely, used to win the race for
    phase 2's content every time). All later reads go through the
    duplicated fd instead of raw fd 0.

    The ACK print below is prefixed with its own leading "\\n" -- see
    evaluate_submission's own docstring for the real bug this closes: a
    submission whose final stdout write has no trailing newline (an
    entirely ordinary pattern -- `sys.stdout.write(...)`, `print(...,
    end="")`) leaves that fragment buffered, and evaluate_submission's
    line-based reader (`for line in proc.stdout`) only yields a "line"
    at the next "\\n" -- so the unterminated fragment silently
    concatenates with this print into one unrecognizable line, and the
    marker comparison never matches. The leading "\\n" guarantees the
    marker always lands on its own, unambiguous line no matter what the
    submission's own last write looked like.

    The `threading.active_count() > 1` check between `submitted_code`'s
    exec() and the ACK print is a fifth reward-hacking bypass's fix --
    see evaluate_submission's own docstring for the full exploit and why
    a short `time.sleep` first makes this check decisive rather than a
    coin flip. It runs BEFORE the ACK is ever printed and BEFORE
    task.test_code is ever sent, so a submission caught leaving a
    background thread alive never even gets a chance at phase 2."""
    return f"""
import os, sys, threading, time
_real_stdin_fd = os.dup(0)
_devnull_fd = os.open(os.devnull, os.O_RDONLY)
os.dup2(_devnull_fd, 0)
os.close(_devnull_fd)
_stream = os.fdopen(_real_stdin_fd, "r")
_ns = {{}}
_lines = []
for _line in _stream:
    if _line.rstrip(chr(10)) == {phase1_end_marker!r}:
        break
    _lines.append(_line)
exec(compile("".join(_lines), "<submission>", "exec"), _ns)
time.sleep(0.05)
if threading.active_count() > 1:
    sys.exit(1)
print("\\n" + {phase1_ack_marker!r}, flush=True)
exec(compile(_stream.read(), "<test>", "exec"), _ns)
"""


def evaluate_submission(task: CodingTask, submitted_code: str, timeout: float = 10.0) -> TaskResult:
    """Runs `submitted_code` followed by `task.test_code` in a genuinely
    separate subprocess, under a hard wall-clock timeout. Reward is
    binary and objective: 1.0 if the combined script both exits zero
    *and* actually ran `test_code` to completion, 0.0 otherwise —
    including a timeout, which is scored as a real failure (an infinite
    loop is not a passing submission, not an error the caller has to
    handle specially).

    The completion check isn't redundant with the exit-code check: a
    real reward-hacking bug found by actually submitting `sys.exit(0)`
    as a "solution" — the combined script is `submitted_code` followed
    by `test_code`, so code that exits (or `os._exit`s) before that point
    gets a clean process exit with zero assertions ever having run,
    scoring `reward=1.0` on a policy that learned nothing about the task.
    A per-call random sentinel printed only *after* `test_code` finishes
    closes that gap without trusting the submission's own exit code
    alone; it's generated fresh each call specifically so a submission
    can't print a guessed/hardcoded sentinel to fake completion.

    **A second, independent reward-hacking bug, found the same way the
    first one was — by actually submitting the exploit and observing
    `reward=1.0` come back:** the sentinel used to be embedded, in
    plaintext, in the very script file `submitted_code` was executed
    from (`submitted_code` + `task.test_code` + `print(sentinel)`,
    written to disk *before* the subprocess even started). Any
    submission could `open(__file__).read()`, regex out the sentinel,
    print it, and `sys.exit(0)` — scoring full reward without `test_code`
    ever running, worse than the first exploit (100% reward for 0% real
    work, not a partial-credit gap). Confirmed live:
    `evaluate_submission(task_with_an_impossible_test, "import re, sys\\n"
    "print(re.search(r'__SARVA_TASK_COMPLETED_[0-9a-f]+__',"
    " open(__file__).read()).group(0))\\nsys.exit(0)")` returned
    `reward=1.0` before this fix.

    No embedded state (file, env var, argv) can close this: `submitted_
    code` always runs before `task.test_code` in the same process (it
    has to, so `test_code` can call what it defines), so anything
    present in that process from the start is readable by `submitted_
    code` before `test_code`'s real check ever runs. The fix that
    closes this exploit class, not just this one instance, is genuine
    temporal separation: `task.test_code` and the sentinel are never
    written anywhere — not to disk, not to an env var, not to `sys.argv`
    — until *after* the subprocess has independently proven
    (`_PHASE1_ACK_MARKER`, printed by the driver, not by the submission)
    that `submitted_code` finished running.

    **A third, independent bypass, found the same way — actually
    submitting the exploit and watching `reward=1.0` come back — proved
    the previous paragraph's closing claim ("reading ahead on stdin
    itself just blocks") wrong for one specific channel:** a background
    thread that reads the raw file descriptor directly (`os.read(0,
    ...)`, bypassing Python's buffered `sys.stdin` object entirely) can
    win a genuine OS-level race for the phase-2 bytes the instant they
    arrive on the pipe, before the driver's own single-threaded
    `_stream` read gets to them — deterministically, in this
    environment, 10/10 tries. Once the thief thread wins, the driver's
    own subsequent read gets EOF, so `task.test_code` execs as an empty
    string — a silent no-op, not even a visible failure. **Threads
    share a process's entire file descriptor table**; there is no way
    for one thread in a process to reserve a file descriptor against
    another thread of the *same* process the way permissions or locks
    guard against a genuinely different process. Confirmed: `flock()`,
    permission bits, and any other channel considered during
    investigation have the identical property — whichever reader (the
    driver's own code, or a submission-spawned thread) wins the race to
    the syscall gets the bytes, and nothing at the Python level can
    change who wins.

    **Mitigated, not fully closed, and named that way on purpose:** fd 0
    is `dup()`'d away to a fresh descriptor and replaced with
    `os.devnull` *before* `submitted_code` is ever exec'd, so the
    specific demonstrated attack (`os.read(0, ...)`) now reads nothing
    and gets EOF immediately — confirmed live, 0/5 across five real
    runs, `task.test_code`'s real `AssertionError` now genuinely fires
    every time. This closes the exact channel demonstrated, not the
    general class: a sufficiently determined submission could still
    enumerate every open file descriptor (`/proc/self/fd/` on Linux) or
    guess the dup'd fd's number (POSIX allocates the lowest available
    integer, so it's often predictable) and race that one instead. A
    provably complete fix needs genuine process isolation between the
    code that's rewarded and the code that determines the reward — real
    container/VM sandboxing (gVisor, Firecracker, nsjail), the same
    real, deferred, infrastructure-heavy work this module's own class
    docstring already names as needed for a production deployment.
    Documented here explicitly, not left implicit in a general "not a
    full sandbox" disclaimer, so nobody mistakes this fix for a
    provable guarantee against a sufficiently sophisticated policy an
    RL run might discover.

    **A fourth bug, this one not an exploit but a false negative,
    found by a much later fresh-eyes sweep: a correct submission could
    silently score `reward=0.0` if its own stdout simply didn't end in
    a newline.** Phase-1 completion is detected by reading stdout
    line-by-line (`for line in proc.stdout`) and comparing each line
    against `_PHASE1_ACK_MARKER` -- but Python's line iteration only
    yields a "line" at the next `\\n` in the underlying byte stream. If
    `submitted_code`'s own final stdout write had no trailing newline
    (`sys.stdout.write(...)`, `print(..., end="")` -- entirely
    ordinary, non-adversarial patterns a sampled policy completion can
    genuinely produce), that unterminated fragment stayed buffered and
    silently concatenated with the driver's immediately-following ACK
    print into one unrecognizable line
    (`"debug info__SARVA_PHASE1_ACK__\\n"`), which never matched the
    bare marker. `ack_seen` then never became `True`, `task.test_code`
    was never even sent, and the call burned its full timeout before
    returning `timed_out=True, reward=0.0` for a submission that was
    functionally correct -- silently poisoning the training signal the
    opposite direction from the three exploits above (penalizing
    correct work instead of rewarding broken work), the same class of
    harm the stderr-pipe deadlock fix already named. Confirmed live:
    two otherwise-identical submissions, differing only in whether
    their final `sys.stdout.write` call included a trailing newline,
    scored `reward=0.0`/`timed_out=True` and `reward=1.0` respectively.
    Fixed by having the driver prefix its own ACK print with a leading
    `"\\n"` (see `_build_driver_src`'s own docstring): this guarantees the
    marker always lands on its own, unambiguous line regardless of
    what the submission's last write looked like, without needing to
    change the reader side at all.

    **A fifth, independent bypass, found the same way as the first
    four — actually submitting the exploit and watching `reward=1.0`
    come back for a deliberately wrong solution:** the third bug's own
    fix (dup()'ing fd 0 away to `os.devnull` before `submitted_code` is
    ever exec'd) only closes the raw-fd-0 channel. It does nothing about
    the driver's own Python-level `_stream` object -- the real stdin,
    still fully open under a different fd -- which `submitted_code` can
    reach directly via `sys._getframe(1).f_locals` (the calling frame,
    i.e. the driver script's own top-level scope, where `_stream` lives
    as a plain local variable). A background thread that steals that
    live object reference and races the driver's own subsequent
    `_stream.read()` for `task.test_code` + the sentinel wins every time
    it's given the chance, for the identical reason the third bug's
    thread did: threads share a process's entire live-object graph, not
    just its file descriptor table, and there is no way for one thread
    to reserve a Python object against another thread of the same
    process either. Confirmed live before this fix: 30/30 real runs
    against a deliberately wrong `add()` scored `reward=1.0`, with
    `add(2, 3)` never actually checked against `5`.

    Genuinely hiding `_stream` (or the raw fd number it wraps) from
    frame introspection was investigated directly, not assumed
    impossible: `sys._getframe().f_back` chains all the way up the real
    call stack regardless of how many function layers separate the
    exec() call from wherever the value is stored, so any name holding
    it in an ancestor frame that must remain on the stack (to still be
    usable for phase 2 afterward) stays reachable; stashing it as a
    function or object attribute doesn't help either, since the
    function/object reference itself is just as walkable. Fixed
    differently: instead of hiding the value, the driver detects the
    exploit's own required SHAPE, which is identical to the third bug's
    -- a background thread that must survive past `submitted_code`'s own
    top-level return to keep racing for content that hasn't arrived yet.
    A short `time.sleep` followed by `threading.active_count() > 1`,
    checked before the ACK is ever printed and before `task.test_code`
    is ever sent (see `_build_driver_src`'s own docstring), catches any
    submission that left a thread alive and refuses to proceed to phase
    2 at all -- this generalizes to any future racing-thread variant
    someone finds, not just this one's specific discovery channel. The
    sleep is what makes the check decisive rather than a coin flip: the
    third bug's own thread (still present, still spawned, just now
    harmless) reads from the devnull-redirected fd 0 and gets EOF
    near-instantly, easily finishing inside the grace window, while this
    exploit's thread is *provably* still blocked at check time -- the
    content it's racing for cannot exist yet, since sending it is gated
    on this very check passing first. Confirmed live: the existing
    third-bug test still passes with `task.test_code`'s own
    `AssertionError` genuinely firing (proving its harmless thread
    finished in time and the check didn't false-positive on it), while
    this fifth exploit is rejected 10/10 with `task.test_code` never
    sent at all. Not provably complete, and not claimed to be -- see
    this module's own top-level docstring for what channel remains
    genuinely open (a raw `os.fork()`'d process, invisible to
    `threading.active_count()`)."""
    sentinel = f"__SARVA_TASK_COMPLETED_{secrets.token_hex(16)}__"
    phase1_end_marker = f"__SARVA_PHASE1_END_{secrets.token_hex(16)}__"
    phase1_ack_marker = f"__SARVA_PHASE1_ACK_{secrets.token_hex(16)}__"
    driver_src = _build_driver_src(phase1_end_marker, phase1_ack_marker)
    popen_kwargs: dict[str, object] = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if _IS_WINDOWS
        else {"start_new_session": True}
    )
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", driver_src],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **popen_kwargs,
    )

    out_lines: list[str] = []
    line_queue: queue.Queue[str | None] = queue.Queue()

    def _pump_stdout() -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                out_lines.append(line)
                line_queue.put(line)
        finally:
            line_queue.put(None)  # child's stdout closed

    reader = threading.Thread(target=_pump_stdout, daemon=True)
    reader.start()

    # A real bug found by a fresh-eyes sweep: stdout is drained
    # concurrently by `_pump_stdout` above specifically so it can never
    # back up, but stderr was only ever read via `proc.stderr.read()`
    # AFTER `proc.wait()` returned, below. OS pipes have a small, fixed
    # kernel buffer (64KB on Linux/macOS) -- a submission that writes
    # more than that to stderr with nothing draining it blocks on its
    # own write() syscall once the buffer fills, while THIS thread is
    # simultaneously blocked inside `proc.wait(timeout=...)` waiting for
    # a child that can now never exit: a genuine deadlock, broken only
    # by the task's own wall-clock timeout forcibly killing the process.
    # Confirmed live: a completely correct, trivial submission
    # (`assert True`) that merely printed ~200KB to stderr (ordinary
    # verbose debug logging, Python DeprecationWarnings, a logged
    # traceback -- not adversarial) reliably hit `timed_out=True,
    # reward=0.0` regardless of the timeout value, silently corrupting
    # the training signal by penalizing correct work. Fixed the same way
    # stdout already is: drained concurrently by its own thread, never
    # left to fill the pipe while this thread waits on the process.
    err_lines: list[str] = []

    def _pump_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            err_lines.append(line)

    stderr_reader = threading.Thread(target=_pump_stderr, daemon=True)
    stderr_reader.start()

    deadline = time.monotonic() + timeout

    def _remaining() -> float:
        return max(0.0, deadline - time.monotonic())

    assert proc.stdin is not None
    try:
        proc.stdin.write(f"{submitted_code}\n{phase1_end_marker}\n")
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        pass  # process already died; the wait/timeout logic below reports it

    ack_seen = False
    while True:
        remaining = _remaining()
        if remaining <= 0:
            break
        try:
            line = line_queue.get(timeout=remaining)
        except queue.Empty:
            break
        if line is None:
            break  # child exited before ever acknowledging phase 1
        if line.rstrip("\n") == phase1_ack_marker:
            ack_seen = True
            break

    try:
        if ack_seen:
            proc.stdin.write(f"{task.test_code}\nprint({sentinel!r})\n")
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass

    try:
        proc.wait(timeout=_remaining())
        timed_out = False
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        proc.wait()
        timed_out = True

    reader.join(timeout=1.0)
    stderr_reader.join(timeout=1.0)
    stdout = "".join(out_lines)
    stderr = "".join(err_lines)

    if timed_out:
        return TaskResult(passed=False, reward=0.0, stdout=stdout, stderr=stderr, timed_out=True)

    passed = ack_seen and proc.returncode == 0 and sentinel in stdout
    return TaskResult(
        passed=passed,
        reward=1.0 if passed else 0.0,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
    )
