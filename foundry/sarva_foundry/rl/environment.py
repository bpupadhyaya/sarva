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
"""

from __future__ import annotations

import platform
import secrets
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
    can't print a guessed/hardcoded sentinel to fake completion."""
    sentinel = f"__SARVA_TASK_COMPLETED_{secrets.token_hex(16)}__"
    combined = f"{submitted_code}\n\n{task.test_code}\nprint({sentinel!r})\n"
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "submission.py"
        script_path.write_text(combined, encoding="utf-8")
        popen_kwargs: dict[str, object] = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if _IS_WINDOWS
            else {"start_new_session": True}
        )
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **popen_kwargs,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            # Safe to call again after a kill: communicate() just drains
            # whatever the now-dead process already wrote and reaps it.
            stdout, stderr = proc.communicate()
            return TaskResult(
                passed=False,
                reward=0.0,
                stdout=stdout or "",
                stderr=stderr or "",
                timed_out=True,
            )

    passed = proc.returncode == 0 and sentinel in stdout
    return TaskResult(
        passed=passed,
        reward=1.0 if passed else 0.0,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
    )
