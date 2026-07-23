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

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


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


def evaluate_submission(task: CodingTask, submitted_code: str, timeout: float = 10.0) -> TaskResult:
    """Runs `submitted_code` followed by `task.test_code` in a genuinely
    separate subprocess, under a hard wall-clock timeout. Reward is
    binary and objective: 1.0 if the combined script exits zero (every
    assertion in `test_code` held), 0.0 otherwise — including a timeout,
    which is scored as a real failure (an infinite loop is not a passing
    submission, not an error the caller has to handle specially)."""
    combined = f"{submitted_code}\n\n{task.test_code}\n"
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "submission.py"
        script_path.write_text(combined, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            # text=True on the Popen call above means e.stdout/e.stderr
            # are already str (or None, if nothing was captured before
            # the timeout fired) -- no bytes-decoding needed here.
            return TaskResult(
                passed=False,
                reward=0.0,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
                timed_out=True,
            )

    return TaskResult(
        passed=proc.returncode == 0,
        reward=1.0 if proc.returncode == 0 else 0.0,
        stdout=proc.stdout,
        stderr=proc.stderr,
        timed_out=False,
    )
