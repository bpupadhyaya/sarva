"""Conformance tests for sarva_foundry.rl.environment — the sandboxed
coding-task harness (spec §3.6e). Every scenario here is exercised
against a REAL subprocess (a real `python -m` style run of real code),
not mocked -- the whole point of this module is that the reward comes
from actually running code and observing whether it passes, so a test
that mocked the subprocess would be testing nothing real."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest
from sarva_foundry.rl.environment import CodingTask, evaluate_submission
from sarva_foundry.rl.tasks import CODING_TASKS

_posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="this test verifies the process-group kill via pgrep, a POSIX tool -- "
    "the Windows taskkill /T path has no equivalent quick check here",
)

_ADD_TASK = CodingTask(
    task_id="add",
    prompt="Write a function `add(a, b)` that returns the sum of a and b.",
    test_code="assert add(2, 3) == 5\nassert add(-1, 1) == 0\nprint('all tests passed')",
)


def test_correct_submission_passes_with_full_reward():
    result = evaluate_submission(_ADD_TASK, "def add(a, b):\n    return a + b")
    assert result.passed is True
    assert result.reward == 1.0
    assert "all tests passed" in result.stdout
    assert result.timed_out is False


def test_incorrect_submission_fails_with_zero_reward():
    result = evaluate_submission(_ADD_TASK, "def add(a, b):\n    return a - b")
    assert result.passed is False
    assert result.reward == 0.0
    assert "AssertionError" in result.stderr


def test_submission_that_raises_at_import_time_fails_cleanly():
    # Real, common failure mode: the submission doesn't even define the
    # required function, or has a syntax error -- must be scored as a
    # real failure, not crash evaluate_submission itself.
    result = evaluate_submission(_ADD_TASK, "def wrong_name(a, b):\n    return a + b")
    assert result.passed is False
    assert result.reward == 0.0
    assert "NameError" in result.stderr


def test_infinite_loop_times_out_and_is_scored_as_a_failure():
    # The defining property of the hard wall-clock timeout: an infinite
    # loop must not hang the test suite, and must be scored as a real
    # failure (reward 0.0), not raise an uncaught exception up to the
    # caller or hang indefinitely.
    infinite_loop_submission = "def add(a, b):\n    while True:\n        pass"
    result = evaluate_submission(_ADD_TASK, infinite_loop_submission, timeout=2.0)
    assert result.passed is False
    assert result.reward == 0.0
    assert result.timed_out is True


def test_a_submission_that_exits_before_test_code_runs_does_not_get_rewarded():
    # A real reward-hacking bug: the combined script is submitted_code
    # followed by task.test_code, so a "solution" that calls sys.exit(0)
    # (or os._exit(0)) before that point gets a clean process exit with
    # zero of test_code's assertions ever having run -- a policy that
    # learned nothing about the task would score reward=1.0 on every task
    # this way. Covers both sys.exit and the lower-level os._exit, since
    # the latter bypasses even Python's own exit machinery.
    for exit_call in ["sys.exit(0)", "os._exit(0)"]:
        module = "sys" if "sys." in exit_call else "os"
        result = evaluate_submission(_ADD_TASK, f"import {module}\n{exit_call}", timeout=5.0)
        assert result.passed is False, f"{exit_call} was wrongly rewarded"
        assert result.reward == 0.0


@_posix_only
def test_timeout_kills_grandchild_processes_the_submission_spawned_not_just_the_direct_child():
    # A real bug found by actually running a submission that forks: the
    # module's own "hard wall-clock timeout" claim wasn't true for code
    # that spawns its own subprocess -- only the direct child (the
    # submission's own `python submission.py` process) was killed on
    # timeout, leaving a grandchild alive and unattended. Proven directly
    # against the real OS process table, not just against the returned
    # TaskResult.
    submission = "import subprocess\nsubprocess.Popen(['sleep', '20'])\nwhile True:\n    pass\n"
    result = evaluate_submission(_ADD_TASK, submission, timeout=1.5)
    assert result.timed_out is True

    time.sleep(2)
    survivors = subprocess.run(
        ["pgrep", "-f", "sleep 20"], capture_output=True, text=True
    ).stdout.strip()
    assert survivors == "", f"grandchild process(es) survived the timeout: {survivors!r}"


def test_submission_runs_in_a_genuinely_separate_process():
    # Proves this isn't exec() in the calling process: a submission that
    # mutates its own process-local state (e.g. sys.modules, globals)
    # can't leak anything back into the test process, and conversely a
    # variable defined here isn't magically visible inside the
    # subprocess -- the submission has to be fully self-contained.
    leaking_submission = (
        "import os\nos.environ['SARVA_RL_TEST_LEAK'] = 'yes'\ndef add(a, b):\n    return a + b"
    )
    import os

    os.environ.pop("SARVA_RL_TEST_LEAK", None)
    result = evaluate_submission(_ADD_TASK, leaking_submission)
    assert result.passed is True
    assert "SARVA_RL_TEST_LEAK" not in os.environ


def test_bundled_coding_tasks_have_unique_ids_and_real_reference_solutions():
    assert len({t.task_id for t in CODING_TASKS}) == len(CODING_TASKS)

    reference_solutions = {
        "add": "def add(a, b):\n    return a + b",
        "is_palindrome": "def is_palindrome(s):\n    return s == s[::-1]",
        "fibonacci": (
            "def fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a"
        ),
    }
    for task in CODING_TASKS:
        assert task.task_id in reference_solutions, f"no reference solution for {task.task_id}"
        result = evaluate_submission(task, reference_solutions[task.task_id])
        assert result.passed is True, f"{task.task_id} reference solution failed: {result.stderr}"


def test_bundled_coding_tasks_reject_a_deliberately_wrong_solution():
    # The reference solutions passing isn't enough on its own to prove
    # the tests are discriminating -- confirm each task's test_code
    # actually rejects a plausible-but-wrong solution too.
    wrong_solutions = {
        "add": "def add(a, b):\n    return a - b",
        "is_palindrome": "def is_palindrome(s):\n    return True",
        "fibonacci": "def fibonacci(n):\n    return n",
    }
    for task in CODING_TASKS:
        result = evaluate_submission(task, wrong_solutions[task.task_id])
        assert result.passed is False, f"{task.task_id} wrongly accepted a bad solution"
