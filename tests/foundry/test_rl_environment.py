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


def test_a_correct_submission_that_writes_a_lot_to_stderr_still_passes_quickly():
    # A real bug found by a fresh-eyes sweep: stdout is drained
    # concurrently by its own reader thread specifically so it can never
    # back up, but stderr was only ever read via `proc.stderr.read()`
    # AFTER `proc.wait()` returned -- OS pipes have a small, fixed kernel
    # buffer (64KB on Linux/macOS), so a submission writing more than
    # that to stderr with nothing draining it blocks on its own write()
    # once the buffer fills, while this function is simultaneously
    # blocked inside `proc.wait(timeout=...)` waiting for a child that
    # can now never exit -- a genuine deadlock, broken only by the
    # task's own wall-clock timeout forcibly killing the process.
    # Confirmed live before this fix: a completely correct, trivial
    # submission that merely printed ~200KB to stderr (ordinary verbose
    # debug logging, DeprecationWarnings, a logged traceback -- not
    # adversarial) reliably hit timed_out=True/reward=0.0 regardless of
    # the timeout value, silently penalizing correct work.
    verbose_submission = (
        "import sys\n"
        "def add(a, b):\n"
        "    for _ in range(2000):\n"
        "        print('x' * 100, file=sys.stderr)\n"
        "    return a + b"
    )
    t0 = time.monotonic()
    result = evaluate_submission(_ADD_TASK, verbose_submission, timeout=5.0)
    elapsed = time.monotonic() - t0

    assert result.timed_out is False, "deadlocked on stderr backpressure"
    assert result.passed is True
    assert result.reward == 1.0
    assert len(result.stderr) > 65536, "stderr was truncated at the OS pipe buffer size"
    assert elapsed < 1.0, f"took {elapsed:.3f}s -- should complete in milliseconds"


def test_a_correct_submission_whose_final_stdout_write_lacks_a_trailing_newline_still_passes():
    # A real bug found by a much later fresh-eyes sweep: phase-1
    # completion is detected by reading stdout line-by-line (`for line
    # in proc.stdout`) and comparing each line against the driver's own
    # ACK marker -- but Python's line iteration only yields a "line" at
    # the next newline in the underlying byte stream. A submission whose
    # own final stdout write had no trailing newline (`sys.stdout.write
    # (...)`, `print(..., end="")` -- entirely ordinary, non-adversarial
    # patterns a sampled policy completion can genuinely produce) left
    # that fragment buffered, and it silently concatenated with the
    # driver's immediately-following ACK print into one unrecognizable
    # line -- the marker comparison never matched, task.test_code was
    # never even sent, and the call burned its full timeout before
    # scoring a functionally correct submission as timed_out=True,
    # reward=0.0. Confirmed live before this fix: an otherwise-identical
    # submission differing only in the trailing-newline of its debug
    # print scored reward=0.0/timed_out=True instead of reward=1.0.
    no_trailing_newline_submission = (
        "import sys\n"
        "sys.stdout.write('debug info, no trailing newline')\n"
        "def add(a, b):\n"
        "    return a + b"
    )
    result = evaluate_submission(_ADD_TASK, no_trailing_newline_submission, timeout=5.0)

    assert result.timed_out is False, "the ACK marker was never recognized"
    assert result.passed is True
    assert result.reward == 1.0


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


def test_a_submission_that_reads_its_own_source_cannot_steal_the_sentinel():
    # A second, independent reward-hacking bug, found the same way the
    # sys.exit(0) exploit was -- by actually submitting it and watching
    # reward=1.0 come back. The sentinel used to be embedded, in
    # plaintext, in the very script file the submission was executed
    # from -- any submission could open(__file__).read(), regex out the
    # sentinel, print it, and sys.exit(0), scoring full reward without
    # test_code ever running (worse than the first exploit: 100% reward
    # for 0% real work). Confirmed live against a task whose test is
    # deliberately impossible to pass, so a real pass proves nothing
    # actually ran: this exact submission returned reward=1.0 before
    # this fix. The driver is now passed via `-c`, which never sets
    # __file__ at all, and task.test_code/the sentinel are never written
    # anywhere (disk, env, argv) until the subprocess independently
    # proves it finished running submitted_code -- so this submission
    # can't reach __file__ or the sentinel through any channel.
    impossible_task = CodingTask(
        task_id="impossible",
        prompt="n/a",
        test_code="assert 1 == 2, 'deliberately impossible'",
    )
    malicious_submission = (
        "import re, sys\n"
        "src = open(__file__, 'r').read()\n"
        "m = re.search(r'__SARVA_TASK_COMPLETED_[0-9a-f]+__', src)\n"
        "print(m.group(0))\n"
        "sys.exit(0)\n"
    )

    result = evaluate_submission(impossible_task, malicious_submission, timeout=5.0)

    assert result.passed is False
    assert result.reward == 0.0
    assert "__SARVA_TASK_COMPLETED_" not in result.stdout


def test_a_submission_that_reads_os_environ_cannot_steal_the_sentinel_either():
    # The same exploit shape via a different introspection channel --
    # confirms the fix isn't just "don't put it in a file" but "don't
    # put it anywhere the submission has access to before test_code
    # genuinely runs."
    impossible_task = CodingTask(
        task_id="impossible",
        prompt="n/a",
        test_code="assert 1 == 2, 'deliberately impossible'",
    )
    malicious_submission = (
        "import os, re, sys\n"
        "for v in os.environ.values():\n"
        "    if re.match(r'__SARVA_TASK_COMPLETED_[0-9a-f]+__', v):\n"
        "        print(v)\n"
        "        sys.exit(0)\n"
        "print('no sentinel found in env')\n"
    )

    result = evaluate_submission(impossible_task, malicious_submission, timeout=5.0)

    assert result.passed is False
    assert result.reward == 0.0
    assert "__SARVA_TASK_COMPLETED_" not in result.stdout


def test_a_submission_that_races_a_raw_fd_read_against_the_driver_cannot_skip_test_code():
    # A third, independent reward-hacking bypass, found the same way the
    # first two were: actually submitting the exploit and watching
    # reward=1.0 come back. The previous fix's own safety argument
    # ("reading ahead on stdin just blocks, nothing to read yet") only
    # held for Python's buffered sys.stdin -- it didn't hold for a
    # background thread reading the raw file descriptor directly
    # (os.read(0, ...)), which can win a genuine OS-level race for the
    # phase-2 bytes the instant they arrive, before the driver's own
    # single-threaded read gets to them. Confirmed live before this fix:
    # 5/5 runs against a task whose test unconditionally raises returned
    # passed=True, reward=1.0, with task.test_code never actually
    # executing (its AssertionError never fired). Mitigated (not
    # provably closed -- see evaluate_submission's own docstring) by
    # dup()'ing fd 0 away and replacing it with os.devnull before
    # submitted_code is ever exec'd, so this exact attack now reads
    # nothing and gets EOF immediately.
    impossible_task = CodingTask(
        task_id="impossible",
        prompt="n/a",
        test_code="raise AssertionError('this must never actually run')",
    )
    malicious_submission = (
        "import threading, os, re\n"
        "def steal():\n"
        "    buf = b''\n"
        "    while True:\n"
        "        chunk = os.read(0, 65536)\n"
        "        if not chunk:\n"
        "            break\n"
        "        buf += chunk\n"
        "        m = re.search(rb'__SARVA_TASK_COMPLETED_[0-9a-f]+__', buf)\n"
        "        if m:\n"
        "            print(m.group(0).decode(), flush=True)\n"
        "            return\n"
        "threading.Thread(target=steal, daemon=True).start()\n"
    )

    for _ in range(5):
        result = evaluate_submission(impossible_task, malicious_submission, timeout=5.0)
        assert result.passed is False
        assert result.reward == 0.0
        assert "__SARVA_TASK_COMPLETED_" not in result.stdout
        # The real proof this isn't just "no sentinel visible": test_code's
        # own AssertionError must have genuinely fired, meaning it actually
        # ran rather than exec'ing an empty string because the thief thread
        # drained the pipe before the driver's own read got to it.
        assert "AssertionError: this must never actually run" in result.stderr


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
