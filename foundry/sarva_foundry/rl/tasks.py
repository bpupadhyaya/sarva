"""sarva_foundry.rl.tasks — a small, real, hand-verified bundled coding
task set. Same honesty discipline as `sarva.eval.benchmarks.ARITHMETIC`:
real tasks with real, hand-checked correct reference solutions, not a
claim to HumanEval-scale coverage or difficulty.
"""

from __future__ import annotations

from sarva_foundry.rl.environment import CodingTask

CODING_TASKS = [
    CodingTask(
        task_id="add",
        prompt="Write a function `add(a, b)` that returns the sum of a and b.",
        test_code=(
            "assert add(2, 3) == 5\n"
            "assert add(-1, 1) == 0\n"
            "assert add(0, 0) == 0\n"
            "print('all tests passed')"
        ),
    ),
    CodingTask(
        task_id="is_palindrome",
        prompt=(
            "Write a function `is_palindrome(s)` that returns True if the "
            "string s reads the same forwards and backwards, False otherwise."
        ),
        test_code=(
            "assert is_palindrome('racecar') is True\n"
            "assert is_palindrome('hello') is False\n"
            "assert is_palindrome('') is True\n"
            "print('all tests passed')"
        ),
    ),
    CodingTask(
        task_id="fibonacci",
        prompt=(
            "Write a function `fibonacci(n)` that returns the nth Fibonacci "
            "number (0-indexed: fibonacci(0) == 0, fibonacci(1) == 1)."
        ),
        test_code=(
            "assert fibonacci(0) == 0\n"
            "assert fibonacci(1) == 1\n"
            "assert fibonacci(5) == 5\n"
            "assert fibonacci(10) == 55\n"
            "print('all tests passed')"
        ),
    ),
]
