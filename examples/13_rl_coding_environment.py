"""Example 13 — The RL environment harness: sandboxed coding tasks with
automatic, verifiable rewards.

Spec §3.6e: "the RL environment harness (sandboxed coding tasks with
automatic verification)." This is the harness a future agentic-RL
training loop would consume, not the training loop itself (a real
policy-gradient algorithm like PPO/GRPO, plus a model-in-the-loop
training run, is real, deferred work this project doesn't have the
compute for yet).

Runs the bundled coding tasks against three different "policies" —
stand-ins for what would normally be a model's sampled completions — to
show the reward signal is genuinely earned, not hardcoded: a correct
solution, a plausible-but-wrong solution, and an infinite loop, each
scored honestly.

Run: uv run python examples/13_rl_coding_environment.py
"""

from __future__ import annotations

from sarva_foundry.rl import CODING_TASKS, evaluate_submission

# Three "policies": what a real agentic-RL rollout would produce is a
# model's sampled code completion for a task's prompt; here, three
# fixed submissions stand in, chosen to exercise genuinely different
# reward outcomes.
CORRECT_SOLUTIONS = {
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

WRONG_SOLUTION = "def add(a, b):\n    return a - b"  # plausible, but fails the real tests
INFINITE_LOOP_SOLUTION = "def add(a, b):\n    while True:\n        pass"


def main() -> None:
    print(f"{len(CODING_TASKS)} bundled coding tasks, each with real, hand-verified tests.\n")

    print("Policy A: correct solutions for every task")
    for task in CODING_TASKS:
        result = evaluate_submission(task, CORRECT_SOLUTIONS[task.task_id])
        print(f"  {task.task_id:15s} reward={result.reward}  passed={result.passed}")

    print("\nPolicy B: a plausible-but-wrong solution for 'add'")
    result = evaluate_submission(CODING_TASKS[0], WRONG_SOLUTION)
    print(f"  add             reward={result.reward}  passed={result.passed}")
    print(f"  real test failure captured: {result.stderr.strip().splitlines()[-1]}")

    print("\nPolicy C: an infinite loop for 'add' (a hard timeout catches it)")
    result = evaluate_submission(CODING_TASKS[0], INFINITE_LOOP_SOLUTION, timeout=2.0)
    print(f"  add             reward={result.reward}  passed={result.passed}  ", end="")
    print(f"timed_out={result.timed_out}")

    print(
        "\nEvery reward above came from actually running the submission against "
        "real tests in a real subprocess -- no reward was assumed or hardcoded. "
        "This is the verifiable-reward signal a real RL training loop would use; "
        "the training loop itself (sampling from a policy, computing a policy-"
        "gradient update from these rewards) is real, deferred work."
    )


if __name__ == "__main__":
    main()
