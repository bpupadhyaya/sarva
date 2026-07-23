"""sarva_foundry.rl — the sandboxed coding-task environment harness with
automatic verification (spec §3.6e). See `environment.py`'s module
docstring for exactly what "sandboxed" does and doesn't mean here.
"""

from sarva_foundry.rl.environment import CodingTask, TaskResult, evaluate_submission
from sarva_foundry.rl.tasks import CODING_TASKS

__all__ = ["CODING_TASKS", "CodingTask", "TaskResult", "evaluate_submission"]
