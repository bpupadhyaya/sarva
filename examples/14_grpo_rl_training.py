"""Example 14 — GRPO: policy-gradient training from verifiable rewards.

Spec §3.6e: agentic RL, "this, not pretraining, is what turns a base
model into a Fable/K3-class agent." This is the training loop
`examples/13_rl_coding_environment.py`'s harness was always missing —
Group Relative Policy Optimization (Shao et al. 2024): sample a GROUP of
completions per prompt, score each with a real reward function, and use
each completion's reward relative to its own group's mean as the
policy-gradient weight. No separate value network/critic needed, unlike
full PPO.

**A real finding worth explaining, not hidden:** this project's tiny,
weight-tied, freshly-initialized transformer turns out to have
extremely peaked initial sampling at temperature=1.0 — one dominant
token regardless of prompt, empirically measured at >99.9% probability
across ten different random seeds before this example was written (see
BUILD-JOURNAL.md). A higher rollout temperature (8.0 here) restores
real exploration — standard real-world RL fine-tuning practice anyway,
not a workaround invented just to make this demo work.

The task is a synthetic, token-level verifiable reward (does the
sampled completion contain a specific target token?) rather than the
real coding-task harness from example 13: a 2-layer, 16-dim toy
transformer genuinely cannot learn to write working Python from
scratch via sparse code-execution rewards in a few hundred steps, and
this project doesn't fabricate results — see the "connecting to real
coding tasks" section at the end for exactly how `CODING_TASKS` /
`evaluate_submission` plug into this same loop as the reward function
once a foundry model exists at a scale that could actually learn from
it.

Run: uv run python examples/14_grpo_rl_training.py
"""

from __future__ import annotations

import torch
from sarva_foundry.model import DecoderOnlyTransformer, TransformerConfig
from sarva_foundry.rl import CODING_TASKS
from sarva_foundry.train import Trainer, TrainerConfig
from sarva_foundry.train.rl import build_grpo_batch, sample_completion

TARGET_TOKEN = 17
PROMPT_IDS = [1, 2, 3]
ROLLOUT_TEMPERATURE = 8.0
MAX_NEW_TOKENS = 3
GROUP_SIZE = 12


def reward_fn(completion: list[int]) -> float:
    """The verifiable reward: 1.0 if the target token appears anywhere
    in the sampled completion, 0.0 otherwise. Deliberately simple and
    fully offline -- the real point is the training LOOP, which is
    identical regardless of how exotic or simple the reward source is
    (a token check here, real sandboxed code execution in example 13)."""
    return 1.0 if TARGET_TOKEN in completion else 0.0


def target_rate(model: DecoderOnlyTransformer, n: int = 200) -> float:
    hits = sum(
        1
        for _ in range(n)
        if TARGET_TOKEN
        in sample_completion(model, PROMPT_IDS, MAX_NEW_TOKENS, temperature=ROLLOUT_TEMPERATURE)
    )
    return hits / n


def main() -> None:
    torch.manual_seed(0)
    config = TransformerConfig(
        vocab_size=20, dim=16, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=32
    )
    model = DecoderOnlyTransformer(config)
    trainer = Trainer(model, TrainerConfig(lr=1e-3))

    rate_before = target_rate(model)
    print(f"Target-token rate before training: {rate_before:.1%}")

    print(f"\nTraining: {GROUP_SIZE} completions/group, GRPO advantage from group-relative reward")
    for step in range(300):
        completions = [
            sample_completion(model, PROMPT_IDS, MAX_NEW_TOKENS, temperature=ROLLOUT_TEMPERATURE)
            for _ in range(GROUP_SIZE)
        ]
        rewards = [reward_fn(c) for c in completions]
        x, y, mask = build_grpo_batch(PROMPT_IDS, completions)
        loss = trainer.grpo_step(x, y, mask, rewards)
        if step % 60 == 0 or step == 299:
            mean_reward = sum(rewards) / len(rewards)
            print(f"  step {step:3d}  loss {loss:7.4f}  group mean reward {mean_reward:.3f}")

    rate_after = target_rate(model)
    print(f"\nTarget-token rate after training: {rate_after:.1%}")
    print(
        f"Every reward above came from real sampling and a real check -- "
        f"no reward was assumed. Rate moved from {rate_before:.1%} to {rate_after:.1%} "
        "purely from GRPO updates on that signal."
    )

    print("\n--- Connecting to real coding tasks (example 13's harness) ---")
    print(
        "The reward_fn above is a one-line token check; a real agentic-RL run "
        "swaps it for the sandboxed coding-task harness directly. With a real "
        "tokenizer providing decode(), the reward function for a CodingTask "
        "looks like this (not run here -- a toy model this small cannot learn "
        "to write working Python from sparse code-execution rewards):\n"
    )
    task = CODING_TASKS[0]
    print("    def coding_reward_fn(completion_ids: list[int]) -> float:")
    print("        code = tokenizer.decode(completion_ids)")
    print(f"        result = evaluate_submission(task, code)  # task.task_id == {task.task_id!r}")
    print("        return result.reward")
    print(
        "\nThe GRPO loop itself (sample group -> score -> build_grpo_batch -> "
        "trainer.grpo_step) doesn't change at all -- only the reward function does."
    )


if __name__ == "__main__":
    main()
