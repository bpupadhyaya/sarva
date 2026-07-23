"""sarva_foundry.train.rl — GRPO-style policy-gradient training from
verifiable rewards (spec §3.6e: agentic RL, "this, not pretraining, is
what turns a base model into a Fable/K3-class agent"). This is the
training loop the environment harness (`sarva_foundry.rl.environment`)
was always missing — closes the one piece of agentic RL still real,
deferred work as of the entry that shipped the harness.

**Group Relative Policy Optimization** (Shao et al. 2024, DeepSeekMath):
for each prompt, sample a *group* of K completions from the current
policy, score each with a real (task-specific) reward function, and use
each completion's advantage *relative to its own group's mean reward* —
`(reward - group_mean) / (group_std + eps)` — as the policy-gradient
weight. No separate value network/critic needed (the second network
full PPO requires), which is exactly why GRPO is the lighter-weight,
teaching-scale-appropriate choice here.

The log-probability term reuses `sarva_foundry.train.dpo.sequence_logprobs`
directly rather than reimplementing it: REINFORCE's gradient estimator
is `E[R · grad_theta log P(action)]`, and computing log P(sampled
completion) under the *current* model parameters (with gradient
tracking) is exactly what that function already does for DPO — same
math, same "mask picks out the completion, never the prompt" contract
`sarva_foundry.train.sft` established first. Sampling itself runs under
`torch.no_grad()` (it isn't differentiable, and doesn't need to be —
the gradient comes entirely from the log-prob re-evaluation afterward,
not from the sampling step).

Mirrors the existing `build_sft_batch` → `Trainer.train_step` and
`build_dpo_batch` → `Trainer.dpo_step` shape: `build_grpo_batch` (pure
data prep, no gradient involved) hands `Trainer.grpo_step` (in
`trainer.py`) an already-built, already-masked batch plus the rewards —
sampling and reward-scoring are the caller's job (see
`examples/14_grpo_rl_training.py`), keeping `Trainer` focused on "given
data and rewards, do the gradient update."
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def sample_completion(
    model: nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    stop_token_id: int | None = None,
) -> list[int]:
    """Autoregressive sampling under `torch.no_grad()` — not
    differentiable, and doesn't need to be (see module docstring).
    `temperature <= 0` means greedy (always the argmax token); otherwise
    samples from the softmax distribution at that temperature. Returns
    only the *newly generated* token ids, not the prompt — matching what
    `build_grpo_batch` expects as a "completion"."""
    model.eval()
    generated: list[int] = []
    ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(torch.tensor(ids).unsqueeze(0))
            next_logits = logits[0, -1]
            if temperature <= 0:
                next_id = int(next_logits.argmax())
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1))
            generated.append(next_id)
            ids.append(next_id)
            if stop_token_id is not None and next_id == stop_token_id:
                break
    return generated


def build_grpo_batch(
    prompt_ids: list[int], completions: list[list[int]], pad_token_id: int = 0
) -> tuple[Tensor, Tensor, Tensor]:
    """Pads `prompt_ids + completion` for every completion in the group
    to the same length and shifts for next-token prediction, returning
    `(input_ids, target_ids, loss_mask)` — the identical shape
    `build_sft_batch` produces, with the loss mask covering only each
    completion's own tokens (never the shared prompt), so
    `sequence_logprobs` sums exactly the log-probability of what the
    policy actually generated."""
    sequences = [(prompt_ids + c, len(prompt_ids)) for c in completions]
    max_len = max(len(ids) for ids, _ in sequences)
    if max_len < 2:
        raise ValueError("every sequence needs at least 2 tokens to form one training pair")

    input_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for ids, prompt_len in sequences:
        pad_len = max_len - len(ids)
        padded = ids + [pad_token_id] * pad_len
        mask = [0] * prompt_len + [1] * (len(ids) - prompt_len) + [0] * pad_len
        input_rows.append(padded[:-1])
        target_rows.append(padded[1:])
        mask_rows.append(mask[1:])

    return (
        torch.tensor(input_rows, dtype=torch.long),
        torch.tensor(target_rows, dtype=torch.long),
        torch.tensor(mask_rows, dtype=torch.float),
    )
