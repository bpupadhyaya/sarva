"""sarva_foundry.train.dpo — Direct Preference Optimization (Rafailov et
al. 2023), spec §3.6e's second post-training step ("SFT -> DPO/RLHF ->
agentic RL"). DPO turns a `(prompt, chosen, rejected)` preference triple
directly into a training signal, without a separate reward model or
RL rollouts — the paper's whole insight is that the optimal reward
model implied by RLHF's objective has a closed form in terms of the
policy itself, so preference data can train the policy directly:

    L_DPO = -log sigmoid(
        beta * [ (log pi(y_w|x) - log ref(y_w|x))
               - (log pi(y_l|x) - log ref(y_l|x)) ]
    )

`y_w` (chosen/"winning") and `y_l` (rejected/"losing") responses to the
same prompt `x`; `pi` is the policy being trained, `ref` a frozen
reference model (in practice, the SFT checkpoint DPO starts from) that
keeps the policy from drifting arbitrarily far in order to satisfy the
preference — `beta` controls how tightly.

Reuses `sarva_foundry.train.sft`'s `SFTExample`/`build_sft_batch`
rather than a parallel encoding path: a DPO preference triple is exactly
two SFT-shaped `(prompt, response)` pairs sharing one prompt, so
`build_dpo_batch` just calls `build_sft_batch` twice (once for chosen
responses, once for rejected) instead of reimplementing tokenization,
padding, and loss-mask construction a second time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor, nn

from sarva_foundry.tokenizer import ByteLevelBPETokenizer
from sarva_foundry.train.sft import SFTExample, build_sft_batch

SFTBatch = tuple[Tensor, Tensor, Tensor]  # (input_ids, target_ids, loss_mask)


@dataclass(frozen=True)
class DPOExample:
    prompt: str
    chosen: str
    rejected: str


def build_dpo_batch(
    examples: list[DPOExample], tokenizer: ByteLevelBPETokenizer
) -> tuple[SFTBatch, SFTBatch]:
    """Returns `(chosen_batch, rejected_batch)`, each an SFT-shaped
    `(input_ids, target_ids, loss_mask)` triple — chosen and rejected are
    padded independently (they're typically different lengths), so
    `sequence_logprobs` sums each one over exactly its own response
    tokens."""
    chosen_batch = build_sft_batch(
        [SFTExample(prompt=e.prompt, response=e.chosen) for e in examples], tokenizer
    )
    rejected_batch = build_sft_batch(
        [SFTExample(prompt=e.prompt, response=e.rejected) for e in examples], tokenizer
    )
    return chosen_batch, rejected_batch


def sequence_logprobs(model: nn.Module, x: Tensor, y: Tensor, loss_mask: Tensor) -> Tensor:
    """`sum_i loss_mask[i] * log p(y[i] | x[:i+1])`, per batch row — the
    log-probability of exactly the response portion of a sequence under
    `model`, the quantity DPO compares between the policy and the
    reference model. Shares the "mask picks out the response, never the
    prompt" contract `sarva_foundry.train.sft` establishes; a mask bug
    here would corrupt the preference signal itself, not just waste
    training compute."""
    logits = model(x)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (batch, seq)
    return (token_logprobs * loss_mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logprobs: Tensor,
    policy_rejected_logprobs: Tensor,
    ref_chosen_logprobs: Tensor,
    ref_rejected_logprobs: Tensor,
    beta: float = 0.1,
) -> Tensor:
    """The DPO objective itself, taking already-computed sequence
    log-probabilities (see `sequence_logprobs`) rather than models —
    kept separate from `Trainer.dpo_step` so the loss math is testable
    in complete isolation from any model forward pass.

    A real bug found by a fresh-eyes sweep, the same severity class as
    `TrainerConfig.grad_clip`/`WarmupCosineSchedule.peak_lr`'s own
    fixes elsewhere in this package: `beta` was never validated, and
    nothing in this function's own math guards its sign. Confirmed
    live with a real model, a real tokenizer, and real DPO training
    steps: `beta=-0.1` took a real (chosen - rejected) log-probability
    margin from +5.07 to -6.30 over 10 real `Trainer.dpo_step` calls --
    the policy was actively trained to PREFER the rejected response
    over the chosen one, the exact opposite of what DPO exists to do,
    not merely a failure to improve. `beta=0` is degenerate too (though
    less severe): `logits` still varies per example, but `beta *
    logits` is identically zero regardless, so `-F.logsigmoid(0).mean()`
    is a constant loss with zero gradient with respect to either
    log-probability difference -- every `dpo_step` becomes a real,
    silent no-op that still advances the step counter. Reachable the
    same way every other numeric parameter in this package is -- a
    caller deriving `beta` programmatically or a simple sign-flip typo
    in a training script, not an adversarial input."""
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError(f"beta must be a finite positive number, got {beta}")
    pi_logratios = policy_chosen_logprobs - policy_rejected_logprobs
    ref_logratios = ref_chosen_logprobs - ref_rejected_logprobs
    logits = pi_logratios - ref_logratios
    return -F.logsigmoid(beta * logits).mean()
