"""A minimal pretraining loop with checkpoint/resume (spec §3.6d, the
single-process slice of it — distributed training (FSDP/3D parallelism)
is future work).

Checkpointing is the entire point of this module: a training run that
can't resume means every crash, preemption, or intentional pause loses
all compute spent so far. Getting resume *bit-identical* to uninterrupted
training requires saving not just model weights but full optimizer state
(AdamW's per-parameter momentum and variance buffers) — a resume that
only restores weights silently restarts momentum from zero, which trains
differently than the run it claims to be resuming, with no error to catch
the difference. `tests/foundry/test_trainer.py` verifies bit-identical
resume directly rather than assuming `state_dict()` round-trips correctly.

Also doubles as the SFT trainer (spec §3.6e's first post-training step):
`train_step`'s optional `loss_mask` is the entire difference between
pretraining and SFT here — same optimizer, same schedule, same
checkpoint/resume machinery, just a masked loss instead of an unmasked
one. See `sarva_foundry.train.sft` for building that mask from
(prompt, response) pairs.

`dpo_step` adds the second post-training step, DPO (`sarva_foundry.
train.dpo`) — a genuinely different shape of update (it needs four
forward passes: policy and a frozen reference model, each on a chosen
and a rejected response), so it's a distinct method rather than another
`train_step` parameter, but shares the same optimizer/grad-clip/step-
counting machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sarva_foundry.train.dpo import dpo_loss, sequence_logprobs
from sarva_foundry.train.schedule import WarmupCosineSchedule


@dataclass
class TrainerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float | None = 1.0
    # None = flat LR (the original behavior, still the default — a
    # schedule is an opt-in choice tied to a specific planned run length,
    # not something to impose silently on every caller).
    schedule: WarmupCosineSchedule | None = None


class Trainer:
    def __init__(self, model: nn.Module, config: TrainerConfig | None = None):
        self.model = model
        self.config = config or TrainerConfig()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self.step = 0

    def train_step(self, x: Tensor, y: Tensor, loss_mask: Tensor | None = None) -> float:
        """`loss_mask` (same shape as `y`, 1.0 = include in loss, 0.0 =
        exclude) is what SFT training uses (spec §3.6e) to exclude prompt
        tokens from the objective — the model must learn to predict the
        *response*, never the prompt. `None` (the default, and every call
        site before this parameter existed) is exactly the original
        unmasked behavior: every position contributes equally."""
        self.model.train()
        if self.config.schedule is not None:
            lr = self.config.schedule.lr_at(self.step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        logits = self.model(x)
        if loss_mask is None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        else:
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
            )
            mask_flat = loss_mask.reshape(-1).float()
            # clamp_min guards an all-masked-out batch (e.g. every
            # example padded to the same length with nothing left to
            # predict) from dividing by zero -- loss is then exactly 0,
            # not NaN, and contributes no gradient either way.
            loss = (per_token_loss * mask_flat).sum() / mask_flat.sum().clamp_min(1.0)
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.step += 1
        return loss.item()

    def dpo_step(
        self,
        ref_model: nn.Module,
        chosen: tuple[Tensor, Tensor, Tensor],
        rejected: tuple[Tensor, Tensor, Tensor],
        beta: float = 0.1,
    ) -> float:
        """One DPO update (spec §3.6e). `chosen`/`rejected` are each an
        `(input_ids, target_ids, loss_mask)` triple from `sarva_foundry.
        train.dpo.build_dpo_batch` — the SAME shape `train_step`'s
        `loss_mask` uses, since DPO's response log-probability is
        exactly the SFT loss's per-token log-probability, summed instead
        of averaged, over the same masked positions.

        `ref_model` is the caller's responsibility to construct and keep
        frozen (typically a `copy.deepcopy` of the SFT checkpoint DPO
        starts from) — `Trainer` doesn't own or manage a second model's
        lifecycle, matching its existing "thin, caller supplies the
        model" contract. Its forward pass runs under `torch.no_grad()`
        regardless of `ref_model`'s own `requires_grad` settings, so a
        caller who forgets to freeze it still gets a correct update."""
        self.model.train()
        ref_model.eval()

        chosen_x, chosen_y, chosen_mask = chosen
        rejected_x, rejected_y, rejected_mask = rejected

        policy_chosen_lp = sequence_logprobs(self.model, chosen_x, chosen_y, chosen_mask)
        policy_rejected_lp = sequence_logprobs(self.model, rejected_x, rejected_y, rejected_mask)
        with torch.no_grad():
            ref_chosen_lp = sequence_logprobs(ref_model, chosen_x, chosen_y, chosen_mask)
            ref_rejected_lp = sequence_logprobs(ref_model, rejected_x, rejected_y, rejected_mask)

        loss = dpo_loss(policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta)

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.step += 1
        return loss.item()

    def save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "step": self.step,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        # weights_only=True: refuse to unpickle anything beyond the
        # documented safe types (tensors, plain containers), the same
        # security posture as any other input that names a file on disk
        # someone else could have written.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.step = checkpoint["step"]
