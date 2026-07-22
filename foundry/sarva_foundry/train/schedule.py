"""sarva_foundry.train.schedule — linear warmup + cosine decay, the
standard learning-rate shape for pretraining runs (spec §3.6d: `Trainer`
previously used a flat LR, a named known gap). A flat LR either risks
early instability at the model's random initialization (no warmup) or
wastes the tail of training that a decayed LR would have converged
further with (no decay) — warmup + cosine decay is what essentially
every real pretraining run, from GPT-2 onward, actually uses.

Implemented as a pure function of step count, not stored/mutable state:
`Trainer.train_step` calls `lr_at(self.step)` fresh on every call, so
`Trainer`'s existing checkpoint/resume — which already restores
`self.step` — resumes the schedule correctly for free. No separate
schedule state to save or restore, and no way for the two to drift out
of sync.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class WarmupCosineSchedule:
    peak_lr: float
    min_lr: float
    warmup_steps: int
    total_steps: int

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.total_steps <= self.warmup_steps:
            raise ValueError(
                f"total_steps ({self.total_steps}) must be > warmup_steps ({self.warmup_steps})"
            )
        if self.min_lr > self.peak_lr:
            raise ValueError(f"min_lr ({self.min_lr}) must be <= peak_lr ({self.peak_lr})")

    def lr_at(self, step: int) -> float:
        """The LR for `step` (0-indexed): linear ramp 0 -> peak_lr over the
        first `warmup_steps`, then a cosine decay from peak_lr down to
        min_lr over the remaining steps, holding at min_lr past
        `total_steps` (a run that overshoots its planned length degrades
        gracefully to the floor rather than the schedule going undefined
        or, worse, cosine's periodicity silently ramping back up)."""
        if step < self.warmup_steps:
            if self.warmup_steps == 0:
                return self.peak_lr
            return self.peak_lr * (step + 1) / self.warmup_steps
        if step >= self.total_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine
