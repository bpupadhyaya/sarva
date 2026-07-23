"""sarva_foundry.train — the pretraining loop, with checkpoint/resume
and a warmup+cosine LR schedule (spec §3.6d, single-process slice), and
spec §3.6e's full post-training line: SFT (`sft.py`, `train_step`'s
`loss_mask`), DPO (`dpo.py`, `Trainer.dpo_step`), GRPO-style agentic RL
(`rl.py`, `Trainer.grpo_step`), and reasoning/thinking-token training
(`reasoning.py` — SFT + GRPO composed with an R1-style verifiable
reward, not a new algorithm of its own)."""

from sarva_foundry.train.dpo import DPOExample, build_dpo_batch, dpo_loss, sequence_logprobs
from sarva_foundry.train.reasoning import (
    THINK_END,
    THINK_START,
    answer_reward,
    format_reward,
    reasoning_reward,
)
from sarva_foundry.train.rl import build_grpo_batch, sample_completion
from sarva_foundry.train.schedule import WarmupCosineSchedule
from sarva_foundry.train.sft import SFTExample, build_sft_batch, encode_sft_example
from sarva_foundry.train.trainer import Trainer, TrainerConfig

__all__ = [
    "DPOExample",
    "SFTExample",
    "THINK_END",
    "THINK_START",
    "Trainer",
    "TrainerConfig",
    "WarmupCosineSchedule",
    "answer_reward",
    "build_dpo_batch",
    "build_grpo_batch",
    "build_sft_batch",
    "dpo_loss",
    "encode_sft_example",
    "format_reward",
    "reasoning_reward",
    "sample_completion",
    "sequence_logprobs",
]
