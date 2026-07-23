"""sarva_foundry.train — the pretraining loop, with checkpoint/resume
and a warmup+cosine LR schedule (spec §3.6d, single-process slice), and
spec §3.6e's full post-training line: SFT (`sft.py`, `train_step`'s
`loss_mask`), DPO (`dpo.py`, `Trainer.dpo_step`), and GRPO-style
agentic RL (`rl.py`, `Trainer.grpo_step`)."""

from sarva_foundry.train.dpo import DPOExample, build_dpo_batch, dpo_loss, sequence_logprobs
from sarva_foundry.train.rl import build_grpo_batch, sample_completion
from sarva_foundry.train.schedule import WarmupCosineSchedule
from sarva_foundry.train.sft import SFTExample, build_sft_batch, encode_sft_example
from sarva_foundry.train.trainer import Trainer, TrainerConfig

__all__ = [
    "DPOExample",
    "SFTExample",
    "Trainer",
    "TrainerConfig",
    "WarmupCosineSchedule",
    "build_dpo_batch",
    "build_grpo_batch",
    "build_sft_batch",
    "dpo_loss",
    "encode_sft_example",
    "sample_completion",
    "sequence_logprobs",
]
