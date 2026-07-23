"""sarva_foundry.train — the pretraining loop, with checkpoint/resume
and a warmup+cosine LR schedule (spec §3.6d, single-process slice), and
SFT data prep (spec §3.6e's first post-training step, `sft.py`) that
turns `Trainer` into an SFT trainer via `train_step`'s `loss_mask`."""

from sarva_foundry.train.schedule import WarmupCosineSchedule
from sarva_foundry.train.sft import SFTExample, build_sft_batch, encode_sft_example
from sarva_foundry.train.trainer import Trainer, TrainerConfig

__all__ = [
    "SFTExample",
    "Trainer",
    "TrainerConfig",
    "WarmupCosineSchedule",
    "build_sft_batch",
    "encode_sft_example",
]
