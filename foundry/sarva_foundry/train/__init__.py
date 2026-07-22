"""sarva_foundry.train — the pretraining loop, with checkpoint/resume
(spec §3.6d, single-process slice)."""

from sarva_foundry.train.trainer import Trainer, TrainerConfig

__all__ = ["Trainer", "TrainerConfig"]
