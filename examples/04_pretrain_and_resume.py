"""Example 04 — Pretrain with checkpoint/resume and a warmup+cosine LR
schedule.

The full foundry pipeline so far, end to end: tokenizer -> chunked
dataset -> transformer -> Trainer, with a checkpoint saved partway through
and a *fresh process* (simulated here by fresh objects) resuming from it.
Prints both loss and the current learning rate across the interruption:
loss keeps descending instead of spiking (optimizer momentum survived
the round-trip, not just model weights), and the LR curve continues
exactly where it left off (warmup, then cosine decay) rather than
restarting — both driven by the same checkpointed step count.

Run: uv run python examples/04_pretrain_and_resume.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from sarva_foundry.data import DOCUMENT_SEPARATOR, TextChunkDataset, tokenize_corpus
from sarva_foundry.model import DecoderOnlyTransformer, TransformerConfig
from sarva_foundry.tokenizer import ByteLevelBPETokenizer
from sarva_foundry.train import Trainer, TrainerConfig, WarmupCosineSchedule

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox is quick and the dog is lazy",
    "she sells seashells by the seashore",
    "how much wood would a woodchuck chuck if a woodchuck could chuck wood",
]


def main() -> None:
    torch.manual_seed(0)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(CORPUS, vocab_size=300, special_tokens=[DOCUMENT_SEPARATOR])
    token_ids = tokenize_corpus(CORPUS, tokenizer)
    dataset = TextChunkDataset(token_ids, seq_len=16)
    print(f"Corpus: {len(token_ids)} tokens -> {len(dataset)} training chunks of 16 tokens each")

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=16
    )
    # total_steps=60 covers the full planned run (30 + 30) across both
    # halves below -- the schedule doesn't know or care that it's split
    # across two Trainer instances, only that the step count is continuous.
    schedule = WarmupCosineSchedule(peak_lr=3e-3, min_lr=3e-4, warmup_steps=10, total_steps=60)
    trainer_config = TrainerConfig(schedule=schedule)

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_path = Path(tmp) / "checkpoint.pt"

        print("\n--- run 1: train 30 steps, checkpoint, then stop ---")
        model_1 = DecoderOnlyTransformer(config)
        trainer_1 = Trainer(model_1, trainer_config)
        for step in range(30):
            x, y = dataset[step % len(dataset)]
            loss = trainer_1.train_step(x.unsqueeze(0), y.unsqueeze(0))
            if step % 10 == 0:
                lr = trainer_1.optimizer.param_groups[0]["lr"]
                print(f"  step {step:3d}  loss {loss:.4f}  lr {lr:.5f}")
        trainer_1.save_checkpoint(checkpoint_path)
        print(f"  checkpoint saved at step {trainer_1.step}")

        print("\n--- run 2: fresh process, resume from checkpoint, train 30 more steps ---")
        model_2 = DecoderOnlyTransformer(config)  # freshly initialized -- overwritten by load
        trainer_2 = Trainer(model_2, trainer_config)
        trainer_2.load_checkpoint(checkpoint_path)
        print(f"  resumed at step {trainer_2.step}")
        for step in range(30):
            x, y = dataset[(30 + step) % len(dataset)]
            loss = trainer_2.train_step(x.unsqueeze(0), y.unsqueeze(0))
            if step % 10 == 0:
                lr = trainer_2.optimizer.param_groups[0]["lr"]
                print(f"  step {trainer_2.step:3d}  loss {loss:.4f}  lr {lr:.5f}")

    print(
        "\nLoss keeps descending across the checkpoint boundary instead of "
        "spiking -- optimizer momentum survived the round-trip. The LR "
        "column keeps decaying smoothly too, rather than resetting to the "
        "warmup value -- both driven by the same checkpointed step count."
    )


if __name__ == "__main__":
    main()
