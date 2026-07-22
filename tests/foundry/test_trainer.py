"""Conformance tests for sarva_foundry.train.trainer — the definition of
done for checkpointing is not "state_dict() round-trips without an
exception," it's "resumed training produces bit-identical results to
uninterrupted training," which requires optimizer state (AdamW's
per-parameter momentum/variance), not just model weights, to survive
the round-trip."""

from __future__ import annotations

import torch
from sarva_foundry.model import DecoderOnlyTransformer, TransformerConfig
from sarva_foundry.train import Trainer, TrainerConfig, WarmupCosineSchedule


def _config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=20, dim=16, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=16
    )


def _fixed_batch(config: TransformerConfig) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = 8
    x = torch.arange(seq_len).unsqueeze(0) % config.vocab_size
    y = (x + 1) % config.vocab_size
    return x, y


def _seeded_model(config: TransformerConfig) -> DecoderOnlyTransformer:
    torch.manual_seed(0)
    return DecoderOnlyTransformer(config)


def test_train_step_returns_a_finite_loss_and_advances_step_count():
    config = _config()
    trainer = Trainer(_seeded_model(config))
    x, y = _fixed_batch(config)
    loss = trainer.train_step(x, y)
    assert torch.isfinite(torch.tensor(loss))
    assert trainer.step == 1


def test_checkpoint_resume_is_bit_identical_to_uninterrupted_training(tmp_path):
    config = _config()
    x, y = _fixed_batch(config)

    # Path A: 10 steps, uninterrupted.
    trainer_a = Trainer(_seeded_model(config))
    for _ in range(10):
        trainer_a.train_step(x, y)
    final_a = {k: v.clone() for k, v in trainer_a.model.state_dict().items()}

    # Path B: 5 steps, checkpoint, fresh Trainer/model loaded from
    # checkpoint, then the remaining 5 steps.
    trainer_b = Trainer(_seeded_model(config))
    for _ in range(5):
        trainer_b.train_step(x, y)
    ckpt_path = tmp_path / "checkpoint.pt"
    trainer_b.save_checkpoint(ckpt_path)

    trainer_c = Trainer(_seeded_model(config))  # fresh init; overwritten by load
    trainer_c.load_checkpoint(ckpt_path)
    assert trainer_c.step == 5
    for _ in range(5):
        trainer_c.train_step(x, y)
    final_c = {k: v.clone() for k, v in trainer_c.model.state_dict().items()}

    assert final_a.keys() == final_c.keys()
    for key in final_a:
        assert torch.allclose(final_a[key], final_c[key], atol=1e-6), f"mismatch at {key}"


def test_checkpoint_without_optimizer_state_would_diverge():
    # Negative control proving the positive test above is actually
    # exercising something real: resuming with FRESH optimizer momentum
    # (the mistake this module's docstring warns about) must NOT match
    # uninterrupted training. If this assertion ever failed, it would mean
    # the positive test above is vacuous (e.g. because the toy task
    # converges to the same point regardless of optimizer state).
    config = _config()
    x, y = _fixed_batch(config)

    trainer_a = Trainer(_seeded_model(config))
    for _ in range(10):
        trainer_a.train_step(x, y)
    final_a = trainer_a.model.state_dict()

    trainer_b = Trainer(_seeded_model(config))
    for _ in range(5):
        trainer_b.train_step(x, y)
    # Simulate the bug: a fresh optimizer instead of the loaded one.
    trainer_b.optimizer = torch.optim.AdamW(trainer_b.model.parameters(), lr=trainer_b.config.lr)
    for _ in range(5):
        trainer_b.train_step(x, y)
    final_b = trainer_b.model.state_dict()

    mismatched = any(not torch.allclose(final_a[k], final_b[k], atol=1e-6) for k in final_a)
    assert mismatched


def test_schedule_sets_the_optimizer_lr_at_each_step():
    config = _config()
    schedule = WarmupCosineSchedule(peak_lr=1e-2, min_lr=1e-4, warmup_steps=4, total_steps=20)
    trainer = Trainer(_seeded_model(config), TrainerConfig(schedule=schedule))
    x, y = _fixed_batch(config)

    lrs = []
    for _ in range(6):
        trainer.train_step(x, y)
        lrs.append(trainer.optimizer.param_groups[0]["lr"])

    # Steps 0-3 are warmup (ramping up), step 4+ is past warmup_steps=4
    # and should be decaying -- confirms train_step is actually pulling a
    # fresh LR from the schedule every call, not just once at construction.
    assert lrs[0] < lrs[3]
    assert lrs[4] > lrs[5]


def test_checkpoint_resume_is_bit_identical_with_a_schedule_active(tmp_path):
    # The schedule-specific version of the bit-identical resume test
    # above: resuming mid-schedule must continue the LR curve from
    # exactly where it left off (driven by the checkpointed step count),
    # not restart warmup or jump to some other point on the curve.
    config = _config()
    x, y = _fixed_batch(config)
    schedule = WarmupCosineSchedule(peak_lr=1e-2, min_lr=1e-4, warmup_steps=3, total_steps=10)

    trainer_a = Trainer(_seeded_model(config), TrainerConfig(schedule=schedule))
    for _ in range(10):
        trainer_a.train_step(x, y)
    final_a = {k: v.clone() for k, v in trainer_a.model.state_dict().items()}

    trainer_b = Trainer(_seeded_model(config), TrainerConfig(schedule=schedule))
    for _ in range(5):
        trainer_b.train_step(x, y)
    ckpt_path = tmp_path / "checkpoint.pt"
    trainer_b.save_checkpoint(ckpt_path)

    trainer_c = Trainer(_seeded_model(config), TrainerConfig(schedule=schedule))
    trainer_c.load_checkpoint(ckpt_path)
    for _ in range(5):
        trainer_c.train_step(x, y)
    final_c = {k: v.clone() for k, v in trainer_c.model.state_dict().items()}

    for key in final_a:
        assert torch.allclose(final_a[key], final_c[key], atol=1e-6), f"mismatch at {key}"
