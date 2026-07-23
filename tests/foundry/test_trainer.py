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


def test_train_step_with_no_mask_is_bit_identical_to_before_the_parameter_existed():
    # Regression guard: loss_mask=None (the default, and every call site
    # before SFT support was added) must produce exactly the same loss
    # as calling train_step with no third argument at all.
    config = _config()
    x, y = _fixed_batch(config)

    trainer_a = Trainer(_seeded_model(config))
    loss_a = trainer_a.train_step(x, y)

    trainer_b = Trainer(_seeded_model(config))
    loss_b = trainer_b.train_step(x, y, loss_mask=None)

    assert loss_a == loss_b


def test_loss_mask_makes_masked_target_values_irrelevant_to_the_loss():
    # The defining correctness property of SFT masking: two batches that
    # differ ONLY at masked-out target positions must produce the exact
    # same loss, proving those positions genuinely don't contribute --
    # not just that the returned loss is "reasonable."
    config = _config()
    x, y_a = _fixed_batch(config)
    y_b = y_a.clone()
    mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    # Corrupt only the masked-out (prompt-like) positions.
    y_b[0, :3] = (y_b[0, :3] + 7) % config.vocab_size
    assert not torch.equal(y_a, y_b)  # sanity: the corruption actually changed something

    trainer_a = Trainer(_seeded_model(config))
    loss_a = trainer_a.train_step(x, y_a, loss_mask=mask)

    trainer_b = Trainer(_seeded_model(config))
    loss_b = trainer_b.train_step(x, y_b, loss_mask=mask)

    assert loss_a == loss_b


def test_loss_mask_makes_unmasked_target_values_still_matter():
    # The complementary property: changing a target at an UNMASKED
    # (response-like) position must change the loss -- otherwise the
    # mask could trivially "pass" the test above by excluding
    # everything, which would make SFT training a no-op instead of
    # actually training on the response.
    config = _config()
    x, y_a = _fixed_batch(config)
    y_b = y_a.clone()
    mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    y_b[0, 3] = (y_b[0, 3] + 7) % config.vocab_size  # an UNMASKED position this time

    trainer_a = Trainer(_seeded_model(config))
    loss_a = trainer_a.train_step(x, y_a, loss_mask=mask)

    trainer_b = Trainer(_seeded_model(config))
    loss_b = trainer_b.train_step(x, y_b, loss_mask=mask)

    assert loss_a != loss_b


def test_sft_style_masked_training_still_decreases_loss_on_the_real_objective():
    # An end-to-end trainability proof using an SFT-shaped batch built
    # through the real sarva_foundry.train.sft pipeline, not a synthetic
    # mask -- mirrors every other trainability test in this suite (loss
    # must actually decrease), but on the masked objective specifically.
    from sarva_foundry.data import DOCUMENT_SEPARATOR
    from sarva_foundry.tokenizer import ByteLevelBPETokenizer
    from sarva_foundry.train.sft import SFTExample, build_sft_batch

    torch.manual_seed(0)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        ["what is 2 2", "four", "hello world"],
        vocab_size=280,
        special_tokens=[DOCUMENT_SEPARATOR],
    )
    example = SFTExample(prompt="what is 2 2 ", response="four")
    x, y, mask = build_sft_batch([example], tokenizer)

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        dim=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=32,
    )
    model = DecoderOnlyTransformer(config)
    trainer = Trainer(model)

    losses = [trainer.train_step(x, y, loss_mask=mask) for _ in range(60)]
    assert losses[-1] < losses[0] * 0.5

    # And the model must have actually learned to predict the response
    # tokens specifically, not just driven the masked loss to zero by
    # some degenerate shortcut -- check the model's own greedy
    # prediction at the first response position now matches the real
    # response token.
    model.eval()
    with torch.no_grad():
        logits = model(x)
    response_start = int((mask[0] == 1).nonzero()[0].item())
    predicted = logits[0, response_start].argmax().item()
    assert predicted == y[0, response_start].item()
