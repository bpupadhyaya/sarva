"""Conformance tests for sarva_foundry.train.schedule."""

from __future__ import annotations

import pytest
from sarva_foundry.train import WarmupCosineSchedule


def _schedule(**overrides) -> WarmupCosineSchedule:
    defaults = dict(peak_lr=1.0, min_lr=0.1, warmup_steps=10, total_steps=100)
    defaults.update(overrides)
    return WarmupCosineSchedule(**defaults)


def test_warmup_ramps_linearly_from_near_zero_to_peak():
    sched = _schedule(warmup_steps=10)
    lr_first = sched.lr_at(0)
    lr_mid = sched.lr_at(5)
    lr_last_warmup_step = sched.lr_at(9)
    assert 0 < lr_first < lr_mid < lr_last_warmup_step
    assert lr_last_warmup_step == pytest.approx(1.0, rel=1e-6)


def test_warmup_of_zero_starts_at_peak_immediately():
    sched = _schedule(warmup_steps=0, total_steps=100)
    assert sched.lr_at(0) == pytest.approx(1.0)


def test_lr_decays_monotonically_after_warmup():
    sched = _schedule(warmup_steps=10, total_steps=110)
    steps = [10, 30, 50, 70, 90, 109]
    lrs = [sched.lr_at(s) for s in steps]
    assert all(a > b for a, b in zip(lrs, lrs[1:], strict=False))


def test_lr_reaches_min_lr_at_total_steps():
    sched = _schedule(peak_lr=1.0, min_lr=0.1, warmup_steps=10, total_steps=110)
    assert sched.lr_at(110) == pytest.approx(0.1, abs=1e-6)


def test_lr_holds_at_min_lr_past_total_steps():
    # A run that overshoots its planned length should degrade gracefully
    # to the floor, not have cosine's periodicity ramp the LR back up.
    sched = _schedule(peak_lr=1.0, min_lr=0.1, warmup_steps=10, total_steps=110)
    assert sched.lr_at(110) == sched.lr_at(500) == pytest.approx(0.1)


def test_lr_never_exceeds_peak_and_never_negative_during_warmup():
    # min_lr is a floor for the post-warmup decay phase, not the whole
    # schedule -- warmup deliberately ramps from near-zero (the standard
    # convention, matching NanoGPT/Megatron-style schedules), so it can
    # and should dip below min_lr early on.
    sched = _schedule(peak_lr=2.0, min_lr=0.2, warmup_steps=20, total_steps=200)
    for step in range(0, 20):
        lr = sched.lr_at(step)
        assert 0 <= lr <= 2.0 + 1e-9


def test_lr_bounded_by_min_and_peak_after_warmup():
    sched = _schedule(peak_lr=2.0, min_lr=0.2, warmup_steps=20, total_steps=200)
    for step in range(20, 250, 7):
        lr = sched.lr_at(step)
        assert 0.2 - 1e-9 <= lr <= 2.0 + 1e-9


def test_rejects_negative_warmup_steps():
    with pytest.raises(ValueError, match="warmup_steps"):
        _schedule(warmup_steps=-1)


def test_rejects_total_steps_not_exceeding_warmup_steps():
    with pytest.raises(ValueError, match="total_steps"):
        _schedule(warmup_steps=100, total_steps=100)


def test_rejects_min_lr_above_peak_lr():
    with pytest.raises(ValueError, match="min_lr"):
        _schedule(peak_lr=0.1, min_lr=1.0)
