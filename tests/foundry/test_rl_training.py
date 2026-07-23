"""Conformance tests for sarva_foundry.train.rl — GRPO-style
policy-gradient training from verifiable rewards (spec §3.6e). Same bar
as SFT/DPO: shape tests aren't enough, the actual effect of training
(the policy's probability of a rewarded behavior increasing) has to be
demonstrated on a real model, not assumed from the math looking right
on paper."""

from __future__ import annotations

import pytest
import torch
from sarva_foundry.model import DecoderOnlyTransformer, TransformerConfig
from sarva_foundry.train import Trainer, TrainerConfig
from sarva_foundry.train.rl import build_grpo_batch, sample_completion

torch.manual_seed(0)


def _tiny_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=20, dim=16, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=32
    )


def test_sample_completion_returns_only_new_tokens_not_the_prompt():
    model = DecoderOnlyTransformer(_tiny_config())
    completion = sample_completion(model, prompt_ids=[1, 2, 3], max_new_tokens=5, temperature=1.0)
    assert len(completion) == 5
    assert all(0 <= tok < 20 for tok in completion)


def test_sample_completion_with_zero_temperature_is_greedy_and_deterministic():
    model = DecoderOnlyTransformer(_tiny_config())
    a = sample_completion(model, prompt_ids=[1, 2, 3], max_new_tokens=4, temperature=0.0)
    b = sample_completion(model, prompt_ids=[1, 2, 3], max_new_tokens=4, temperature=0.0)
    assert a == b


def test_sample_completion_stops_at_the_stop_token():
    model = DecoderOnlyTransformer(_tiny_config())
    # With temperature=0 (greedy), find whatever token the model picks
    # first, then use it as the stop token to prove generation actually
    # halts there instead of continuing to max_new_tokens.
    first_token = sample_completion(model, [1, 2, 3], max_new_tokens=1, temperature=0.0)[0]
    completion = sample_completion(
        model, [1, 2, 3], max_new_tokens=10, temperature=0.0, stop_token_id=first_token
    )
    assert completion == [first_token]


def test_build_grpo_batch_masks_only_completion_tokens():
    prompt_ids = [1, 2, 3]
    completions = [[4, 5], [6, 7, 8]]
    x, y, mask = build_grpo_batch(prompt_ids, completions)

    assert x.shape == y.shape == mask.shape
    assert x.shape[0] == 2
    # Shifting for next-token prediction drops exactly one position from
    # the front of the mask -- always a prompt position as long as the
    # prompt itself is non-empty, so the number of masked-IN
    # (completion) positions after the shift equals len(completion)
    # exactly, for every row regardless of padding.
    assert mask[0].sum().item() == len(completions[0])  # 2
    assert mask[1].sum().item() == len(completions[1])  # 3

    # Row 0 is the shorter, padded sequence -- confirm its padding
    # region is masked out too (not counted above only by coincidence).
    row0_ids = prompt_ids + completions[0]
    pad_len = len(prompt_ids + completions[1]) - len(row0_ids)
    assert pad_len == 1
    assert mask[0, -1].item() == 0  # the one padded position


def test_grpo_step_rejects_a_rewards_length_mismatch():
    model = DecoderOnlyTransformer(_tiny_config())
    trainer = Trainer(model)
    x, y, mask = build_grpo_batch([1, 2, 3], [[4, 5], [6, 7]])
    with pytest.raises(ValueError, match="rewards"):
        trainer.grpo_step(x, y, mask, rewards=[1.0])  # only 1 reward for 2 rows


def test_grpo_step_is_a_deliberate_noop_when_the_group_has_zero_variance():
    # Every completion in the group scoring identically means there's no
    # relative signal to learn from -- must be a real no-op (zero loss,
    # unchanged weights, step counter still advances), not a
    # divide-by-near-zero producing garbage.
    model = DecoderOnlyTransformer(_tiny_config())
    trainer = Trainer(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    x, y, mask = build_grpo_batch([1, 2, 3], [[4, 5], [6, 7]])
    loss = trainer.grpo_step(x, y, mask, rewards=[1.0, 1.0])

    assert loss == 0.0
    assert trainer.step == 1
    after = model.state_dict()
    for key in before:
        assert torch.equal(before[key], after[key]), f"weights changed at {key} on a no-op step"


def test_grpo_training_increases_the_rewarded_behaviors_probability():
    # The real end-to-end proof, mirroring DPO's preference-margin test:
    # after real GRPO training, the policy's probability of producing a
    # rewarded completion must be measurably HIGHER than at
    # initialization -- the actual thing GRPO training exists to
    # accomplish, not just "loss is finite."
    #
    # A high sampling temperature (8.0) is used deliberately for
    # rollout collection: this project's tiny, weight-tied,
    # freshly-initialized transformer was found (empirically, before
    # writing this test -- see BUILD-JOURNAL.md) to have extremely
    # peaked initial logits at temperature=1.0, collapsing sampling to
    # a single dominant token regardless of prompt and leaving no
    # exploration for GRPO to learn from at all. Elevated rollout
    # temperature is also standard real-world practice for RL
    # fine-tuning exploration, not just a workaround picked to make
    # this test pass.
    torch.manual_seed(0)
    config = _tiny_config()
    model = DecoderOnlyTransformer(config)
    trainer = Trainer(model, TrainerConfig(lr=1e-3))

    target_token = 17
    prompt_ids = [1, 2, 3]
    temperature = 8.0
    max_new_tokens = 3

    def reward_fn(completion: list[int]) -> float:
        return 1.0 if target_token in completion else 0.0

    def target_rate(n: int = 200) -> float:
        hits = sum(
            1
            for _ in range(n)
            if target_token
            in sample_completion(model, prompt_ids, max_new_tokens, temperature=temperature)
        )
        return hits / n

    rate_before = target_rate()

    for _ in range(300):
        completions = [
            sample_completion(model, prompt_ids, max_new_tokens, temperature=temperature)
            for _ in range(12)
        ]
        rewards = [reward_fn(c) for c in completions]
        x, y, mask = build_grpo_batch(prompt_ids, completions)
        trainer.grpo_step(x, y, mask, rewards)

    rate_after = target_rate()

    assert rate_after > rate_before + 0.3, (
        f"expected a large increase in target-token rate, got {rate_before} -> {rate_after}"
    )
