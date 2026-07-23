"""The real end-to-end proof for sarva_foundry.train.reasoning (spec
§3.6a: reasoning/thinking-token training) -- mirrors the exact scenario
in examples/17_reasoning_token_training.py: cold-start SFT teaches the
<think>...</think> format, then GRPO refines answer accuracy on top of
that baseline. The property that matters is real, measured improvement
in a GENUINELY well-formed completion's answer correctness, not just
that training runs without crashing -- see the two dedicated regression
tests below for a real reward-hacking exploit this scenario surfaced
and required fixing in reasoning.py itself before this test could pass
honestly."""

from __future__ import annotations

import random

import torch
from sarva_foundry.data.dataset import DOCUMENT_SEPARATOR
from sarva_foundry.model import DecoderOnlyTransformer, TransformerConfig
from sarva_foundry.tokenizer import ByteLevelBPETokenizer
from sarva_foundry.train import (
    SFTExample,
    Trainer,
    TrainerConfig,
    build_grpo_batch,
    build_sft_batch,
)
from sarva_foundry.train.reasoning import (
    THINK_END,
    THINK_START,
    answer_reward,
    format_reward,
    reasoning_reward,
)
from sarva_foundry.train.rl import sample_completion

N = 4
PAIRS = [(a, b) for a in range(N) for b in range(N)]


def _prompt_for(a: int, b: int) -> str:
    return f"{a} plus {b} = "


def _target_completion(a: int, b: int) -> str:
    return f"{THINK_START}{a}+{b}={a + b}{THINK_END}{a + b}"


def _tiny_config(tokenizer: ByteLevelBPETokenizer) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=tokenizer.vocab_size, dim=48, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=48
    )


def test_grpo_refines_answer_accuracy_on_top_of_cold_start_sft():
    torch.manual_seed(0)
    sft_examples = [
        SFTExample(prompt=_prompt_for(a, b), response=_target_completion(a, b)) for a, b in PAIRS
    ]

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        [e.prompt + e.response for e in sft_examples],
        vocab_size=300,
        special_tokens=[DOCUMENT_SEPARATOR, THINK_START, THINK_END],
    )
    stop_token_id = tokenizer.special_tokens[DOCUMENT_SEPARATOR]

    def decode_completion(ids: list[int]) -> str:
        if ids and ids[-1] == stop_token_id:
            ids = ids[:-1]
        return tokenizer.decode(ids)

    def measure(model: DecoderOnlyTransformer, n_trials: int = 4) -> tuple[float, float]:
        fmt_hits = ans_hits = total = 0
        for a, b in PAIRS:
            prompt_ids = tokenizer.encode(_prompt_for(a, b))
            for _ in range(n_trials):
                completion = sample_completion(
                    model, prompt_ids, 15, temperature=0.0, stop_token_id=stop_token_id
                )
                text = decode_completion(completion)
                fmt_hits += format_reward(text)
                ans_hits += answer_reward(text, str(a + b))
                total += 1
        return fmt_hits / total, ans_hits / total

    config = _tiny_config(tokenizer)
    model = DecoderOnlyTransformer(config)
    sft_trainer = Trainer(model, TrainerConfig(lr=3e-4))
    for _ in range(125):
        x, y, mask = build_sft_batch(sft_examples, tokenizer)
        sft_trainer.train_step(x, y, mask)

    fmt_before, ans_before = measure(model)
    # Cold-start SFT reliably nails the FORMAT (a fixed template) but
    # leaves real headroom on actually getting the arithmetic right --
    # both pinned directly, not just the after-GRPO improvement, so a
    # future change that accidentally makes cold-start SFT alone solve
    # the whole task (removing GRPO's headroom to demonstrate anything)
    # would be caught here specifically.
    assert fmt_before == 1.0
    assert ans_before < 0.5

    grpo_trainer = Trainer(model, TrainerConfig(lr=1e-4))
    random.seed(0)
    for step in range(400):
        a, b = PAIRS[step % len(PAIRS)]
        prompt_ids = tokenizer.encode(_prompt_for(a, b))
        completions = [
            sample_completion(model, prompt_ids, 15, temperature=1.0, stop_token_id=stop_token_id)
            for _ in range(8)
        ]
        rewards = [reasoning_reward(decode_completion(c), str(a + b)) for c in completions]
        x, y, mask = build_grpo_batch(prompt_ids, completions, pad_token_id=0)
        grpo_trainer.grpo_step(x, y, mask, rewards)

    fmt_after, ans_after = measure(model)

    assert fmt_after == 1.0, "GRPO must not degrade the format compliance cold-start SFT achieved"
    assert ans_after > ans_before + 0.15, (
        f"expected a real increase in answer accuracy, got {ans_before} -> {ans_after}"
    )


def test_a_real_reward_hacking_exploit_this_scenario_surfaced_stays_fixed():
    """Pins the exact exploit found empirically while building the
    example this test mirrors: an early, looser version of `format_
    reward`/`answer_reward` matched only the FIRST `</think>` in a
    completion, so GRPO discovered that padding a completion with many
    extra `</think>` copies (abandoning the real format entirely)
    inflated `answer_reward`'s loose "contains" check without genuinely
    answering correctly -- training runs actually converged to this
    degenerate, un-useful strategy before the fix. Both reward functions
    now require EXACTLY one `<think>`/`</think>` pair; this is the
    literal string that broke them, kept as a permanent regression
    pin."""
    degenerate_completion = "<think>2</think>5</think>5</think>5</think>5</think>3+2</think>"

    assert format_reward(degenerate_completion) == 0.0
    assert answer_reward(degenerate_completion, "5") == 0.0
    assert reasoning_reward(degenerate_completion, "5") == 0.0
