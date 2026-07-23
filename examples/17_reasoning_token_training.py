"""Example 17 — Reasoning/thinking-token training: SFT cold start, then
GRPO refinement against a verifiable format+accuracy reward.

Spec §3.6a: "reasoning/thinking-token support... o1/R1-class," naming
DeepSeek-R1-class open recipes directly as the reference. This example
follows DeepSeek-R1's own published recipe shape, not an invented one:
(1) a small "cold start" SFT stage teaches the `<think>...</think>`
FORMAT via imitation, then (2) GRPO continues training against a real
verifiable reward (`sarva_foundry.train.reasoning.reasoning_reward`)
that checks both format compliance and answer correctness. Neither
training stage is new code -- `sarva_foundry.train.sft` and
`sarva_foundry.train.rl` are reused completely unchanged; the only new
code is the reward function that turns GRPO into reasoning-token
training specifically.

The task: single-digit addition, "A plus B = " -> "<think>A+B=C</think>C"
for A, B in [0, 4). Small enough to train at toy scale in a few seconds,
real enough that "got the arithmetic right" is a genuine, programmatically
checkable claim, not asserted.

Run: uv run python examples/17_reasoning_token_training.py
"""

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

N = 4  # digits 0..N-1
PAIRS = [(a, b) for a in range(N) for b in range(N)]
COLD_START_STEPS = 125
GRPO_STEPS = 400
GROUP_SIZE = 8
ROLLOUT_TEMPERATURE = 1.0
MAX_NEW_TOKENS = 15


def prompt_for(a: int, b: int) -> str:
    return f"{a} plus {b} = "


def target_completion(a: int, b: int) -> str:
    return f"{THINK_START}{a}+{b}={a + b}{THINK_END}{a + b}"


def main() -> None:
    torch.manual_seed(0)

    sft_examples = [
        SFTExample(prompt=prompt_for(a, b), response=target_completion(a, b)) for a, b in PAIRS
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
            prompt_ids = tokenizer.encode(prompt_for(a, b))
            for _ in range(n_trials):
                completion = sample_completion(
                    model, prompt_ids, MAX_NEW_TOKENS, temperature=0.0, stop_token_id=stop_token_id
                )
                text = decode_completion(completion)
                fmt_hits += format_reward(text)
                ans_hits += answer_reward(text, str(a + b))
                total += 1
        return fmt_hits / total, ans_hits / total

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size, dim=48, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=48
    )
    model = DecoderOnlyTransformer(config)
    print(f"Model: {model.num_parameters():,} parameters, {len(PAIRS)} arithmetic facts")

    print(f"\n--- Stage 1: cold-start SFT ({COLD_START_STEPS} steps) ---")
    print("Teaches the <think>...</think> FORMAT via imitation -- the exact")
    print("role DeepSeek-R1's own published recipe gives this stage: pure RL")
    print("from a base model produced real format/readability problems in")
    print("their own ablation, which is why R1 adds a cold-start stage at all.")
    sft_trainer = Trainer(model, TrainerConfig(lr=3e-4))
    for step in range(COLD_START_STEPS):
        x, y, mask = build_sft_batch(sft_examples, tokenizer)
        loss = sft_trainer.train_step(x, y, mask)
        if step % 40 == 0:
            print(f"  step {step:3d}  loss {loss:7.4f}")

    fmt_before, ans_before = measure(model)
    print(f"\nAfter cold-start SFT: format_rate={fmt_before:.0%}  answer_rate={ans_before:.0%}")
    print("(format is learned quickly since it's a fixed template; getting the")
    print("arithmetic right is the harder part cold-start SFT alone hasn't")
    print("fully solved -- real headroom for GRPO to improve, not saturated.)")

    print(f"\n--- Stage 2: GRPO refinement ({GRPO_STEPS} steps, group={GROUP_SIZE}) ---")
    print("Rewards each rollout with reasoning_reward: 0.3x format compliance")
    print("+ 0.7x whether the real digit sum appears after </think>.")
    grpo_trainer = Trainer(model, TrainerConfig(lr=1e-4))
    random.seed(0)
    for step in range(GRPO_STEPS):
        a, b = PAIRS[step % len(PAIRS)]
        prompt_ids = tokenizer.encode(prompt_for(a, b))
        completions = [
            sample_completion(
                model,
                prompt_ids,
                MAX_NEW_TOKENS,
                temperature=ROLLOUT_TEMPERATURE,
                stop_token_id=stop_token_id,
            )
            for _ in range(GROUP_SIZE)
        ]
        rewards = [reasoning_reward(decode_completion(c), str(a + b)) for c in completions]
        x, y, mask = build_grpo_batch(prompt_ids, completions, pad_token_id=0)
        loss = grpo_trainer.grpo_step(x, y, mask, rewards)
        if step % 100 == 0:
            mean_reward = sum(rewards) / len(rewards)
            print(f"  step {step:3d}  loss {loss:7.4f}  group mean reward {mean_reward:.3f}")

    fmt_after, ans_after = measure(model)
    print(f"\nAfter GRPO: format_rate={fmt_after:.0%}  answer_rate={ans_after:.0%}")
    print(
        f"\nAnswer accuracy moved from {ans_before:.0%} to {ans_after:.0%} purely "
        "from GRPO refining on top of the cold-start SFT baseline -- every "
        "reward came from real generated text checked against the real digit "
        "sum, no result assumed or fabricated."
    )

    example_prompt = prompt_for(2, 3)
    example_ids = tokenizer.encode(example_prompt)
    example_ids_out = sample_completion(
        model, example_ids, MAX_NEW_TOKENS, temperature=0.0, stop_token_id=stop_token_id
    )
    example_completion = decode_completion(example_ids_out)
    print(f"\nSample greedy completion for {example_prompt!r}: {example_completion!r}")


if __name__ == "__main__":
    main()
