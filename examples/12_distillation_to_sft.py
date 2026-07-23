"""Example 12 — Distillation: turning a real frontier model's answers
into foundry SFT training data.

Spec §3.6c: "synthetic-data generation (frontier-as-teacher via the
provider layer)." `sarva.distill` (core) generates `(prompt,
completion)` pairs from any real `Provider`; this example bridges that
output into `sarva_foundry.train.sft.SFTExample` with one line of glue
code per record — proving core and foundry compose cleanly at the
script level, with zero package-level dependency either direction (see
`sarva/distill.py`'s own module docstring for why that boundary is
deliberate: `core`'s and `sarva_foundry`'s `pyproject.toml`s name
completely disjoint dependency sets).

Requires a real API key — unlike every other foundry example, which
runs fully offline, distillation's entire point is calling a real
frontier model as the teacher. See the other examples for offline
demos if you don't have one yet.

Run: ANTHROPIC_API_KEY=sk-... uv run python examples/12_distillation_to_sft.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import torch
from sarva.distill import distill
from sarva.providers.anthropic_provider import AnthropicProvider
from sarva_foundry.data import DOCUMENT_SEPARATOR
from sarva_foundry.model import DecoderOnlyTransformer, TransformerConfig
from sarva_foundry.tokenizer import ByteLevelBPETokenizer
from sarva_foundry.train import SFTExample, Trainer, build_sft_batch

PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "What is 2 + 2? Answer with just the number.",
    "Name the largest planet in the solar system. One word.",
]


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run this example (see other examples for offline demos).")
        sys.exit(1)

    torch.manual_seed(0)

    print(f"Distilling {len(PROMPTS)} prompts from claude-haiku-4-5 (the teacher)...")
    provider = AnthropicProvider()
    records = await distill(PROMPTS, provider, model="claude-haiku-4-5")
    for r in records:
        print(f"  {r.prompt!r} -> {r.completion!r}")
    await provider.close()

    # The one line of glue between core and foundry this whole example
    # exists to demonstrate: a DistillationRecord becomes an SFTExample.
    sft_examples = [SFTExample(prompt=r.prompt, response=r.completion) for r in records]

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        [e.prompt + e.response for e in sft_examples],
        vocab_size=800,
        special_tokens=[DOCUMENT_SEPARATOR],
    )
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size, dim=64, n_layers=3, n_heads=4, n_kv_heads=2, max_seq_len=64
    )
    model = DecoderOnlyTransformer(config)
    trainer = Trainer(model)

    x, y, mask = build_sft_batch(sft_examples, tokenizer)
    print(f"\nTraining a {model.num_parameters():,}-parameter toy model on the teacher's answers")
    for step in range(200):
        loss = trainer.train_step(x, y, loss_mask=mask)
        if step % 50 == 0 or step == 199:
            print(f"  step {step:3d}  loss {loss:.4f}")

    print(
        "\nDistillation complete: a real frontier model's answers trained a toy "
        "model to answer the same questions -- no hand-labeling involved."
    )


if __name__ == "__main__":
    asyncio.run(main())
