"""Example 11 — Direct Preference Optimization: teaching a toy model to
prefer one response over another, without a reward model or RL rollouts.

Spec §3.6e: "SFT -> DPO/RLHF -> agentic RL." DPO (Rafailov et al. 2023)
is the second post-training step -- example 10 showed SFT teaching a
model to answer at all; this example shows a further round of training
that shifts the model's preference between two responses it could
already produce, using nothing but which one a human preferred.

The setup: start from an SFT'd base model that can already answer a
question two different ways with roughly comparable likelihood (an
overly formal answer and a friendly one). DPO training on a single
preference pair -- "the friendly answer is chosen, the formal one is
rejected" -- should measurably shift the model's relative preference
toward the friendly answer, without a reward model, without sampling
rollouts, and without ever computing an explicit reward at all.

Run: uv run python examples/11_dpo_preference_tuning.py
"""

from __future__ import annotations

import copy

import torch
from sarva_foundry.data import DOCUMENT_SEPARATOR
from sarva_foundry.model import DecoderOnlyTransformer, TransformerConfig
from sarva_foundry.tokenizer import ByteLevelBPETokenizer
from sarva_foundry.train import DPOExample, SFTExample, Trainer, build_dpo_batch, build_sft_batch
from sarva_foundry.train.dpo import sequence_logprobs

PROMPT = "how do I say hello? "
CHOSEN = "hey there"
REJECTED = "greetings to you"

# SFT on both responses first, so the base model can already produce
# either one -- DPO's job is to shift PREFERENCE between two things the
# model can already say, not to teach it a brand-new capability (that
# was SFT's job in example 10).
SFT_EXAMPLES = [
    SFTExample(prompt=PROMPT, response=CHOSEN),
    SFTExample(prompt=PROMPT, response=REJECTED),
]


def _preference_margin(model, chosen_batch, rejected_batch) -> float:
    model.eval()
    with torch.no_grad():
        chosen_lp = sequence_logprobs(model, *chosen_batch)
        rejected_lp = sequence_logprobs(model, *rejected_batch)
    return (chosen_lp - rejected_lp).item()


def main() -> None:
    torch.manual_seed(0)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        [PROMPT + CHOSEN, PROMPT + REJECTED], vocab_size=300, special_tokens=[DOCUMENT_SEPARATOR]
    )

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size, dim=48, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32
    )
    model = DecoderOnlyTransformer(config)

    print("Stage 1 -- SFT on both responses (so the model can produce either one)...")
    trainer = Trainer(model)
    x, y, mask = build_sft_batch(SFT_EXAMPLES, tokenizer)
    for step in range(150):
        loss = trainer.train_step(x, y, loss_mask=mask)
        if step % 50 == 0 or step == 149:
            print(f"  step {step:3d}  loss {loss:.4f}")

    dpo_example = [DPOExample(prompt=PROMPT, chosen=CHOSEN, rejected=REJECTED)]
    chosen_batch, rejected_batch = build_dpo_batch(dpo_example, tokenizer)

    margin_before = _preference_margin(model, chosen_batch, rejected_batch)
    print(f"\nPreference margin after SFT alone: {margin_before:+.3f} (log-prob units)")
    print("(chosen minus rejected sequence log-probability -- positive means already preferred)")

    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad = False

    print(f"\nStage 2 -- DPO: '{CHOSEN}' is chosen over '{REJECTED}' for the same prompt...")
    for step in range(100):
        loss = trainer.dpo_step(ref_model, chosen_batch, rejected_batch, beta=0.1)
        if step % 25 == 0 or step == 99:
            print(f"  step {step:3d}  dpo loss {loss:.4f}")

    margin_after = _preference_margin(model, chosen_batch, rejected_batch)
    print(f"\nPreference margin after DPO: {margin_after:+.3f} (log-prob units)")
    print(
        f"Margin moved by {margin_after - margin_before:+.3f} in the direction DPO was "
        "trained to push it -- no reward model, no rollouts, just the preference pair."
    )


if __name__ == "__main__":
    main()
