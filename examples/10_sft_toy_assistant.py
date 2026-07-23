"""Example 10 — Supervised fine-tuning: turning a pretrained toy model
into a toy "assistant" that only answers, never repeats the question.

Spec §3.6e: "SFT -> DPO/RLHF -> agentic RL... this, not pretraining, is
what turns a base model into a Fable/K3-class agent." This example shows
the mechanism at toy scale: pretrain a small model on plain text (same
pipeline as example 03), then fine-tune it on (prompt, response) pairs
with `sarva_foundry.train.sft`'s loss mask -- the model is only ever
trained to predict the RESPONSE, never the prompt.

The proof that this actually worked, not just that training ran: before
SFT, greedy-decoding from any of the three questions produces the SAME
babbled continuation (whatever the plain-pretraining corpus made most
statistically likely to follow "?"), since the base model has no notion
of "answer this specific question." After SFT, greedy-decoding from
each of the three distinct prompts produces its own distinct, correct
response -- proof the model learned to condition its answer on the
actual question, not just memorize one fixed continuation.

Run: uv run python examples/10_sft_toy_assistant.py
"""

from __future__ import annotations

import torch
from sarva_foundry.data import DOCUMENT_SEPARATOR
from sarva_foundry.model import DecoderOnlyTransformer, TransformerConfig
from sarva_foundry.tokenizer import ByteLevelBPETokenizer
from sarva_foundry.train import SFTExample, Trainer, TrainerConfig, build_sft_batch

PRETRAIN_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the sky is blue and the grass is green",
    "two plus two is four",
    "the capital of France is Paris",
]

SFT_EXAMPLES = [
    SFTExample(prompt="what is two plus two? ", response="four"),
    SFTExample(prompt="what color is the sky? ", response="blue"),
    SFTExample(prompt="what is the capital of France? ", response="Paris"),
]


def _greedy_generate(
    model: DecoderOnlyTransformer, tokenizer: ByteLevelBPETokenizer, prompt: str, max_new: int = 6
) -> str:
    model.eval()
    ids = list(tokenizer.encode(prompt))
    eot_id = tokenizer.encode(DOCUMENT_SEPARATOR)[0]
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(torch.tensor(ids).unsqueeze(0))
            next_id = int(logits[0, -1].argmax())
            if next_id == eot_id:
                break
            ids.append(next_id)
    return tokenizer.decode(ids[len(tokenizer.encode(prompt)) :])


def main() -> None:
    torch.manual_seed(0)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        PRETRAIN_CORPUS + [e.prompt + e.response for e in SFT_EXAMPLES],
        vocab_size=500,
        special_tokens=[DOCUMENT_SEPARATOR],
    )

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size, dim=64, n_layers=3, n_heads=4, n_kv_heads=2, max_seq_len=64
    )
    model = DecoderOnlyTransformer(config)
    print(f"Model: {model.num_parameters():,} parameters")

    # --- Stage 1: plain pretraining on the corpus (same as example 03) ---
    from sarva_foundry.data import tokenize_corpus

    token_ids = tokenize_corpus(PRETRAIN_CORPUS, tokenizer)
    pretrain_x = torch.tensor(token_ids[:-1]).unsqueeze(0)
    pretrain_y = torch.tensor(token_ids[1:]).unsqueeze(0)

    trainer = Trainer(model, TrainerConfig(lr=3e-3))
    print("\nStage 1 -- pretraining on plain text...")
    for step in range(150):
        loss = trainer.train_step(pretrain_x, pretrain_y)
        if step % 50 == 0 or step == 149:
            print(f"  step {step:3d}  loss {loss:.4f}")

    print("\nBefore SFT, greedy continuation of a question (no fine-tuning yet):")
    for example in SFT_EXAMPLES:
        print(f"  {example.prompt!r} -> {_greedy_generate(model, tokenizer, example.prompt)!r}")

    # --- Stage 2: SFT on (prompt, response) pairs, masked loss ---
    x, y, mask = build_sft_batch(SFT_EXAMPLES, tokenizer)
    print(f"\nStage 2 -- SFT on {len(SFT_EXAMPLES)} (prompt, response) pairs...")
    for step in range(200):
        loss = trainer.train_step(x, y, loss_mask=mask)
        if step % 50 == 0 or step == 199:
            print(f"  step {step:3d}  masked loss {loss:.4f}")

    print("\nAfter SFT, greedy continuation of the same prompts:")
    for example in SFT_EXAMPLES:
        generated = _greedy_generate(model, tokenizer, example.prompt)
        match = "matches" if generated.strip() == example.response else "differs from"
        print(f"  {example.prompt!r} -> {generated!r} ({match} target {example.response!r})")


if __name__ == "__main__":
    main()
