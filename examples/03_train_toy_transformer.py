"""Example 03 — Train a toy transformer on real tokenized text.

Wires together both foundry components built so far: the byte-level BPE
tokenizer (example 02) feeds real token ids into the from-scratch
decoder-only transformer (`sarva_foundry.model`), trained for a handful of
steps on a repeating toy corpus. This is intentionally tiny (a few
thousand parameters, CPU, seconds to run) — it exists to prove the whole
pipeline (tokenize -> embed -> attend -> predict -> backprop) actually
works end to end, not to produce a useful model. See the usefulness
ladder in the design of record for what real scale requires.

Run: uv run python examples/03_train_toy_transformer.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sarva_foundry.model import DecoderOnlyTransformer, TransformerConfig
from sarva_foundry.tokenizer import ByteLevelBPETokenizer

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox is quick and the dog is lazy",
]


def main() -> None:
    torch.manual_seed(0)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(CORPUS, vocab_size=300)

    text = "the quick brown fox jumps over the lazy dog"
    ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # (1, seq_len)
    inputs, targets = ids[:, :-1], ids[:, 1:]

    generation_steps = 8
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        # Must cover the longest sequence this run will ever forward()
        # through — including the greedy-generation loop below, which
        # grows past the training sequence length one token at a time.
        max_seq_len=max(inputs.shape[1], len(tokenizer.encode("the quick")) + generation_steps),
    )
    model = DecoderOnlyTransformer(config)
    print(f"Model: {model.num_parameters():,} parameters, vocab_size={config.vocab_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(200):
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == 199:
            print(f"step {step:3d}  loss {loss.item():.4f}")

    # Greedy-decode a continuation from the first few tokens, to see the
    # tiny model actually reproduce a memorized fragment of its one
    # training sentence.
    model.eval()
    prompt_ids = tokenizer.encode("the quick")
    generated = list(prompt_ids)
    with torch.no_grad():
        for _ in range(generation_steps):
            logits = model(torch.tensor(generated).unsqueeze(0))
            next_id = int(logits[0, -1].argmax())
            generated.append(next_id)
    print(f"\nGreedy continuation of 'the quick': {tokenizer.decode(generated)!r}")


if __name__ == "__main__":
    main()
