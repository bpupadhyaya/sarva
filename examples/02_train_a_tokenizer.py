"""Example 02 — Train a byte-level BPE tokenizer from scratch.

The first foundry component: no HuggingFace `tokenizers`, no `tiktoken` —
just the algorithm, in plain Python, trained on a tiny local corpus. Shows
the byte-level guarantee (unseen Unicode/emoji still round-trips exactly,
with no <unk>) and how learned merges compress a sentence from the corpus
versus raw byte-level encoding.

Run: uv run python examples/02_train_a_tokenizer.py
"""

from __future__ import annotations

from sarva_foundry.tokenizer import ByteLevelBPETokenizer

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox is quick and the dog is lazy",
    "she sells seashells by the seashore",
    "how much wood would a woodchuck chuck if a woodchuck could chuck wood",
]


def main() -> None:
    raw = ByteLevelBPETokenizer()  # untrained: pure byte-level, no merges
    tok = ByteLevelBPETokenizer()
    tok.train(CORPUS, vocab_size=300, special_tokens=["<|endoftext|>"])

    print(f"Trained vocab size: {tok.vocab_size} ({len(tok.merges)} learned merges)")

    sentence = "the quick brown fox"
    raw_ids = raw.encode(sentence)
    trained_ids = tok.encode(sentence)
    print(f"\n{sentence!r}")
    print(f"  byte-level tokens: {len(raw_ids)}")
    print(f"  trained tokens:    {len(trained_ids)}  {trained_ids}")

    # The byte-level guarantee: text the tokenizer never saw during
    # training — including a different script and emoji — still
    # round-trips losslessly. There is no <unk>.
    unseen = "héllo wörld —日本語 🎉🚀"
    ids = tok.encode(unseen)
    decoded = tok.decode(ids)
    print(f"\nUnseen text: {unseen!r}")
    print(f"Round-trips: {decoded == unseen} ({len(ids)} tokens)")


if __name__ == "__main__":
    main()
