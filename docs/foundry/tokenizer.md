# The tokenizer, from scratch

`sarva_foundry.tokenizer.ByteLevelBPETokenizer` — the first component of
the foundry track (Part VI: "Where Intelligence Comes From"). No
HuggingFace `tokenizers`, no `tiktoken`: this is the same family of
algorithm GPT-2/GPT-4 use, implemented from first principles so the whole
thing is readable in one file (`foundry/sarva_foundry/tokenizer/bpe.py`).

## Why byte-level

A word-level or character-level tokenizer needs an `<unk>` token for
anything outside its training vocabulary — a real problem for a
multilingual, multimodal tool. Byte-level BPE sidesteps this entirely:
every possible byte value (0–255) gets a dedicated symbol in the base
vocabulary, so *any* UTF-8 text — including scripts, emoji, and code the
tokenizer never saw during training — decomposes into bytes it already
knows and round-trips losslessly. There is no `<unk>` by construction.

## How training works

1. **Pretokenize.** Split text into word-ish chunks (roughly: contractions,
   runs of letters, runs of digits, runs of punctuation, whitespace) so BPE
   merges don't cross word boundaries in ways that would hurt generalization.
2. **Map to byte-symbols.** Each chunk's UTF-8 bytes are remapped 1:1 to a
   dedicated printable Unicode character, so the rest of the algorithm just
   operates on ordinary strings.
3. **Count pairs, merge, repeat.** Count every adjacent symbol pair across
   the whole corpus (frequency-weighted by how often each word occurs),
   merge the most frequent pair into a new symbol, record that merge rule,
   and repeat until the vocabulary reaches the requested size.

Encoding replays the learned merges greedily, in the order they were
trained (earliest-learned merge wins when multiple pairs in a word are
mergeable). Decoding is the exact inverse: token ids → symbols →
concatenate → map back to raw bytes → UTF-8 decode.

## Try it

```bash
uv run python examples/02_train_a_tokenizer.py
```

Trains on a four-sentence toy corpus in well under a second, and prints
both the compression from learned merges (`"the quick brown fox"` drops
from 19 byte-level tokens to 4) and a round-trip check on text the
tokenizer never saw — a different script plus emoji — to demonstrate the
byte-level guarantee directly.

## What's next

The tokenizer is one piece of §3.6 in the design of record (data
pipelines, model architecture, pretraining, post-training, inference,
evals). It's sequenced first because everything downstream — the model,
the training loop — needs token ids to operate on.
