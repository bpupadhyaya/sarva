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

## Decoding has to survive more than valid input

Encoding a real string always produces ids that decode perfectly by
construction. `decode()` has a harder job: it has to handle *any* token
id sequence a model might actually generate, including ones no
`encode()` call would ever produce — an undertrained checkpoint, an RL
rollout sampling adversarially, or a genuinely corrupted checkpoint
bundle. Two real gaps in that discipline, both found by actually
constructing the exact input that breaks it, not assumed:

- **An out-of-vocabulary id crashed `decode()` with a raw `KeyError`.**
  A foundry checkpoint's `config.json` sets `vocab_size`, which sizes
  the model's own output layer — sampling can legitimately produce
  *any* id in `[0, vocab_size)` — but nothing cross-checks that value
  against what `tokenizer.json` actually serializes. A bundle where the
  two disagree (a corrupted or mismatched checkpoint) made every single
  generation from that model crash with an unhelpful numeric error,
  discarding all the valid text already generated in the same turn
  along with it. Fixed the same way an invalid UTF-8 byte sequence
  already is: emit the standard replacement character (`�`) for that
  one id and keep going, rather than raising and aborting the whole
  generation.
- **A `special_tokens` id colliding with a real vocabulary id silently
  misdecoded, with no error at all.** `decode()` checks special tokens
  first, so a colliding id always produced the special token's text
  instead of the real character — confirmed live: a real vocabulary id
  whose correct decoding was `"2"` decoded as `"<|endoftext|>"`
  instead. A legitimately-trained tokenizer can never produce this
  (`train()` always assigns special-token ids strictly after every real
  vocabulary id), so it's only reachable via a hand-edited or corrupted
  `tokenizer.json` — the same threat model as the gap above. Fixed by
  rejecting it at `load()` time with a clear error, rather than leaving
  every future `decode()` call to silently produce wrong text; already
  safely caught by `FoundryProvider.__init__`'s existing broad
  exception handling around loading a checkpoint bundle, which records
  a broken bundle and skips it rather than crashing.

Verified both the same way: reverted each fix and watched its new test
fail for the exact right reason (the raw `KeyError`; a clean pass where
a rejection should have fired) before re-applying.

**`save()` writes atomically, not via a direct `path.write_text()`.**
The same interrupted-write bug already found and fixed in `core`'s
`sarva.config`/`sarva.memory.session` had an unfixed twin here:
`write_text()` truncates the file to 0 bytes the instant it's opened,
before a single byte of new content is written, confirmed live before
fixing it. A crash mid-write destroys a previously-trained, real
tokenizer — hours of BPE merge-learning, not regenerable from the saved
file itself. Fixed via `sarva_foundry.atomic_write` — see the training
chapter for the fuller history of this bug class across `foundry`.

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
