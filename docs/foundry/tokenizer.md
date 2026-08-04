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

### A real bug found by fuzzing the pretokenizer, not reasoning about it: every underscore silently vanished

A round dedicated to a genuinely different METHOD — property-based
fuzzing against hundreds of varied inputs, rather than reasoning about
a specific hypothesis first — found `_PRETOKENIZE_PATTERN`'s
punctuation alternative, `[^\s\w]+` ("not whitespace, not a word
character"), silently excluded `_` from matching at all. Python's `\w`
includes underscore, so `[^\s\w]` excludes it — and the letters
alternative (`[^\W\d_]+`) *also* deliberately excludes underscore, to
isolate real letters from the rest of `\w`. Between the two, a bare
underscore matched no alternative in the whole pattern, and
`re.findall()` doesn't error on an unmatched character — it just skips
it, silently.

Confirmed live: `encode("snake_case_variable")` decoded back to
`"snakecasevariable"` — every underscore gone, with no exception, no
warning, at any layer. This corrupted more than one-off `encode()`
calls too: `train()` runs the identical pretokenizer over the training
corpus before ever counting word frequencies, so a real training run
over an underscore-containing corpus (code, identifiers, filenames)
would learn a permanently corrupted vocabulary that never even sees
the character exists. Real GPT-2 tokenization doesn't have this gap —
`\p{L}`/`\p{N}` don't include `_` either, so a real underscore run
falls through to the punctuation catch-all and gets its own token(s);
this implementation's `\w`-based approximation of that catch-all just
happened to accidentally exclude the one character it most needed to
include.

Reachable through the real product, not just the tokenizer in
isolation: `FoundryProvider.generate()` calls `tokenizer.encode()`
directly on caller-supplied prompt text from `/chat`, `/ws/chat`, or
`sarva run` — any ordinary prompt containing code, a filename, a
`snake_case` identifier, or an environment-variable name would
silently lose every underscore before the foundry model ever saw it.

Fixed by changing the punctuation alternative to `(?:[^\s\w]|_)+` —
true punctuation OR underscore, grouped the same way any other
punctuation run already is (`"a_b"` pretokenizes as `["a", "_",
"b"]`, `"__init__"` as `["__", "init", "__"]`, mirroring how
punctuation between two words is handled elsewhere in this pattern).
Verified live every repro case above now round-trips exactly, plus a
broader smoke test across mixed-case/kebab-case/repeated-underscore
inputs. Verified by reverting and watching the new test fail with the
literal old bug's own output reproducing itself
(`'ab' == 'a_b'` — false). The fuzzing pass that found this also
checked hundreds of other inputs (CJK/Arabic/emoji mixes, 200KB+
repeated-pattern strings, malformed-byte sequences, boundary vocab
sizes, thirteen kinds of malformed `decode()` input) with no other
round-trip mismatch or crash surviving verification — this was the one
real finding, not a symptom of a broader pattern.

The same fuzzing pass also found a second, smaller gap on the encode
side: `_text_to_symbols()` called `text.encode("utf-8")` strictly, so a
lone (unpaired) UTF-16 surrogate code point — `"\ud800"`, a real, legal
Python `str` value with no UTF-8 encoding — raised an uncaught
`UnicodeEncodeError` instead of encoding at all. Asymmetric with
`decode()`'s own `errors="replace"` handling just below it in the same
file. Reachable end to end: a raw JSON request body carrying an
embedded lone surrogate passes untouched through FastAPI/pydantic's
real request parsing into a plain `str` field with no validator, then
straight into `FoundryProvider.generate()`'s `tokenizer.encode()` call
— `AgentLoop`'s broad exception handling already turned this into a
clean `state=failed` rather than crashing the process, but with a raw
exception string instead of the graceful replacement behavior this
tokenizer already promises elsewhere. Fixed by matching the decode
side: `text.encode("utf-8", errors="replace")`. Verified live —
`tok.encode("hello \ud800 world")` no longer raises, and now decodes
back to `"hello ? world"` (Python's `encode(errors="replace")`
substitutes a literal `?`, unlike `decode(errors="replace")`'s U+FFFD).
Verified by reverting and watching the new test fail with the exact
old `UnicodeEncodeError`. 2 new tests total this round, 723 → 725
Python tests.

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
