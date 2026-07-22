# Pretraining, with resume that actually resumes

`sarva_foundry.data` and `sarva_foundry.train` — a corpus-to-batches
pipeline and a training loop with checkpoint/resume (design of record
§3.6c/§3.6d, the single-process slice of both).

## Sourcing: load, dedup, filter

`sarva_foundry.data.corpus` is the sourcing/cleaning/dedup slice of
§3.6c, at the scale this project can actually run and test today: a
local directory of text files, not a Common Crawl-scale pipeline.
`load_text_files` reads a directory's files as one document each
(sorted, for deterministic ordering, and raising rather than silently
skipping a file it can't decode). `dedup_documents` drops exact
duplicates by content hash, keeping first-occurrence order —
near-duplicate detection (minhash/simhash, catching two documents that
differ by a sentence or a timestamp) is real, separate scope, named
rather than silently assumed covered. `filter_by_length` drops documents
outside a `[min_chars, max_chars]` range — the crudest real quality
filter (too-short is usually navigation/boilerplate junk, too-long is
often scrape garbage), and the one every larger pipeline layers richer
heuristics on top of, not a replacement for them.

These three stages compose directly into the tokenize/chunk pipeline
below: `load_text_files → dedup_documents → filter_by_length →
tokenize_corpus → TextChunkDataset`, verified as a real pipeline (not
three functions that happen to share a module) in
`tests/foundry/test_corpus.py`.

### Near-duplicate detection: the scope `dedup_documents` deferred

`dedup_documents` only catches byte-identical documents. Real corpora
have near-duplicates too — a re-published article with one word edited,
a scraped page with a different timestamp — and `sarva_foundry.data.near_dedup.dedup_near_duplicates`
catches those via MinHash: each document's character-shingle set is
reduced to a fixed-size signature (one minimum hash value per hash
function), and the fraction of matching signature positions between two
documents' signatures estimates their true Jaccard similarity without
ever materializing and comparing full shingle sets pairwise. Implemented
from the underlying hashing (`hashlib.sha256`, salted per hash
function), not vendored from an external minhash library.

Worth recording honestly: the first draft of this module's tests
assumed a "near-duplicate" meant appending a whole extra sentence to a
document. Empirically, that dilutes shingle-set Jaccard similarity far
more than intuition suggests (~0.66 true similarity for a realistic
document length — well below any reasonable dedup threshold). A real
near-duplicate — a small in-place edit — scores much higher (~0.85).
The *implementation* was correct throughout; the test's assumption about
what "near-duplicate" looks like in shingle-similarity terms was wrong,
caught by actually computing the true Jaccard similarity for the test
documents chosen rather than assuming a threshold would obviously pass.

## The dataset: concatenate, then chunk

`tokenize_corpus` encodes every document in a corpus and concatenates
them into one token stream, inserting a `<|endoftext|>` separator between
documents so the model can learn document boundaries instead of treating
unrelated documents as one continuous story. `TextChunkDataset` then
slices that stream into fixed-length `(input, target)` pairs, where
`target` is `input` shifted right by one token — the standard
next-token-prediction framing. This "concatenate and chunk" approach is
what real pretraining pipelines use to avoid wasting compute on padding,
not a simplified stand-in for it.

## The trainer: checkpointing that's actually correct

`Trainer.save_checkpoint`/`load_checkpoint` exist because a training run
that can't resume loses all its compute on any crash, preemption, or
intentional pause. The subtle part: bit-identical resume requires saving
**optimizer state**, not just model weights. AdamW tracks per-parameter
momentum and variance estimates (`exp_avg`, `exp_avg_sq`) that evolve
over training — a checkpoint that only restores weights silently restarts
that momentum from zero, which trains *differently* from the run it
claims to resume, with no exception to catch the difference. It would
still "work" in the sense of not crashing, while quietly not being what
it claims to be.

`tests/foundry/test_trainer.py` verifies this directly with two
paired tests:

1. **The positive test** trains 10 steps two ways — uninterrupted, and as
   5 steps → checkpoint → fresh `Trainer` loaded from checkpoint → 5 more
   steps — and asserts the final model weights are bit-identical (within
   float tolerance) between the two paths.
2. **The negative control** repeats the interrupted path but swaps in a
   *fresh* optimizer after loading (the exact bug the module's docstring
   warns about) and asserts the result **diverges** from uninterrupted
   training. Without this control, a passing positive test wouldn't
   prove much — the toy task could just happen to converge to the same
   point regardless of optimizer state. The negative control is what
   makes the positive test meaningful.

## The learning-rate schedule: warmup, then cosine decay

`WarmupCosineSchedule` replaces what was originally a flat learning
rate — a real limitation named honestly in an earlier entry, not
silently left in place. A flat LR risks instability right at the
model's random initialization (no warmup) and leaves quality on the
table by never converging into a sharper minimum at the end of training
(no decay). Warmup + cosine decay is the shape essentially every real
pretraining run uses, from GPT-2 onward.

The implementation is a pure function of step count — `lr_at(step)` —
rather than mutable schedule state. `Trainer.train_step` calls it fresh
on every step, which means the existing checkpoint/resume machinery
(which already restores `self.step`) resumes the LR curve correctly
*for free*: there's no separate schedule state that could drift out of
sync with the checkpointed step count, because there's no separate
state at all. `tests/foundry/test_trainer.py`'s
`test_checkpoint_resume_is_bit_identical_with_a_schedule_active` verifies
this directly — resuming mid-schedule must continue the LR curve from
exactly where it left off, not restart warmup or jump to some other
point on it.

## Try it

```bash
uv run python examples/04_pretrain_and_resume.py
```

Runs the full pipeline built so far — tokenizer → dataset →
transformer → trainer, with a warmup+cosine LR schedule — on a toy
corpus: 30 training steps, a checkpoint save, then a *fresh* model and
trainer resuming from that checkpoint for 30 more steps. Watch the
printed loss and LR columns: loss keeps descending smoothly across the
checkpoint boundary instead of spiking back up (momentum survived the
round-trip), and the LR keeps decaying smoothly too instead of resetting
to the warmup value (the schedule resumed from the checkpointed step
count).

### Provenance and license tracking

`sarva_foundry.data.provenance.SourcedDocument` carries a document's
source path and license through the same load → dedup → filter →
near-dedup stages as the plain-string pipeline above, for callers who
need to know *where* a training document came from and *what license it
carries* — required if this project's docs are ever going to state
honestly what a trained model was actually trained on.

The design choice worth naming: `sarva_foundry.data.corpus`/`near_dedup`
stay exactly as they were — plain `list[str]` in, plain `list[str]` out,
untouched and still the simplest path for callers who don't need
tracking. Provenance is a separate, thin layer built on the *same*
tested logic, not a rewrite: `_dedup_by_key`, `_filter_by_length_key`,
and `_dedup_near_duplicates_by_key` are generic over a `key` extractor,
so `dedup_documents(docs)` and `dedup_sourced_documents(docs)` call the
identical underlying function — one keyed on `lambda d: d`, the other on
`lambda d: d.text`. This matters for a reason beyond code reuse: naively
running the string-based pipeline and then trying to guess which
`SourcedDocument` each surviving string came from breaks the moment two
*different* source files happen to contain identical text — exactly the
case `dedup_sourced_documents`'s own test exists to pin (two source
files, byte-identical content: the correct behavior is dropping the
second file while keeping the *first* file's provenance, not an
ambiguous or arbitrary choice).

`load_text_files_with_provenance` applies one `license` string uniformly
to every file loaded in a single call — real per-file license variation
within one directory (a manifest mapping path → license) is real,
separate scope, named rather than silently assumed covered.

## What's next

Web/code/books/math-scale corpus sourcing and mixing recipes (local
files, exact + near-duplicate dedup, length filtering, and provenance/license
tracking all exist now — larger-scale sourcing and per-file license
manifests don't yet; nor does an LSH banding index, which near-duplicate
dedup would need to scale past the current O(kept²) pairwise
comparison), and the distributed training slice of §3.6d (FSDP → 3D
parallelism, loss-spike handling, scaling-law tooling) once a model
worth training at that scale exists.
