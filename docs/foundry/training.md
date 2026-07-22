# Pretraining, with resume that actually resumes

`sarva_foundry.data` and `sarva_foundry.train` — a corpus-to-batches
pipeline and a training loop with checkpoint/resume (design of record
§3.6c/§3.6d, the single-process slice of both).

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

## Try it

```bash
uv run python examples/04_pretrain_and_resume.py
```

Runs the full pipeline built so far — tokenizer → dataset →
transformer → trainer — on a toy corpus: 30 training steps, a checkpoint
save, then a *fresh* model and trainer resuming from that checkpoint for
30 more steps. Watch the printed loss: it keeps descending smoothly
across the checkpoint boundary instead of spiking back up, which is the
visible sign that momentum survived the round-trip.

## What's next

Real corpus sourcing (web/code/books/math, cleaning, dedup, quality
filtering, mixing recipes — the actual content of §3.6c beyond the
chunking mechanism here), and the distributed-training slice of §3.6d
(FSDP → 3D parallelism, loss-spike handling, scaling-law tooling) once a
model worth training at that scale exists.
