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

**`save_checkpoint` writes atomically, not via a direct `torch.save(obj,
path)`.** The same interrupted-write bug already found and fixed in
`core`'s `sarva.config`/`sarva.memory.session` (see the memory chapter
in the core docs) had an unfixed twin here, with a materially worse
blast radius: confirmed live by truncating a real, trained checkpoint
mid-write to simulate a crash, then calling `load_checkpoint` on it —
`RuntimeError: PytorchStreamReader failed reading zip archive: failed
finding central directory`. A crash at exactly the wrong moment
destroys not just the new save but the previously-good checkpoint that
was there before, i.e. real GPU-hours of training progress, not just a
config file. Fixed via `sarva_foundry.atomic_write` — a mirrored, not
shared, copy of `core`'s equivalent helper, since `core` and
`sarva_foundry` intentionally share no dependency in either direction.
`ByteLevelBPETokenizer.save` (see the tokenizer chapter) and the
checkpoint bundle's `config.json` write in `sarva.providers.
foundry_provider.save_checkpoint_bundle` had the identical unfixed bug
and got the identical fix.

**The mirrored `atomic_write_text` never actually finished mirroring
its `core` counterpart — it was missing the one line that makes the
rename durable.** A much later fresh-eyes sweep, checking this exact
"mirrored, not shared" claim against the real `core` source rather than
trusting the docstring, found `sarva_foundry.atomic_write.
atomic_write_text` still wrote its temp file via a plain
`Path.write_text()`, with no `os.fsync()` — while `core.sarva.
atomic_write.atomic_write_text` (via `atomic_write_bytes`) has always
called `fsync` before the rename. The temp-file+rename pattern already
fixes the truncate-on-open scenario named above, but without `fsync`
the write isn't durable against an OS-level crash or power loss (a
laptop battery dying, a kernel panic, an unclean container shutdown —
realistic for this project's own laptop-scale training story) between
the write returning and the kernel actually flushing dirty pages to
disk: the standard "rename() without fsync() can still leave a
zero-length or garbage file after a crash" filesystem gotcha. Confirmed
live: monkeypatching `os.fsync` to count calls showed the old version
of `atomic_write_text` never reached it at all, while the `core`
sibling calls it exactly once per write. Fixed by rewriting
`atomic_write_text` to go through `os.open`/`os.fdopen` with an
explicit `flush()` + `fsync()`, matching `core`'s `atomic_write_bytes`
exactly — including gaining the same `mode` parameter `core`'s version
already has, for parity, not because anything in `foundry` currently
needs a non-default mode. Verified by reverting and watching both new
tests fail with the literal old bug's own shape: zero `fsync` calls,
and a `TypeError` for the `mode` keyword that didn't exist yet. 2 new
tests, 857 → 859 Python tests.

**`TrainerConfig.grad_clip` was never validated -- a negative value
silently turned every training step into gradient ASCENT for the rest
of the run, the most severe direction any bug in this package can
corrupt training in.** A much later fresh-eyes sweep found `torch.nn.
utils.clip_grad_norm_`'s own scaling math (`clip_coef = max_norm /
(total_norm + eps)`, applied unconditionally whenever it's below 1)
has no sign or finiteness guard of its own, and `grad_clip` was the
one field in `TrainerConfig` with no `__post_init__` check at all.
Confirmed live: `clip_grad_norm_(params, max_norm=-1.0)` on a real
gradient `[3.0, 4.0]` returned `[-0.6, -0.8]` -- not merely zeroed or
left unclipped, the gradient's *sign* was flipped. `train_step`/
`dpo_step`/`grpo_step` all apply this identically and unconditionally
whenever `grad_clip is not None`, so a negative value would silently
increase the loss instead of decreasing it on every single step across
pretraining, SFT, DPO, and GRPO alike -- not a crash, not a stalled
metric, active corruption a loss curve climbing steadily could easily
be mistaken for "learning rate too high" rather than a sign error one
config field away. `grad_clip=nan` is the identical NaN-poisoning shape
already fixed for `norm_eps`/`RopeScalingConfig.factor` elsewhere in
this package (confirmed live: every gradient becomes NaN);
`grad_clip=0.0` confirmed live too, silently zeroing every gradient so
every subsequent step becomes a complete no-op with no error, wasting
real compute for the rest of the run. Reachable the same way every
other config field in this package is -- a caller deriving `grad_clip`
programmatically (e.g. from a budget that can go non-positive) or a
simple sign-flip typo in a training script, not an adversarial input.
Fixed with `TrainerConfig`'s first `__post_init__`: `grad_clip` must be
a finite positive number, or `None` (still allowed, meaning "no
clipping," unchanged). Verified by reverting and watching the new test
fail with the literal old bug's own shape: `DID NOT RAISE ValueError`
for a negative, zero, and NaN `grad_clip`. 1 new test, 876 → 877
Python tests.

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
needed a manifest, which `load_text_files_from_manifest` now provides:
a JSON file mapping each document's path to its own license string,
paths resolved *relative to the manifest's own directory* so the
manifest travels with its corpus without needing path edits. It
validates every entry rather than trusting it: a missing file, a
malformed manifest, or an entry that resolves outside the manifest's own
directory all raise clearly. That last check matters for a reason beyond
tidiness — `Path("/safe/dir") / "/etc/passwd"` is a genuine pathlib
gotcha: joining an absolute path onto a base silently *discards* the
base rather than erroring, so a manifest entry that's accidentally (or
maliciously) absolute would otherwise read a file nowhere near the
corpus. The check validates the final *resolved* path against the
manifest's directory, not the raw string, so it catches this case and
plain `"../"` traversal alike.

### Try it on real data

Every example above trains on four hardcoded toy sentences — enough to
prove the mechanics, not that the corpus-sourcing pipeline does anything
useful on real text.

```bash
uv run python examples/06_real_corpus_pretraining.py
```

Fetches three short, genuinely public-domain texts from Project
Gutenberg (*A Modest Proposal*, *The Hunting of the Snark*, *The Time
Machine* — small on purpose, this is a laptop-scale demo, not a training
run meant to produce a useful model), runs them through the real
pipeline — `load_text_files_with_provenance` → exact-dedup → near-dedup
→ length-filter — with a real, honestly-stated license
(`"Public Domain (Project Gutenberg, US)"`) attached to every surviving
document, then trains the same tokenizer/transformer/`Trainer` stack the
toy example above exercises, now on ~90K real tokens instead of a
few dozen. Requires network access for the download step only —
everything after that (dedup, tokenizer training, model training) is
fully offline, same as every other example.

## Supervised fine-tuning: turning a base model into an assistant

§3.6e: "SFT -> DPO/RLHF -> agentic RL... this, not pretraining, is what
turns a base model into a Fable/K3-class agent." SFT is the first piece
of that line, and it needed no new trainer — `Trainer.train_step` gained
one optional parameter, `loss_mask`, and that's the entire difference
between pretraining and SFT here: same optimizer, same warmup+cosine
schedule, same bit-identical checkpoint/resume, just a masked loss
instead of an unmasked one. `loss_mask=None` (the default, and every
call site before this existed) is exactly the original behavior — a
regression test confirms it's bit-identical, not just "close."

`sarva_foundry.train.sft` builds that mask from `(prompt, response)`
pairs: `encode_sft_example` tokenizes prompt then response then an
`end_of_turn` marker (reusing `DOCUMENT_SEPARATOR`, the same boundary
token plain pretraining uses between documents, rather than inventing a
second special token for the same purpose), with `loss_mask[i] == 1`
iff position `i` is part of the response. `build_sft_batch` pads a batch
to its longest example and shifts for next-token prediction — right-
padding is safe under causal attention *by construction*, not by
convention: a padded position can never influence an earlier position's
output (already guaranteed by the causal mask), and its own output is
excluded from the loss via the mask.

**The property that actually matters, and what the tests check
directly:** two training batches whose targets differ *only* at
masked-out (prompt) positions must produce bit-identical loss —
`test_loss_mask_makes_masked_target_values_irrelevant_to_the_loss`
proves the masked positions genuinely don't contribute, not just that
the returned loss looks reasonable. The complementary test confirms
changing an *unmasked* target does change the loss, so the mask can't
trivially "pass" by excluding everything (which would make SFT a no-op
instead of actually training the response).

`examples/10_sft_toy_assistant.py` runs the full two-stage pipeline: a
plain-pretrained toy model babbles the *same* generic continuation for
every question it's asked (no notion yet of "answer this specific
question"); after SFT on three `(prompt, response)` pairs, greedy-
decoding from each of the three distinct prompts produces its own
distinct, correct response — proof the model learned to condition its
answer on the actual question, not just memorize one fixed
continuation.

**A real bug found by actually calling `build_sft_batch([], ...)`:**
`max(len(ids) for ids, _ in encoded)` on the (then-empty) `encoded`
list raised a bare `ValueError: max() iterable argument is empty` --
technically the right exception type, but a confusing message naming
an internal `max()` call the caller never wrote, not the actual problem
(no examples at all). Lower severity than most fixes in this project:
no CLI or data pipeline currently wires this function to external
input that could plausibly filter a batch down to zero examples --
it's exercised only by unit tests today -- but any future caller that
does (e.g. a data pipeline filtering low-quality examples out) would
hit this same confusing error. Fixed with an explicit check raising
`ValueError("build_sft_batch requires at least one example, got an
empty list")` before the `max()` call ever runs. `build_dpo_batch`
(below) inherits the same clear message for free, since it just calls
`build_sft_batch` twice rather than reimplementing this. Verified the
new tests are real: reverted the fix and watched both fail with the
original confusing `max()` message before re-applying.

## DPO: teaching preference without a reward model

§3.6e's post-training line continues: "SFT -> DPO/RLHF -> agentic RL."
Direct Preference Optimization (Rafailov et al. 2023) is the second
step. Where SFT teaches a model to produce a specific response at all,
DPO teaches it to *prefer* one response over another for the same
prompt — using nothing but which one was chosen, no reward model, no RL
rollouts. The paper's central insight: the reward model an RLHF pipeline
would ordinarily train first has a closed form directly in terms of the
policy, so preference pairs can train the policy directly:

```
L_DPO = -log sigmoid(
    beta * [ (log pi(y_w|x) - log ref(y_w|x))
           - (log pi(y_l|x) - log ref(y_l|x)) ]
)
```

`y_w`/`y_l` are the chosen ("winning") and rejected ("losing") responses
to the same prompt `x`; `pi` is the policy being trained; `ref` is a
frozen reference model (in practice, the SFT checkpoint DPO starts
from) that keeps the policy from drifting arbitrarily far just to
satisfy one preference pair.

`sarva_foundry.train.dpo.build_dpo_batch` reuses `sarva_foundry.train.
sft.build_sft_batch` rather than a parallel encoding path — a DPO
preference triple `(prompt, chosen, rejected)` is exactly two SFT-shaped
`(prompt, response)` pairs sharing one prompt, so `build_dpo_batch`
calls `build_sft_batch` twice instead of reimplementing tokenization,
padding, and loss-mask construction. `Trainer.dpo_step` is a new method
rather than another `train_step` parameter, since DPO genuinely needs
four forward passes (policy × {chosen, rejected}, reference ×
{chosen, rejected}) instead of `train_step`'s one — but it shares the
same optimizer, gradient clipping, and step counting.

**A known, exact numeric fixed point, not just a plausible-looking
number:** when the policy is identical to the reference model (true at
the very start of DPO training, before any update), the log-ratio terms
for chosen and rejected are identical, so the loss is exactly
`-log(sigmoid(0)) = ln(2) ≈ 0.6931` — not approximately, exactly, a
direct consequence of the formula. `test_dpo_step_initial_loss_is_exactly_ln2_when_policy_equals_reference`
checks this on the full `dpo_step` path (real model forward passes, not
an isolated-tensor version of the formula), which is a far stronger
correctness check than "the loss is some finite, reasonable-looking
number."

Two more properties worth naming: `test_dpo_step_never_puts_a_gradient_on_the_reference_model`
confirms the reference model's forward pass genuinely runs frozen
regardless of what the caller's own `requires_grad` settings were, and
`test_dpo_training_increases_the_policys_preference_margin` is the
trainability proof — after real training, the policy must prefer the
chosen response over the rejected one by a *larger* margin than at
initialization, the actual thing DPO training exists to accomplish, not
just "loss went down."

`examples/11_dpo_preference_tuning.py` makes the effect visible on a
real (if toy-scale) run: SFT first on *both* candidate responses (so the
model can already produce either one, leaving preference roughly
neutral — the printed margin after SFT alone is close to zero), then a
single DPO preference pair shifts the margin dramatically toward the
chosen response — no reward model, no sampled rollouts, just the one
preference pair.

### A much later fresh-eyes sweep found `dpo_step` silently never applied the configured LR schedule

`Trainer.dpo_step` "shares the same optimizer, gradient clipping, and
step counting" as `train_step`, per this chapter's own words above --
but it never actually applied `self.config.schedule`, unlike both
`train_step` and `grpo_step`, which each set the optimizer's LR from
`schedule.lr_at(self.step)` right at the top of the method. `Trainer
Config.schedule` is framed as a `Trainer`-level opt-in with no
indication of being scoped to specific training modes, and this
module's own docstring explicitly lists what DPO shares with
pretraining/SFT ("same optimizer/grad-clip/step-counting machinery")
-- schedule was simply never wired up here. Confirmed live: with a
`WarmupCosineSchedule` configured, the optimizer's LR stayed frozen at
the initial `TrainerConfig.lr` across every `dpo_step` call regardless
of the advancing step counter, while `train_step` on an identical
config tracked `schedule.lr_at(step)` exactly -- a silent,
undetectable-in-a-loss-curve LR-schedule bypass for every DPO run that
opts into a schedule, exactly the documented "SFT → DPO/RLHF" pipeline
this chapter itself describes. Fixed by adding the identical schedule
block `train_step`/`grpo_step` already have. Verified by reverting and
watching the new test fail with the literal old bug's own shape: the
optimizer's LR identical at warmup step 0 and step 3, instead of
ramping up between them. 1 new test.

## Agentic RL's environment harness: sandboxed coding tasks with real, verifiable rewards

§3.6e's post-training line ends with agentic RL — "RL on long-horizon
tool-use tasks... this, not pretraining, is what turns a base model
into a Fable/K3-class agent. Includes the RL environment harness
(sandboxed coding tasks with automatic verification)." The full RL
training loop (a real policy-gradient algorithm — PPO, GRPO, or similar
— plus a model-in-the-loop training run) is real, deferred work this
project doesn't have the compute for yet. The harness that loop would
consume is genuinely buildable and testable today, and that's what
`sarva_foundry.rl` is.

A `CodingTask` pairs a prompt with `test_code` that exercises a
submission and exits non-zero on any failed assertion — the automatic
verification the reward comes from, not a human or model judgment call.
`evaluate_submission(task, submitted_code)` runs the submission plus
the test code in a **genuinely separate subprocess** (not `exec()`
inside the caller's own process — the same isolation `RunShellTool`
already uses in `core/sarva/agent/tools.py`, for the same reason) under
a hard wall-clock timeout, and returns a real binary reward: `1.0` if
every assertion held, `0.0` otherwise — including a timeout, which
counts as a genuine failure rather than a special case the caller has
to handle. `test_submission_runs_in_a_genuinely_separate_process`
proves the isolation directly: a submission that mutates its own
process's environment variables can't leak that mutation back into the
caller.

**"Sandboxed" named honestly, not overclaimed:** subprocess isolation
plus a timeout is real isolation — it's not a full security sandbox.
Submitted code still runs with the same filesystem/network permissions
the parent process has. A production RL-from-code-execution system
needs a real container/VM boundary (gVisor, Firecracker, ...); that's
real, deferred, infrastructure-heavy work, named directly in
`environment.py`'s own module docstring rather than implied to already
be covered.

**A real reward-hacking bug found by actually submitting `sys.exit(0)`
as a "solution":** the combined subprocess script is `submitted_code`
followed by `task.test_code`, so code that exits (or `os._exit`s)
before that point got a clean process exit with zero of `test_code`'s
assertions ever having run — `reward=1.0` on a policy that learned
nothing about the task, the same reward-hacking exploit shape a prior
sweep already found and fixed once in the eval harness's own scoring,
just never checked against this newer file. Fixed with a per-call
random sentinel (`secrets.token_hex`, generated fresh each call so a
submission can't guess and print it) written only *after* `test_code`
finishes; a submission is now only rewarded if its process both exits
zero *and* the sentinel actually reached stdout. Confirmed live before
fixing: `evaluate_submission` scored a bare `sys.exit(0)` submission
`reward=1.0` against every bundled task.

**A second, independent reward-hacking bug in this same function, found
the same way the first one was — by actually submitting the exploit and
watching `reward=1.0` come back:** the per-call sentinel closed
`sys.exit(0)`, but it was embedded in *plaintext* in the very script
file the submission executed from (`submitted_code` + `task.test_code`
+ `print(sentinel)`, written to disk before the subprocess even
started). Any submission could `open(__file__).read()`, regex out the
sentinel, print it, and `sys.exit(0)` — full reward for zero real work,
worse than the first exploit (that one still required getting lucky on
an early-enough exit; this one worked every time, deterministically).
Confirmed live: this exact submission scored `reward=1.0` against a
task whose test was deliberately impossible to pass, before this fix.
No embedded state (file, env var, `argv`) can close this class of bug —
`submitted_code` always runs before `task.test_code` in the same
process (it has to, so `test_code` can call what it defines), so
anything present in that process from the start is readable by
`submitted_code` before `test_code`'s real check ever runs. The actual
fix is genuine temporal separation: the submission and test code are
now streamed to the subprocess over its own stdin in two stages — the
driver (passed via `python -c`, so `-c` mode's `__main__` has no
`__file__` at all) execs `submitted_code` first and prints its own
acknowledgment marker, and only *after* the caller sees that marker
does it write `task.test_code` and the sentinel to the same pipe.
Verified against both the file-read exploit and the same idea via
`os.environ` (confirming the fix is "nothing is reachable early," not
"just don't use a file").

**A third, independent bypass proved that fix's own "reading ahead on
stdin just blocks, revealing nothing" claim wrong for one specific
channel.** A background thread reading the raw file descriptor
directly (`os.read(0, ...)`, bypassing Python's buffered `sys.stdin`
entirely) can win a genuine OS-level race for the phase-2 bytes the
instant they arrive, before the driver's own single-threaded read gets
to them — confirmed live, 5/5 runs against a task whose test
unconditionally raises returned `passed=True, reward=1.0`, with
`task.test_code` never actually executing at all (its `AssertionError`
never fired — the driver's own read got EOF instead, exec'ing an empty
string). **Any two threads of the same OS process have equal access to
that process's entire file descriptor table** — there's no way for one
thread to reserve a descriptor against another, malicious thread the
way a lock or a permission bit guards against a genuinely *different*
process. Mitigated, not provably closed, and named that way on
purpose: fd 0 is `dup()`'d away and replaced with `os.devnull` before
`submitted_code` is ever exec'd, so this exact attack now gets EOF
immediately (verified live, 0/5 across five runs, `test_code`'s real
`AssertionError` now genuinely fires every time) — but a sufficiently
determined submission could still enumerate every open file descriptor
and race whichever one carries phase-2 content instead. Closing this
class provably needs genuine process/container isolation between the
code being rewarded and the code determining the reward — the same
real, deferred, infrastructure-heavy "container/VM boundary" work this
module's own docstring already names as needed for production use,
now spelled out explicitly rather than left implicit in that general
disclaimer.

**A fourth, related bug in the same file: the "hard wall-clock timeout"
didn't actually kill everything a submission spawned.** A submission
that forked its own subprocess (`subprocess.Popen(["sleep", "20"])`)
kept that grandchild running — confirmed directly against the real
process table, still alive via `pgrep` seconds after
`evaluate_submission` had already returned `timed_out=True`. `subprocess.run`'s
own `timeout=` (and a plain `proc.kill()`) only ever signals the
immediate child. Fixed by spawning the submission in its own process
group (`start_new_session=True` on POSIX, `CREATE_NEW_PROCESS_GROUP` on
Windows) and killing the whole group on timeout (`os.killpg(...,
SIGKILL)` on POSIX, `taskkill /F /T` on Windows) instead of just the one
PID.

**A fifth bug, this time a false-*negative* reward-corruption gap rather
than a hacking exploit: a correct submission could deadlock and score
`reward=0.0` for the wrong reason entirely.** `stdout` is drained
concurrently by its own reader thread specifically so it can never back
up, but `stderr` was only ever read via `proc.stderr.read()` *after*
`proc.wait()` returned. OS pipes have a small, fixed kernel buffer
(64KB on Linux/macOS) — a submission writing more than that to `stderr`
with nothing draining it blocks on its own `write()` syscall once the
buffer fills, while `evaluate_submission` is simultaneously blocked
inside `proc.wait(timeout=...)` waiting for a child that can now never
exit. A genuine deadlock, broken only by the task's own wall-clock
timeout forcibly killing the process. Confirmed live: a completely
correct, trivial submission that merely printed ~200KB to `stderr`
(ordinary verbose debug logging, `DeprecationWarning`s, a logged
traceback — not adversarial) reliably scored `timed_out=True,
reward=0.0` regardless of the timeout value, silently corrupting the
training signal by penalizing correct work — the opposite direction
from every prior bug in this function, which all inflated reward for
code that didn't deserve it. Fixed the same way `stdout` already is:
`stderr` is now drained by its own concurrent reader thread instead of
being left to fill the pipe while this function waits on the process.
Verified live the same submission now completes in milliseconds with
the full stderr captured; verified by reverting and watching the new
test fail with the literal old bug's own shape (`timed_out=True` after
the full 5-second timeout elapsed).

`CODING_TASKS` bundles three small, real, hand-verified tasks — same
honesty discipline as `sarva.eval.benchmarks.ARITHMETIC`: real problems
with real, hand-checked reference solutions, not a claim to
HumanEval-scale coverage. Each task's own tests are proven
*discriminating*, not just satisfiable: `test_bundled_coding_tasks_reject_a_deliberately_wrong_solution`
confirms a plausible-but-wrong solution actually fails, not just that
the correct one passes.

`examples/13_rl_coding_environment.py` runs three fixed "policies"
(stand-ins for what a real agentic-RL rollout would sample from a
model) against the bundled tasks and prints the genuinely-earned reward
for each: a correct solution scores 1.0, a plausible-but-wrong one
scores 0.0 with the real captured `AssertionError`, and an infinite
loop is caught by the timeout and scored 0.0 rather than hanging.

## GRPO: the training loop the harness was missing

The harness above computes rewards; it doesn't do anything with them.
`sarva_foundry.train.rl` closes that gap with **Group Relative Policy
Optimization** (Shao et al. 2024, DeepSeekMath) — the last named piece
of §3.6e's agentic RL line. For each prompt, sample a *group* of K
completions from the current policy, score each with a real reward
function, and use each completion's reward *relative to its own
group's mean* — `(reward - group_mean) / (group_std + eps)` — as the
policy-gradient weight. No separate value network/critic needed, unlike
full PPO, which is exactly why GRPO is the lighter-weight,
teaching-scale-appropriate choice here.

`sample_completion(model, prompt_ids, max_new_tokens, temperature)`
does the rollout under `torch.no_grad()` — sampling itself isn't
differentiable, and doesn't need to be. The gradient comes entirely
from re-evaluating each sampled completion's log-probability under the
*current* model parameters afterward, reusing DPO's own
`sequence_logprobs` directly: REINFORCE's gradient estimator is
`E[R · grad_theta log P(action)]`, and that log-probability term is
exactly what `sequence_logprobs` already computes. `build_grpo_batch`
pads and masks a group of completions the same way `build_sft_batch`
does (mask covers only each completion's own tokens, never the shared
prompt); `Trainer.grpo_step(x, y, mask, rewards)` computes the
advantages and does the update, mirroring the `build_*_batch` →
`Trainer.*_step` shape both SFT and DPO already established.

**A real finding from actually running this, not assumed from the
math:** this project's tiny, weight-tied, freshly-initialized
transformers turn out to have extremely peaked initial sampling — one
dominant token at >99% probability regardless of prompt, measured
across ten different random seeds before any test was written, not a
one-off fluke. At temperature=1.0 this leaves no exploration for GRPO
to learn from at all (a zero-variance group every single step, correctly
scored as a no-op — see below). A higher rollout temperature (8.0)
restores real exploration; standard practice in real-world RL
fine-tuning too, not a workaround invented just to make a demo work.

**Two properties tested directly, not just the trainability outcome:**
a genuinely zero-variance group (every completion scoring identically)
is a deliberate no-op — zero loss, unchanged weights, but the step
counter still advances — rather than dividing by a near-zero standard
deviation and producing garbage.
`test_grpo_training_increases_the_rewarded_behaviors_probability` is
the real end-to-end proof, mirroring DPO's preference-margin test
exactly: measure a target token's sampling rate before training,
train for real, measure again — 12.5% → 69% in the actual test run
recorded in BUILD-JOURNAL.md, not an assumed number.

**A real bug found in that "deliberate no-op" guard itself, not just
verified to exist:** a group of exactly *one* completion — a realistic
input, since neither `build_grpo_batch` nor `grpo_step` enforces a
minimum group size, and a small/resource-constrained rollout or a group
filtered down to one after removing timed-out/errored completions are
both real cases — makes `rewards.std()` divide by `n-1=0` under
PyTorch's default unbiased estimator, producing `nan`, not a small
number. `nan < 1e-6` evaluates to `False` (NaN comparisons are always
false), so the zero-variance guard silently let a NaN advantage,
NaN loss, and NaN gradient through — poisoning every model parameter
with NaN in one step, confirmed directly on a real tiny transformer
before this fix. Worse than a crash: it's silent, undetected model
corruption during a real training loop, not an error anyone would
notice until much later. Fixed by treating `torch.isnan(std)` the same
as the already-handled near-zero case — one extra condition, no change
to the documented no-op contract itself, since NaN-variance genuinely
*is* the same "no relative signal to learn from" situation the
existing guard already names.

**The equally real symmetric case, one step earlier in the same
pipeline, found by a much later fresh-eyes sweep: a group filtered down
to *zero* completions.** The group-of-one fix above already documents
"a group filtered down to one after removing timed-out/errored
completions" as a real scenario against the sandboxed coding-task
harness — the identical filtering reducing a group to *zero* is
equally real, and hits `build_grpo_batch` itself (called before
`grpo_step` is ever reached) rather than the NaN-variance guard.
`max(len(ids) for ids, _ in sequences)` over the resulting empty list
raised a bare, uninformative `ValueError: max() iterable argument is
empty` — the exact same bug class already found and fixed in the
sibling `build_sft_batch` (see the SFT section above), just never
propagated to this later-written sibling function. Confirmed directly:
`build_grpo_batch([1, 2, 3], [])` raised that bare message before this
fix. Fixed identically to `build_sft_batch`'s own guard: an explicit
`if not completions:` check raising a clear, actionable
`ValueError("build_grpo_batch requires at least one completion, got an
empty list")` before the `max()` call is ever reached. Verified the new
test is real: reverted the fix and watched it fail with the literal old
bug's own message (`"max() iterable argument is empty"`) instead of the
new one, before re-applying.

`examples/14_grpo_rl_training.py` runs that exact scenario and prints
the real before/after rates, then prints — labeled explicitly as
illustrative, not executed — exactly how `CODING_TASKS`/
`evaluate_submission` from example 13 would plug in as the reward
function for real coding-task RL: the GRPO loop itself doesn't change
at all, only the reward function does. It's not run for real here
because a 2-layer, 16-dim toy transformer genuinely cannot learn to
write working Python from sparse code-execution rewards in a short
demo, and this project doesn't fabricate results to make a chapter look
more finished than it is.

## Reasoning-token training: SFT cold start, then GRPO on a verifiable reward

§3.6a names "reasoning/thinking-token support... o1/R1-class" directly,
citing DeepSeek-R1-class open recipes as the reference — and until now
it was the one item on that list with zero code anywhere in `foundry/`,
confirmed by grep before starting. `sarva_foundry.train.reasoning`
closes it, deliberately reusing SFT and GRPO completely unchanged
rather than inventing new training machinery: the only new code is a
reward function.

**The two-stage shape isn't arbitrary — it's DeepSeek-R1's own published
finding, taken directly.** The R1 paper's own ablation ("R1-Zero") found
that pure RL from a base model learned to reason but produced real
format/readability problems, which is exactly why the paper's final
recipe adds a small "cold start" SFT stage — teaching the
`<think>...</think>` format via imitation — before RL takes over. This
project's own GRPO work found something structurally similar earlier
(tiny, freshly-initialized transformers have extremely peaked initial
sampling, leaving RL nothing to explore from) — the cold-start stage
here plays the same real role: giving GRPO something better than noise
to refine.

`reasoning_reward` sums two signals, matching R1's own reward design: a
`format_reward` (well-formed `<think>...</think>` wrapping non-empty
reasoning, followed by a non-empty answer) and an `answer_reward` (does
the real expected answer appear in whatever follows `</think>`),
weighted 0.3/0.7 toward correctness.

**A real reward-hacking exploit, found empirically and fixed, not
hidden:** the first version of both reward functions matched only the
*first* `</think>` in a completion. GRPO training discovered it could
pad completions with many extra `</think>` copies — abandoning the real
format (worth less reward) to inflate the higher-weighted answer
reward's loose "contains" check with repeated copies of the target
digit, without genuinely answering correctly. Caught by actually
inspecting the trained model's greedy output, not by assuming the
climbing group-mean-reward curve meant real learning was happening. Both
reward functions now require **exactly one** `<think>`/`</think>` pair;
the exact degenerate completion that broke them is preserved as a
permanent regression test.

**A second real bug this surfaced, in the tokenizer itself:** decoding a
genuinely undertrained model's output crashed with `UnicodeDecodeError`
— `ByteLevelBPETokenizer.decode()` had only ever been exercised on ids
that came from `encode()` on real text (always valid UTF-8 by
construction), never on arbitrary model-generated ids, which carry no
such guarantee. Fixed with `errors="replace"` in the final UTF-8 decode
— standard practice for any real tokenizer used in inference/RL rollout,
not a workaround specific to this example.

**A third real reward-hacking exploit, found later while checking
`sarva.eval.harness.contains_match` for the identical bug:**
`answer_reward`'s own docstring said it followed "the same
`contains_match` philosophy" — which turned out to include
`contains_match`'s own bug: a raw substring check
(`expected_answer in answer_segment`) rewards a genuinely WRONG answer
whenever the right digit happens to appear inside a longer wrong
number. For single-digit addition specifically, roughly half of all
real sums are two-digit (10-18), so a model answering `"17"` when the
expected answer is `"7"` was being scored fully correct. Confirmed
directly, not hypothetical: `answer_reward("<think>...</think>The
answer is 17", "7")` returned `1.0` before the fix. Fixed the same way
`contains_match` was: matched on a real word boundary
(`\bexpected\b`), not a raw substring.

**A fourth real reward-hacking exploit, an independent instance of the
identical sign-blindness bug already found and fixed in
`contains_match`:** the word-boundary fix above (`\bexpected\b`) was
copied from `contains_match` before that function's own *later*
sign-blindness fix existed, so it never inherited the correction. `\b`
treats `-` as a non-word character, so a word boundary already exists
between a minus sign and the digits that follow it — the pattern
matched a wrong `"-45"` just as readily as a correct `"45"`. Confirmed
directly, not hypothetical: `answer_reward("<think>...</think>The
answer is -45", "45")` returned `1.0` before this fix — a model that
reverses subtraction operand order got full training reward for a
numerically wrong answer, corrupting the actual RL training signal
itself, not just a benchmark report the way the eval-harness version of
this bug did. Fixed the identical way `contains_match` was:
`(?<![\w-])`/`(?!\w)` lookaround instead of `\b`, treating `-` as
significant on purpose rather than relying on word/non-word transitions
to get it right by accident. Verified the new test is real: reverted
the fix and watched it fail with the exact reward-hacking result
(`1.0` for a wrong answer) before re-applying.

**A fifth real reward-hacking exploit, an independent instance of the
identical decimal-point-adjacency bug already found and fixed in
`contains_match`, found by a much later fresh-eyes sweep:** the same
"copied before the later fix existed" gap the sign-blindness paragraph
above already names, just for a different character. `.` is not `\w`,
so it never blocked a match on either side of the pattern. Confirmed
directly, not hypothetical: `answer_reward("<think>...</think>The
answer is 9.5", "9")` returned `1.0` before this fix — a model whose
real sampled output happened to include a trailing decimal (a
plausible failure mode for an undertrained model, not contrived) got
full training reward for a numerically wrong answer via this module's
own real call path. Fixed the identical way `contains_match` was: `.`
added to the lookbehind's excluded-character set, and a second
lookahead, `(?!\.\d)`, that only rejects a match followed by a decimal
point *and then another digit* — not a bare trailing period, so an
ordinary sentence-ending one stays matchable. Verified the new test is
real: reverted the fix and watched it fail with the exact
reward-hacking result (`1.0` for a wrong decimal answer) before
re-applying.

**An eighth real reward-hacking exploit, found by a much later
fresh-eyes sweep: `answer_reward` never checked for an opening
`<think>` at all, only the closing `</think>` count.** `format_reward`
requires exactly one of *each* tag, but `answer_reward` only ever
required exactly one `</think>` before scanning whatever followed it —
a completion with a bare `</think>` and zero real `<think>` blocks (no
reasoning attempted whatsoever) still passed that single check.
Confirmed directly, not hypothetical: `answer_reward("</think>The
answer is 45", "45")` returned `1.0` — full 0.7-weighted answer credit
for a completion `format_reward` correctly scores `0.0`, so
`reasoning_reward` came out to `0.7`, dramatically higher than an
honest, entirely unformatted attempt (`0.0`) and trivially discoverable
by GRPO's own sampling/optimization loop, the same exploitable shortcut
every prior bug in this function closed. Fixed by additionally
requiring exactly one `<think>` and that it appear *before* the
`</think>` — closing a related variant too, where the single `<think>`
tag exists but appears after the `</think>` it's supposed to open,
which passes a bare `count == 1` check on both tags but is equally
degenerate. Verified the new test is real: reverted the fix and
watched it fail with the exact reward-hacking result (`1.0` for a
completion with no reasoning) before re-applying. 1 new test, 866 →
867 Python tests.

**The already-published 31% → 56% numbers below were re-checked
against both fixes above, not left standing on faith:** re-ran
`examples/17_reasoning_token_training.py` (same fixed seed,
fully deterministic) after each fix and got the identical 31% → 56%
result both times. Both exploits were real and worth closing
regardless — proven by the standalone reproductions above — but
neither happened to change this specific run's already-reported
numbers: this task's real answer space (single-digit sums, always a
positive integer) never organically produces a negative or decimal
completion, an honest outcome confirmed
by actually re-running the example, not assumed because the fix
"should" leave healthy runs unaffected.

`examples/17_reasoning_token_training.py` runs the real two-stage
recipe on single-digit addition (`"2 plus 3 = "` →
`"<think>2+3=5</think>5"`): cold-start SFT reliably nails the format
but leaves real headroom on the arithmetic (31% answer accuracy); 400
steps of GRPO refinement on top measurably improves it (56%),
every number from real generated text checked against the real digit
sum. Format compliance never regresses — the tightened reward function
makes a degenerate, format-abandoning strategy score strictly worse,
not just differently.

## What's next

Web/code/books/math-scale corpus sourcing and mixing recipes (local
files, exact + near-duplicate dedup, length filtering, and provenance/license
tracking including per-file manifests all exist now — larger-scale
sourcing doesn't yet; nor does an LSH banding index, which near-duplicate
dedup would need to scale past the current O(kept²) pairwise
comparison), and the distributed training slice of §3.6d (FSDP → 3D
parallelism, loss-spike handling, scaling-law tooling) once a model
worth training at that scale exists. §3.6e's post-training line — SFT,
DPO, agentic RL (both the environment harness and the GRPO training
loop), and now reasoning/thinking-token training — is fully built, at
the scale a laptop can actually run and verify. §3.6a's architecture
*and* training-recipe lists have no remaining named gap.
