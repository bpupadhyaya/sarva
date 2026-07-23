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

## What's next

Web/code/books/math-scale corpus sourcing and mixing recipes (local
files, exact + near-duplicate dedup, length filtering, and provenance/license
tracking including per-file manifests all exist now — larger-scale
sourcing doesn't yet; nor does an LSH banding index, which near-duplicate
dedup would need to scale past the current O(kept²) pairwise
comparison), and the distributed training slice of §3.6d (FSDP → 3D
parallelism, loss-spike handling, scaling-law tooling) once a model
worth training at that scale exists. §3.6e's post-training line — SFT,
DPO, and agentic RL (both the environment harness and now the GRPO
training loop) — is fully built, at the scale a laptop can actually run
and verify.
