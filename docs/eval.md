# Eval: grading every model with the same yardstick

`sarva.eval` closes §3.6g's named gap: "benchmark harness shared with
the registry (grades our models and third-party models with the same
yardstick)."

## Why it's built against `Provider`, not any specific backend

The harness's one real function, `run_benchmark()`, takes a
`Provider` and a model id — nothing else. That's deliberate: `Provider`
is already the abstraction that makes Anthropic, OpenAI, Google, Ollama,
and the offline Mock provider interchangeable everywhere else in this
codebase (the agent loop, the router, the CLI). Reusing it here means
`sarva eval` grades every registered model identically, with zero
special-casing per backend — literally "the same yardstick."

The same reasoning already extends all the way to the foundry track:
`sarva.providers.foundry_provider.FoundryProvider` plugs a
foundry-trained checkpoint into the registry as a real `Provider`, and
it's gradable by this exact same harness with zero changes here —
verified directly, not just claimed, by
`tests/conformance/test_foundry_provider.py`'s own `run_benchmark()`
test against a real (if untrained) toy checkpoint.

## What's in the box

```python
from sarva.eval import ARITHMETIC, run_benchmark

report = await run_benchmark(ARITHMETIC, provider, model="claude-opus-4-8")
print(report.accuracy)  # 0.0-1.0
```

- `BenchmarkCase` — a `(prompt, expected)` pair.
- `Benchmark` — a named list of cases.
- `run_benchmark()` — runs every case as an independent single-turn
  request (reusing `sarva.providers.base.complete()`, the existing
  "drain the stream, get the `DoneEvent`" helper — no new stream-handling
  code), grades each with a `Grader` function, and returns a
  `BenchmarkReport` with per-case results and `.accuracy`.
- `exact_match` / `contains_match` (the default) — `contains_match` is
  deliberately the default because real models rarely answer with
  *only* the expected string; grading on whether the expected answer
  appears anywhere in the output is a more honest measure of correctness
  than penalizing normal phrasing.
- `ARITHMETIC` — a bundled, ten-case benchmark. Genuinely small and
  simple on purpose: ten arithmetic problems, each answer computed and
  checked by hand, not generated and assumed correct — the same "real,
  not a fabricated placeholder" discipline the corpus pipeline's length
  filter and the multimodal degraders apply elsewhere in this project.
  Not a claim to GSM8K-scale coverage.

## A real grading bug found by re-checking a repeated claim

This project's own docs and journal have repeatedly said "Mock scores
0%, the honest result" — restated without ever re-running the actual
number. Doing exactly that (running `sarva eval --model mock` for real
and looking at the printed accuracy) found it was measurably **30%**,
not 0%: `contains_match`'s naive substring check graded a genuinely
wrong numeric answer as correct whenever the right digits happened to
appear inside a longer wrong number (expected `"9"` matched inside a
wrong `"89"`), and two of `ARITHMETIC`'s own cases (`div-1`, `div-2`)
had used a perfect square as the dividend with its own square root as
the divisor (`144 / 12`, `81 / 9`) — so the correct answer was already
sitting in the prompt text verbatim, and Mock's own prompt echo passed
grading without computing anything.

Fixed both halves: `contains_match` now matches on a word boundary
(`\bexpected\b`, not a raw substring), and `div-1`/`div-2` were
replaced with `84 / 7` and `45 / 5`, where the quotient never appears
in the prompt. `sarva eval --model mock` now genuinely reports `0%
(0/10)`. A related test bug this surfaced too: the CLI conformance
test meant to catch this asserted `"0%" in result.stdout` — which
"30%" also contains as a trailing substring, so that test had been
silently passing throughout, whatever the real number was. Fixed to
check the precise `"0/10"` marker instead.

## The word-boundary fix above was itself sign-blind, and its own inverse never matched at all

A later sweep, checking `contains_match` again rather than assuming
the word-boundary fix above closed every gap in the same shape:
`\b` treats `-` as a non-word character, so a word boundary already
exists between a minus sign and the digits that follow it. Confirmed
live and concretely reachable via `ARITHMETIC`'s own bundled cases:
`sub-1` expects `"45"` for `"92 - 47"`, and a model that reverses
operand order (a common weak-model mistake — computing `47 - 92 = -45`
instead) got full credit for a numerically wrong answer, since `"45"`
inside a wrong `"-45"` satisfied the exact same `\b45\b` pattern a
correct `"45"` did.

**A second, related defect surfaced while verifying that fix, not
reachable by any bundled case today but real:** the mirror-image bug.
`\b` doesn't fire between two non-word characters either — a space and
a leading `-` — so a genuinely negative `expected` value (e.g.
`"-5"`) never matched *at all* under the same `\b`-based pattern; a
correctly negative model answer would have been scored wrong. No
bundled `ARITHMETIC` case has a negative `expected` answer, so this
never fired in practice, but it's the same root defect (`\b` treating
`-` inconsistently) manifesting in the opposite direction, not a
separate, unrelated issue.

Fixed by replacing `\b` with explicit lookaround that treats `-` as
significant on purpose, rather than relying on word/non-word
transitions to get it right by accident: `(?<![\w-])` before the
pattern (rejects a match preceded by a digit, a letter, *or* a minus
sign) and `(?!\w)` after it (rejects a match followed by a digit or
letter, preserving the original digit-adjacency fix — `"9"` still
can't match inside `"89"`). Verified the new tests are real: reverted
the fix and watched both fail — the sign-blind case scoring `-45` as a
match, the negative-expected case never matching `-5` at all — before
re-applying. All 11 pre-existing eval-harness tests pass unchanged.

## A third bug in the same boundary logic, unaddressed by either fix above: decimal-point adjacency

A much later fresh-eyes sweep checked the same function a third time,
rather than assuming two rounds of fixes had closed every gap in the
boundary logic's shape. Neither the lookbehind nor the lookahead
excludes `.` — it isn't `\w`, so (exactly like `-` before the
sign-blindness fix) it never blocked a match on either side. Confirmed
live, two distinct failure directions: `expected="9"` matched inside
`"0.9"` (a decimal point immediately *before* the match), and
`expected="12"` matched inside `"12.5"` (a decimal point immediately
*after* it) — both numerically wrong answers scored `correct=True`.
Concretely reachable via `ARITHMETIC`'s own bundled division cases
(`div-1: 84/7=12`, `div-2: 45/5=9`) — the cases a real weaker model is
most likely to answer with a decimal instead of a clean integer.

The fix isn't symmetric with the `-` fix, and deliberately so. Adding
`.` to the lookbehind's excluded-character set (mirroring `-` exactly)
correctly blocks `"0.9"` from matching `"9"` — a decimal point
directly preceding a number is never *not* part of that number.  But
unconditionally excluding a trailing `.` the same way would break the
single most common real case in this benchmark's own output shape: an
ordinary sentence-ending period (`"The answer is 12."`). A second,
narrower lookahead, `(?!\.\d)`, threads this correctly — it only
rejects a match immediately followed by a decimal point *and then
another digit* (a genuine decimal continuation, `"12.5"`), leaving a
bare trailing period exactly as matchable as it always was. Verified
live across both directions plus the existing digit-adjacency,
sign-adjacency, and sentence-ending-period cases together, confirming
the fix doesn't regress anything the two earlier fixes established.
Verified by reverting and watching the new test fail with the literal
old bug reproducing itself: `contains_match("0.9", expected="9")`
returning `True`. 1 new test, 735 → 736 Python tests.

## A `ProviderError` on one case doesn't sink the whole run

If a case's request fails (rate limit, auth, any `ProviderError`), that
case is scored incorrect with the error text recorded as its output —
`run_benchmark()` keeps going rather than aborting the entire benchmark.
One flaky case shouldn't hide every other case's real result.

### A real gap found by a fresh-eyes sweep of `complete()`'s own callers: one failure shape bypassed that protection entirely

`sarva.providers.base.complete()` (the "drain the stream, return the
`DoneEvent`" helper both `run_benchmark` and `sarva.distill.distill`
build on) raised a bare `RuntimeError` — not a `ProviderError` — when a
provider's `generate()` async generator finished iterating without
ever yielding a `DoneEvent` or `StreamErrorEvent`. Both real callers
only ever catch `ProviderError` around each individual call, so this
one specific failure shape slipped straight through: confirmed live, a
structurally-valid provider (this chapter's own opening section names
grading third-party models a first-class use case) whose stream ended
early — exactly what an ordinary network drop mid-stream does to a
not-as-defensively-written adapter — crashed the *entire* benchmark
run, discarding every already-graded case's result, directly
contradicting the section above. The identical root cause hit
`distill()` even harder: it bypassed `DistillationError` entirely,
silently discarding an already-generated (real, potentially expensive)
completion instead of preserving it via `partial_records` the way
every other failure already does.

Fixed by raising `ProviderError` instead — caught by both callers
exactly like the `StreamErrorEvent` branch immediately above it already
is; this is the same "stream ended abnormally" family of failure, just
reached a different way. Verified live both callers now degrade
correctly: `run_benchmark` scores the affected case incorrect and
keeps going, `distill()` raises `DistillationError` carrying every
record already generated. Verified by reverting and watching both new
tests fail with the literal old bug's own `RuntimeError` propagating
uncaught. 2 new tests.

## Try it

```bash
sarva eval                      # every available model
sarva eval --model claude-opus-4-8
```

With no API keys configured, this grades the offline Mock provider
(which just echoes the prompt back — expect a low, honest score, not an
inflated one) against the bundled arithmetic benchmark, printing each
model's accuracy and correct/total count side by side.
