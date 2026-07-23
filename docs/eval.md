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

The same reasoning extends forward: once §3.1's planned foundry adapter
exists (a foundry-trained checkpoint plugged into the registry as a real
`Provider`), it will be gradable by this exact same harness with no
changes here. That adapter doesn't exist yet — named as real, deferred
work, not implied to already be done.

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

## A `ProviderError` on one case doesn't sink the whole run

If a case's request fails (rate limit, auth, any `ProviderError`), that
case is scored incorrect with the error text recorded as its output —
`run_benchmark()` keeps going rather than aborting the entire benchmark.
One flaky case shouldn't hide every other case's real result.

## Try it

```bash
sarva eval                      # every available model
sarva eval --model claude-opus-4-8
```

With no API keys configured, this grades the offline Mock provider
(which just echoes the prompt back — expect a low, honest score, not an
inflated one) against the bundled arithmetic benchmark, printing each
model's accuracy and correct/total count side by side.
