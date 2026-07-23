# Distillation: frontier-as-teacher synthetic data

`sarva.distill` closes §3.6c's named gap: "synthetic-data generation
(frontier-as-teacher via the provider layer)."

## Why it's built on `Provider`, not a specific backend

`distill()` takes any `Provider` and a model id — the exact same
abstraction `sarva.eval.harness.run_benchmark` uses to *grade* every
registered model identically. Here it's used to *generate* data instead
of scoring it: whichever model is configured (Anthropic, OpenAI,
Google, a local Ollama model) can serve as the teacher, with zero
backend-specific code in `distill.py` itself.

```python
from sarva.distill import distill, save_jsonl

records = await distill(
    ["What is the capital of France?", "What is 2 + 2?"],
    provider,
    model="claude-haiku-4-5",
)
save_jsonl(records, Path("distilled.jsonl"))
```

Each `DistillationRecord` carries `prompt`, `completion`, and `model` —
plain data, serialized as line-delimited JSON.

## A deliberate non-dependency: core and foundry stay disjoint

`sarva.distill` produces plain records, not `sarva_foundry.train.sft.
SFTExample` objects directly. This is intentional: `core`'s and
`sarva_foundry`'s `pyproject.toml`s name completely disjoint dependency
sets (`anthropic`/`openai`/`google-genai`/`fastapi`/... vs. `torch`/
`numpy`), and neither package imports the other. A caller who wants to
turn distilled data into foundry SFT training data writes the one line
of glue themselves:

```python
from sarva_foundry.train import SFTExample

sft_examples = [SFTExample(prompt=r.prompt, response=r.completion) for r in records]
```

`examples/12_distillation_to_sft.py` is exactly that glue script, made
runnable end to end: distill from a real Claude model, then SFT-train a
toy foundry transformer on the results.

## Errors propagate, deliberately differently from the eval harness

`run_benchmark` scores a failing case as incorrect and keeps going —
one bad benchmark case shouldn't hide every other case's real result.
`distill()` does the opposite: a `ProviderError` on any prompt
propagates immediately rather than being caught. Distillation output
becomes training data; a silently-missing or garbage record is a worse
outcome than a loud failure a caller can retry or investigate.

## Try it

```bash
sarva distill prompts.txt --model claude-haiku-4-5 --out distilled.jsonl
```

`prompts.txt` is one prompt per line. Requires the model's provider to
actually be configured (an API key set, or a reachable local Ollama
instance) — `sarva distill` fails loudly with which provider is missing
if not, the same "loud, fixable, not silently wrong" principle every
other input-validation error in this project follows.
