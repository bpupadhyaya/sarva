# Serving a trained checkpoint: the foundry `Provider` adapter

Every other chapter in this book treats `sarva_foundry` as a training
library: tokenizer, transformer, pretraining, SFT, DPO, GRPO. None of
those checkpoints could come back into `sarva` (core) as an actual,
routable model — until now. `sarva.providers.foundry_provider.FoundryProvider`
plugs a checkpoint trained by any of the previous chapters into the exact
same `Provider` registry Anthropic, OpenAI, Google, and Ollama already
share, so the eval harness, the CLI, the agent loop, and `models.yaml`'s
router all treat a from-scratch checkpoint identically to a frontier one.

## Why this stayed a hard boundary until now

`core`/`sarva_foundry` have been kept **dependency-disjoint** since the
distillation glue script: `core`'s dependencies are lightweight API-client
SDKs, `sarva_foundry`'s are `torch`/`numpy`. Most Sarva installs never
train or run a local model and shouldn't be forced to pull in torch just
to `pip install sarva`. So `sarva_foundry` is an **optional extra**:

```bash
pip install sarva[foundry]
# or, inside this repo's own uv workspace:
uv sync --all-packages
```

`core/sarva/providers/foundry_provider.py` imports `torch`/`sarva_foundry`
lazily, function-by-function — importing the module itself always
succeeds, even on a plain-core install. Only actually loading or running a
checkpoint requires the extra, and does so with a clear, actionable
`ImportError` if it's missing rather than a confusing crash somewhere
deep in torch's own import machinery.

## Checkpoint bundles

A checkpoint "bundle" is a directory with three files:

| File | What it is |
|---|---|
| `model.pt` | A `Trainer.save_checkpoint()` output — real trained weights |
| `tokenizer.json` | A `ByteLevelBPETokenizer.save()` output |
| `config.json` | The flat `TransformerConfig` fields needed to reconstruct the model's shape before loading weights into it |

```python
from sarva.providers.foundry_provider import save_checkpoint_bundle

save_checkpoint_bundle(Path("checkpoints/my-model"), trainer, tokenizer, config)
```

**Honestly scoped:** MoE and long-context RoPE-scaling are real, shipped
foundry architecture features (see the transformer chapter), but their
configs aren't serialized into `config.json` yet — `save_checkpoint_bundle`
raises `NotImplementedError` rather than silently writing a bundle that
would reload as a plain dense, unscaled model that doesn't match what was
actually trained. Save a dense, unscaled checkpoint to use this adapter
today; wiring the two nested configs through is real, deferred follow-up.

## Wiring a bundle into the CLI

Point `SARVA_FOUNDRY_CHECKPOINTS` at a directory of bundles (one
subdirectory per checkpoint, named after the model id it should get):

```bash
export SARVA_FOUNDRY_CHECKPOINTS=~/checkpoints
sarva models                        # foundry/my-model now listed, [x] available
sarva eval --model foundry/my-model # graded by the exact same harness as every other model
```

`sarva.runtime.build_router()`/`build_providers()` gate this the same way
they already gate Ollama — a cheap probe (`_foundry_extra_installed()`,
mirroring `ollama_reachable()`) decides both whether a discovered
checkpoint is marked available in the registry and whether
`FoundryProvider` actually gets constructed, from one source of truth, so
a model is never marked available with no provider able to serve it.

No entry is added to `models.yaml`/`routing.yaml` — unlike the frontier
models, the set of foundry checkpoints is entirely per-install, so they're
discovered and registered into the registry dynamically
(`Registry.register()`, new this chapter) rather than declared statically.
They're never a default routing candidate for real tasks; use them via an
explicit `--model foundry/<name>` override.

## What the adapter honestly does and doesn't do

- **No chat template.** The prompt sent to the model is just the
  concatenated text of the system prompt (if any) and every message's
  text, in order — no `"User: "`/`"Assistant: "` role tags. This matches
  exactly how the SFT chapter's own toy examples train (raw prompt text,
  no role tags); a checkpoint trained with some other convention would
  need this adapter to match it, a real, named limitation rather than an
  assumed-universal one.
- **Coarse streaming, not incremental.** There's no wire protocol to
  translate the way there is for a real network API — generation runs
  synchronously (`asyncio.to_thread`, so the event loop still yields) and
  the full completion is streamed as one `TextDeltaEvent`, not true
  per-token streaming.
- **No batching, no KV-cache reuse.** One naive forward pass per generated
  token. This is exactly the gap a real foundry inference server (batched
  inference + KV-cache reuse around this same `DecoderOnlyTransformer`,
  named in §3.6f) would close — separate, deferred scope, not silently
  assumed solved by this adapter.

## Verified, not just unit-tested

Beyond the conformance suite (`tests/conformance/test_foundry_provider.py`
— real save/load round trips, a real generation producing a real
`DoneEvent`, and the `sarva.runtime` wiring), this was run through the
actual CLI end to end against a real toy bundle: `sarva models` correctly
lists `foundry/toy` as `[x]` available, and `sarva eval --model
foundry/toy` runs the real arithmetic benchmark against it — scoring 0%,
the honest result for an untrained toy checkpoint, the same
no-fabrication discipline the eval harness chapter established for the
zero-config Mock provider.
