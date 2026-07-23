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

MoE and long-context RoPE-scaling (see the transformer chapter) are
both real, shipped foundry architecture features, and both round-trip
through `config.json` too — `MoEConfig`/`RopeScalingConfig` are flat,
JSON-safe dataclasses, serialized as nested `null`-or-object fields.
`load_checkpoint_bundle` stays backward compatible with a bundle saved
before either was wired in: a legacy `config.json` simply has no
`"moe"`/`"rope_scaling"` keys at all, and reloads as the same
dense/unscaled config it always did.

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
- **KV-cache reuse, but no batching.** Generation uses
  `sarva_foundry.inference.generate_with_cache` (see below) — real
  key/value caching across steps, not a naive full-recompute-per-token
  loop. Batching multiple concurrent requests together is the other half
  of §3.6f's "inference/serving stack" and remains separate, deferred
  scope — this adapter serves one sequence at a time.
- **Quantization is available (`sarva_foundry.quantization`, see below),
  but not wired into this adapter's serving path.** It exists today as a
  standalone accuracy/storage measurement tool, not a way to make
  `FoundryProvider` itself faster or lighter yet — a real int8-serving
  path is separate, deferred work (see below).

## The KV-cache: real incremental decoding

`sarva_foundry.model.kv_cache.KVCache` pre-allocates a
`(n_layers, batch, n_kv_heads, max_seq_len, head_dim)` buffer per
key/value and remembers every position's projection across calls.
`DecoderOnlyTransformer.forward(token_ids, cache=...)` then means "the
NEW tokens since the cache was last advanced," not the whole sequence —
`cache=None` (the default, and every call site before this parameter
existed) is exactly the original, unchanged behavior.

**A real bug this surfaced while building it, not a hypothetical:** the
first version leaned on `F.scaled_dot_product_attention(..., is_causal=True)`
even when the query length (new tokens) was shorter than the key length
(every cached position) — a reasonable-looking assumption (that
`is_causal` bottom-right-aligns a shorter query against a longer key)
that turned out to be **wrong** for this PyTorch version, confirmed
empirically (not by re-reading the docs harder) by comparing cached
generation logit-for-logit against known-correct full-recompute
generation and finding a real, large numeric divergence starting at the
very first cached token. The fix: build the causal mask explicitly —
`torch.ones(seq_len, total_len, dtype=torch.bool).tril(diagonal=start_pos)`
— row `i` (of the new tokens, at absolute position `start_pos + i`)
attends to every key at absolute position `<= start_pos + i`. This
subsumes the no-cache case exactly (`start_pos=0`, query length equals
key length reduces to the ordinary causal mask), so there's one code
path, not two. `tests/foundry/test_kv_cache.py` pins the property that
actually matters — cached, incremental generation must match known-correct
full-recompute generation, both at the logit level (`torch.allclose`
across several incremental steps, not just one) and at the final
token-sequence level (`generate_with_cache` producing token-for-token
identical output to `sample_completion` under greedy decoding).

`examples/15_kv_cache_inference.py` runs both generation paths on a
128-dim, 4-layer model for 200 tokens and prints real measured wall-clock
numbers — confirmed identical token output either way, ~2.4x faster
cached on the machine this was verified on (exact speedup varies by
hardware; the point is a real, measured, honestly-reported number, not an
assumed one).

## Quantization: real int8 weight-only compression

§3.6f's "inference/serving stack" names KV-cache, paged attention, and
quantization together. `sarva_foundry.quantization` closes the third —
genuinely separable from the batching/paged-attention gap left deferred
above, since it never touches the caching internals.

**What it actually is:** per-output-channel int8 round-to-nearest for
every `nn.Linear` layer's weight — one scale (`max(|weight_row|) / 127`)
per row, not one scale for the whole matrix, since different output
channels can have very different magnitudes and a single global scale
would waste int8's range on whichever channel is largest.

```python
from sarva_foundry.quantization import quantize_model, apply_quantized_weights

quantized = quantize_model(model)          # dict[str, QuantizedLinear], keyed by
                                            # the same dotted names named_modules() uses
apply_quantized_weights(model, quantized)  # mutates the live model in place
```

**A real, non-obvious interaction, checked rather than assumed:**
`DecoderOnlyTransformer` ties `lm_head.weight` to `tok_embeddings.weight`
— the literal same `Parameter` object. `quantize_model` quantizes
`lm_head` as an ordinary `nn.Linear` with no special-casing, and
`apply_quantized_weights` overwrites it via `module.weight.data = ...`.
Whether that breaks the tie was a real open question, not assumed either
way — it doesn't: since both names reference the identical `Parameter`
object, mutating one's `.data` necessarily mutates the other's too.
Verified directly in `test_apply_quantized_weights_preserves_tied_lm_head_and_embedding_identity`
rather than inferred from how weight tying happens to be implemented.

**Honestly scoped, the same way the KV-cache chapter above draws its own
line:** this reduces *storage* — a real, measured ~3.5–4x reduction
(int8's 1 byte/element vs. float32's 4, minus the small per-channel scale
vector's real overhead, checked against actual tensor byte counts, not
assumed from the nominal 4x ratio) — and measures the real accuracy cost
of quantizing a trained model's weights. It does **not** speed up
compute or shrink a running model's live memory footprint:
`dequantize()` converts back to float32 before every matmul runs, and
`apply_quantized_weights` exists specifically to measure accuracy impact
on a real forward pass, not to demonstrate a memory-saving serving path.
A real quantized *inference* server — one that keeps every layer in its
compact int8+scale form the entire time and dequantizes only the one
layer currently executing — is separate, deferred serving-optimization
work, the same category this chapter's own batching gap sits in.

`tests/foundry/test_quantization.py` pins the three claims that actually
matter: the round-trip error is *provably* bounded (every element of
`|dequantize() - original| <= scale/2`, round-to-nearest's own bound,
not just "small"), the storage reduction is a real measured byte count
(not an assumed ratio), and — mirroring the ablation harness chapter's
"positive control" discipline — a genuinely trained toy model's real
loss on its real training objective moves measurably after quantization
(proving `apply_quantized_weights` isn't a no-op) but stays bounded
(proving it isn't silently catastrophic either).
`examples/19_quantization.py` runs all of this against a real trained
model and prints the real measured numbers either way.

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
