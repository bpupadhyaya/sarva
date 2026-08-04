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

### Every checkpoint used to reload from disk on every single request in a long-running server

A round-44 sweep, looking specifically for "expensive work redone on
every ordinary call over the life of a long-running process" (the same
lens that found `AgentLoop`'s leaking run directories the round
before), found `load_checkpoint_bundle` had no caching at all.
`sarva.server.app`'s `/chat` and `/ws/chat` handlers build a fresh
`AgentLoop` — and therefore a fresh `FoundryProvider` — on **every
request**, deliberately, so a saved `config.json` API-key change takes
effect without restarting the server. But `FoundryProvider.__init__`
eagerly calls `load_checkpoint_bundle` for every discovered bundle, and
that function did a real `torch.load()` from disk every single time it
ran. Confirmed live: 5 simulated requests against a real toy checkpoint
fired `torch.load()` 5/5 times, with zero reuse — and this happened for
**every** request through a `sarva serve` process with any foundry
checkpoint configured, including requests that ended up routed to a
completely different provider (Claude/GPT/Ollama), not just ones that
actually used the foundry model. With a realistic (hundreds of MB to
multi-GB) checkpoint this is real added latency and memory/GC churn on
every request, not milliseconds.

Fixed with a module-level cache in `load_checkpoint_bundle`, keyed on
the resolved bundle directory path **and** `model.pt`'s own mtime — not
path alone, the same "mtime distinguishes touched from untouched"
technique already used to fix the cross-process lock's Windows bug (see
the packaging chapter) — so a checkpoint retrained and re-saved in
place is picked up on its very next load instead of silently serving
stale weights forever. Confirmed live both ways: 5 requests against an
unchanged checkpoint now trigger exactly 1 real `torch.load()`, and
reloading after retraining-in-place (a new `model.pt` written to the
same path) produces genuinely different weights, not a stale cached
copy. Sharing the returned model instance across concurrent requests is
safe: `generate_with_cache` (`sarva_foundry.inference`) allocates a
fresh `KVCache` per call and never mutates persistent model state, and
this adapter never trains through a loaded model. Verified by reverting
and watching the new test fail with the exact wrong count (5 calls
instead of 1) before re-applying. 2 new tests, all 19 pre-existing
foundry-provider tests pass unchanged.

### The fix above then leaked every superseded checkpoint copy forever — found by sweeping the fix itself

A much later round applied this project's own "does yesterday's fix
have a bug" lens to `_bundle_cache` itself: keyed on `(directory,
mtime)` so a retrained checkpoint is picked up fresh — but nothing
anywhere ever evicted the OLD entry once a new one landed for the same
directory. Confirmed live: repeated retrain/re-save/reload cycles
against one checkpoint directory — the exact "no server restart
needed" workflow this cache exists to support — left every superseded
model copy resident in memory simultaneously (20 cycles → 20 full
model copies, 20× a single model's real footprint), even though only
the newest entry is ever reachable again through ordinary use. Over a
long-running `sarva serve` process's uptime, with realistic
(hundreds-of-MB-to-multi-GB) checkpoints, a handful of retrain
iterations is enough to exhaust memory.

Fixed by evicting every other cached entry for the same resolved
directory path right before adding the new one — a directory only
ever has one current mtime at a time, so any entry under a different
mtime for that same path is permanently unreachable the moment a new
one lands, and is now removed rather than left to accumulate. Verified
live both properties hold: the cache stays at exactly one entry after
20 retrain cycles on one directory, and loading two genuinely distinct
checkpoint directories still keeps both entries — eviction is scoped
to the same path, not global. Verified by reverting and watching the
new test fail with the exact wrong count (5 stale entries instead of
1). 1 new test, 714 → 715 Python tests.

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

**A real bug found by actually corrupting a real bundle and calling
`build_providers()`, not assumed impossible:** `discover_checkpoint_bundles`
only checks that a bundle's three files *exist* — never that they're
actually valid — and `FoundryProvider.__init__` used to load every
discovered bundle via a plain dict comprehension with no error
handling at all. A bundle with a truncated `model.pt` (an interrupted
save, disk corruption — a realistic failure mode, not a contrived one)
made `torch.load()` raise an uncaught `OSError`, which crashed
`FoundryProvider` construction entirely — and since `build_providers()`
calls it with no try/except either, this crashed *every* caller of
`build_providers()` (every CLI command, `/chat`, `/ws/chat`, server
startup), not just a request that tried to use that specific
checkpoint. The same "corrupted on-disk state" bug class already fixed
twice elsewhere in this project (`~/.sarva/config.json`, a saved
session file), just not yet closed here.

Fixed the same way `sarva.eval.harness.run_benchmark` already treats a
single bad case: `FoundryProvider.__init__` now loads each bundle in
its own `try`/`except`, records a broken bundle's name and error
message on `provider.broken_bundles` (a plain `dict[str, str]`, publicly
readable, not silently swallowed), and skips it — every *other* valid
bundle still loads and the provider still constructs successfully.
Construction only fails (the same `ValueError` shape as the existing
"no bundles found at all" case) when literally every discovered bundle
turned out to be broken, since a `FoundryProvider` serving zero models
isn't meaningfully different from one that failed to construct.

**The same class of bug in the sibling call site that fix didn't cover,
found by a later sweep:** `build_router()` has its own, separate loop
over `discover_checkpoint_bundles()` — it calls `model_info_for_bundle()`
directly to register a registry entry, unrelated to
`FoundryProvider.__init__`'s own bundle-loading loop above. That
function is documented "torch-free" (it only reads `config.json`), but
had the identical gap: no error handling at all. Confirmed live two
ways: a bundle with malformed `config.json` raised an uncaught
`json.JSONDecodeError`; a bundle with syntactically valid JSON missing
the required `"max_seq_len"` key raised an uncaught `KeyError`. Either
one crashed `build_router()` entirely — and since `build_router()` is
called by every CLI command and server startup, one corrupted
checkpoint's metadata (not even its weights) took down everything, the
same broad blast radius the `FoundryProvider` fix above closed for a
different call site. Fixed the identical way: catch `(OSError,
ValueError, KeyError)` around `model_info_for_bundle()` per bundle and
skip the broken one, rather than let it crash every other, perfectly
valid bundle's registration. Verified the new test is real: reverted
the fix and watched it fail with the raw, uncaught `JSONDecodeError`
before re-applying. All 4 pre-existing runtime tests pass unchanged.

## What the adapter honestly does and doesn't do

- **No chat template.** The prompt sent to the model is just the
  concatenated text of the system prompt (if any) and every message's
  text, in order — no `"User: "`/`"Assistant: "` role tags. This matches
  exactly how the SFT chapter's own toy examples train (raw prompt text,
  no role tags); a checkpoint trained with some other convention would
  need this adapter to match it, a real, named limitation rather than an
  assumed-universal one.
- **Text-only, and it says so loudly, not silently.** A foundry
  checkpoint's registry entry declares `modalities_in={TEXT}` and
  `tool_use=False`, and the adapter means it end to end: sending it an
  `ImageBlock`/`ToolCallBlock`/anything but plain text raises a clear
  `ValueError` rather than silently dropping the content the way
  `Message.text()`'s own "just give me the words" helper otherwise
  would — the same "never answer as if unsupported content was never
  sent" discipline the Anthropic/OpenAI/Google adapters already apply
  to their own untranslatable block types. Reachable only via an
  explicit model override, since the router's own modality check would
  never route such a request here on its own.
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

### An empty message crashed with a raw torch dtype error instead of a clean empty response

A fresh-eyes sweep found a real gap right next to the budget guard
above: nothing upstream of `FoundryProvider.generate()` validates
non-empty message text (`sarva chat`'s CLI argument and the server's
`ChatRequest.message` both accept `""`), and
`ByteLevelBPETokenizer.encode("")` genuinely returns `[]` by design —
correct behavior for the tokenizer itself, but `generate()` didn't
account for what an empty `prompt_ids` list does downstream.
`budget = config.max_seq_len - len(prompt_ids) - 1` stays positive with
zero prompt tokens, so it sailed past the `budget <= 0` guard straight
into `generate_with_cache`, where `torch.tensor([[]])` — no elements to
infer a dtype from — silently defaults to `float32` instead of
`int64`, crashing the token embedding lookup with a raw, implementation
-leaking `RuntimeError`: `"Expected tensor for argument #1 'indices' to
have one of the following scalar types: Long, Int; but got
torch.FloatTensor instead"`. `AgentLoop`'s broad exception handling
around `provider.generate()` already turns this into a clean
`state=failed` rather than crashing the process, but with that raw
torch/embedding string as the detail instead of a meaningful message.

Even fixing just the dtype wouldn't have actually made an empty prompt
generatable: prefilling zero token positions leaves no "last" logit for
`next_logits = logits[0, -1]` to sample from either — an `IndexError`
one line later. There's genuinely nothing to prefill with, the same
"no useful work to do" shape the `budget <= 0` case already handles
gracefully. Fixed by folding `not prompt_ids` into that same guard, so
an empty message gets the identical clean, empty-response treatment —
not its own special case, not a crash.

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
