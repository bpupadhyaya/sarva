# The transformer, from scratch

`sarva_foundry.model` — the teaching-baseline dense decoder-only
transformer (design of record §3.6a), the architecture every current
LLaMA/Qwen/Mistral-class model is a variation of. Written directly against
the math in `foundry/sarva_foundry/model/`, not imported from
`transformers`.

## Where "from scratch" stops

`torch.nn.Linear`, `nn.Embedding`, and PyTorch's fused
`scaled_dot_product_attention` kernel are treated as commodity substrate —
the same tier as `torch.matmul`. Sarva's "no black boxes" principle draws
the line at PyTorch/CUDA itself, not at every tensor operation built on
top of it. What's ours is the *model math*: how RMSNorm normalizes, how
RoPE rotates queries and keys, how grouped-query attention shares KV heads
across query groups, how the residual stream is composed layer by layer.

## The building blocks

- **RMSNorm** (`layers.py`) — normalizes by root-mean-square only, no
  mean-centering or bias, matching LLaMA/Mistral/Qwen.
- **RoPE** (`layers.py`) — rotary position embeddings, "rotate-half"
  convention. The property that actually matters: the dot product between
  a rotated query at position *m* and a rotated key at position *n*
  depends only on the *relative* offset *m − n*, never on the absolute
  positions. `tests/foundry/test_model.py` verifies this directly rather
  than trusting a correct-looking implementation.
- **Grouped-query attention** (`attention.py`) — query heads are split
  into groups that each share one KV head (the LLaMA-3/Qwen/Mistral
  middle ground between full multi-head attention's expensive KV cache
  and multi-query attention's quality loss). Causal masking is enforced
  unconditionally inside `forward` — there's no non-causal mode to
  accidentally select.
- **SwiGLU** (`layers.py`) — the gated feedforward every current
  frontier-class open model uses in place of a plain ReLU MLP.
- **`TransformerBlock` / `DecoderOnlyTransformer`** (`transformer.py`) —
  the pre-norm residual composition (`x = x + sublayer(norm(x))`) and the
  full token-ids-in, logits-out model, with the embedding and unembedding
  matrices tied (Press & Wolf 2017).

## Two bugs the test suite exists to catch

Shape-correct code can still be **wrong** in ways shape checks never
surface. Two examples from building this module, both now pinned as
regression tests:

1. **Causal masking that silently doesn't mask.** `is_causal=True` is one
   flag — get it wrong (or apply it to the wrong tensor) and the model
   still produces plausible-looking logits of the right shape while
   quietly leaking future tokens into earlier positions, which would
   invalidate every downstream training run without ever throwing an
   error. `test_causal_masking_prevents_attending_to_future_tokens`
   perturbs only the last token in a sequence and asserts every earlier
   position's output is bit-for-bit unchanged.
2. **RoPE tables that silently truncate past `max_seq_len`.** Found by
   actually running `examples/03_train_toy_transformer.py`'s
   greedy-generation loop (which grows the sequence past the length used
   at training time), not by any unit test — slicing a precomputed
   cos/sin table past its length doesn't raise in Python, it just returns
   something shorter, and the real failure surfaced several calls later
   as a confusing shape-mismatch error deep inside `apply_rope`. Fixed
   with an explicit bounds check at the top of `forward()` that raises
   immediately and clearly instead.

## Try it

```bash
uv run python examples/03_train_toy_transformer.py
```

Trains the real byte-level BPE tokenizer (see the
[tokenizer chapter](tokenizer.md)) on a toy corpus, feeds real token ids
into a ~142K-parameter transformer, trains for 200 steps on CPU in a few
seconds, and greedy-decodes a continuation to show the whole
tokenize → embed → attend → predict → backprop pipeline working end to
end — memorizing (intentionally, at this toy scale) the sentence it was
trained on.

## Mixture-of-Experts: the first frontier-class extension

`sarva_foundry.model.moe` is the first of §3.6a's named frontier-class
extensions — the K3/DeepSeek-class design: **fine-grained experts** (many
smaller experts rather than a few large ones), a **shared expert**
(always active for every token, alongside whichever routed experts get
selected), and **aux-loss-free load balancing**. It swaps in for the
dense baseline via `TransformerConfig.moe` — leave it `None` (the
default) and nothing here changes; set it and every block's `SwiGLU`
feedforward becomes a routed `MoEFeedForward` instead. Composable, not a
fork: the attention stack, RMSNorm, RoPE, and the rest of
`TransformerBlock` are completely untouched either way.

### Why aux-loss-free, specifically

Traditional MoE load balancing adds an auxiliary loss term that
penalizes uneven expert usage — but that loss term competes with the
actual language-modeling loss for gradient budget, and tuning its weight
is its own fragile hyperparameter problem. DeepSeek-V3's alternative:
give each expert a **bias** added to the router's logits, used *only*
for deciding which experts get selected (top-k), never for weighting how
much a selected expert's output counts (that weight comes from a
softmax over the *raw*, unbiased logits of just the selected experts).
After each forward pass, `update_expert_bias()` nudges the bias for
overloaded experts down and underloaded experts up by a fixed amount —
plain arithmetic on a `register_buffer`, not a `Parameter`, so it can
never accumulate a gradient. No loss term anywhere ever sees this
signal — that's the entire meaning of "aux-loss-free."

Keeping selection and weighting genuinely separate is the one detail
that makes this real rather than a relabeled auxiliary loss:
`test_route_bias_changes_selection_but_not_weight_of_a_selected_expert`
pins it directly — a large enough bias forces a different expert to be
selected, and the weight that expert's output receives is still
identical to what an *unbiased* selection of it would have produced.

### A validation gap: `bias_update_speed` had no check at all

A round-42 sweep applying the same "push individually-valid-looking
numerical parameters to their extreme" lens used for RoPE scaling
(below) to `MoEConfig` found `bias_update_speed` had zero validation —
not even a sign check, unlike every other numeric field in this
project. Confirmed live: constructing `MoEFeedForward` with
`bias_update_speed=float("inf")` and calling `update_expert_bias()`
once — exactly the module's own documented "call once per training
step" usage — turned `expert_bias` into `[inf, inf, -inf, -inf]`. On a
**completely fresh, unrelated batch of 500 random tokens**, the two
`+inf`-biased experts captured all 500 selections and the `-inf` ones
captured zero, regardless of what the actual gate logits said —
routing became permanently disconnected from input content, since
`inf -= speed` stays `inf` forever. `bias_update_speed=float("nan")`
produced the same collapse (`torch.topk` treats NaN as always-largest).
In both cases the forward pass's actual output tensor stayed fully
finite and raised no exception — a completely silent defeat of the
"aux-loss-free load balancing" mechanism that is this module's entire
documented purpose. Fixed with `math.isfinite(self.bias_update_speed)
and self.bias_update_speed > 0` in `MoEConfig.__post_init__`, the same
pattern as `RopeScalingConfig`'s fix. Verified by reverting and
confirming the new tests fail with `DID NOT RAISE ValueError`.

### A real test-construction mistake, caught by running it, not shipped

The first draft of the load-balancing convergence test used an
all-zero, frozen gate — reasoning that this would isolate the bias's
effect from noisy real routing signal. Running it showed the opposite of
convergence: with zero per-token signal from the gate, routing became a
pure popularity contest decided entirely by whichever expert currently
had the highest bias, so every single token piled onto one expert each
round — and the "winner" flipped to a *different* single expert every
few rounds as the bias update caught up, rather than smoothly spreading
load across experts. Same load standard deviation before and after,
just relabeled. The fix was realizing the mistake was in the test's
setup, not the algorithm: a real (untouched, randomly-initialized) gate
gives different tokens different raw preferences, so as the bias narrows
the gap between an overloaded and an underloaded expert, only *some*
tokens peel off to the alternative each round — the graceful,
incremental rebalancing the mechanism is actually designed to produce.
`examples/07_moe_transformer.py`'s printed per-step expert-load column
shows this directly on a real (if toy-scale) training run, not just in
an isolated test.

## Long-context scaling: linear interpolation and NTK-aware RoPE

`RopeScalingConfig` is the second frontier-class extension from §3.6a's
list ("position-interpolation/NTK scaling"). Swaps in via
`TransformerConfig.rope_scaling` — leave it `None` (the default) and
`precompute_rope`'s output is bit-for-bit identical to before this
feature existed. Two real, distinct, named techniques, not one generic
"scale factor" knob:

- **Linear** (Chen et al. 2023, position interpolation): every position
  is divided by `factor` before computing rotation angles — the table
  covers `factor` times as many raw positions while every frequency
  rotates proportionally slower, so the model never sees a rotation
  angle outside the range it saw in training. Simple, but compresses the
  *highest*-frequency (most local) dimension right along with the low-
  frequency ones.
- **NTK-aware** (bloc97): instead of touching positions, raises the RoPE
  base `theta` itself. The highest-frequency dimension's rotation rate
  is `theta^0 = 1` regardless of `theta`, so NTK scaling leaves that
  dimension **exactly** unaffected — only the lower-frequency
  (longer-range) dimensions stretch. `test_ntk_scaling_leaves_the_highest_frequency_dimension_unchanged`
  pins this directly (bit-identical to the unscaled table at dimension
  0, for every position), and its direct contrast,
  `test_linear_scaling_does_change_the_highest_frequency_dimension`,
  proves linear scaling genuinely doesn't have this property — these are
  two different techniques, not the same math wearing two names.

Both preserve RoPE's core guarantee — a rotated dot product still
depends only on relative position, never absolute — for a *fixed*
scaling config, verified the same way the unscaled table's relative-
position invariance is verified in `test_model.py`.

**Four real validation gaps found by a sweep specifically targeting
"correctness under extreme/adversarial `factor` values," a genuinely
fresh angle after many rounds spent elsewhere.** All four are silent-
corruption or ugly-crash bugs, not crashes on obviously-malformed
input — every value involved individually passes whatever check
existed at the time, and only produces a NaN/Inf/exception once it
reaches the actual math:

- **`factor=nan` was accepted with no error at all**, since Python's
  `nan <= 0` evaluates to `False` — the original check only rejected
  non-positive values. Confirmed live: `precompute_rope`'s own cos/sin
  tables came back 100% NaN, and a real `DecoderOnlyTransformer`'s
  logits were 100% NaN too, no exception anywhere. Reachable through a
  real, non-synthetic path, not just a direct constructor call —
  `save_checkpoint_bundle`/`load_checkpoint_bundle` round-trip
  `rope_scaling` through a checkpoint's `config.json` via
  `json.dumps`/`json.loads`, and Python's `json` module accepts and
  round-trips the non-standard `NaN` literal by default (confirmed
  live: `json.dumps({"factor": float("nan")})` produces literal
  `{"factor": NaN}`, and `json.loads` on that string returns
  `float("nan")` right back) — a NaN factor in a saved checkpoint would
  silently corrupt every future load of it. `RopeScalingConfig.
  __post_init__` now checks `math.isfinite(factor)`, rejecting NaN and
  ±Inf in the same check that already rejected non-positive values.
- **`factor=1e-300` (linear scaling) — individually finite and
  positive, so it passed every existing check — divides positions down
  to a value that overflows, and `cos`/`sin` of a non-finite angle is
  itself NaN.** Confirmed live: 100% NaN cos/sin tables, no exception,
  the identical silent-corruption shape as the NaN-factor bug, just
  reached through a factor that individually looked valid.
  `precompute_rope` now checks `torch.isfinite(cos).all()`/`sin` after
  computing them and raises a clear `ValueError` naming the actual
  theta/head_dim/max_seq_len/scaling combination that produced it —
  one check that catches this and any other numerically-extreme
  combination this function can't practically enumerate in advance,
  rather than trying to guess a "reasonable" bound for `factor` itself.
- **`factor=1e300` (NTK scaling) raised an uncaught native
  `OverflowError`** from `factor ** (head_dim/(head_dim-2))` exceeding
  float range — a real crash, just not the module's own documented
  `ValueError` contract. Now caught and re-raised as a clear
  `ValueError` naming the factor and head_dim involved.
- **`head_dim=2` with NTK scaling raised an uncaught
  `ZeroDivisionError`.** `head_dim=2` is even, so it passes the
  earlier "head_dim must be even" check, but NTK's own exponent formula
  (`head_dim / (head_dim - 2)`) divides by zero for exactly this value
  — a real, if pathological, edge case (a 2-dimensional attention head
  is an unusual config, but nothing before this fix actually prevented
  it). Now an explicit, named check before the division.

Verified each independently: reverted the fix and watched all four new
tests fail for the exact right reason (the raw `OverflowError`/
`ZeroDivisionError`; `Failed: DID NOT RAISE ValueError` for the two
silent-NaN cases) before re-applying. All 9 pre-existing rope-scaling
tests pass unchanged.

## Native multimodal input: a vision encoder + projector

`sarva_foundry.model.vision` is the third and last named piece of
§3.6a's architecture list ("native multimodal (vision encoder +
projector; audio later)"). Three real, standard LLaVA-class pieces,
each reusing this project's already-tested substrate rather than a
parallel implementation:

- **`PatchEmbed`**: splits an image into non-overlapping patches and
  linearly projects each to `dim`, via a single strided `nn.Conv2d`
  (kernel = stride = patch size) — the standard "patchify" trick.
  `test_patch_embed_matches_manual_flatten_and_linear` proves this is
  exactly equivalent to manually slicing each patch, flattening it, and
  applying one shared linear layer, not just "produces the right shape."
- **`VisionEncoder`**: patchify, then N *bidirectional* transformer
  blocks — reuses `GroupedQueryAttention`/`RMSNorm`/`SwiGLU` with the
  new `causal=False` option (the text decoder's blocks still default to
  `causal=True`, unchanged). An image patch needs to see every other
  patch, not just ones that happen to come "earlier" in some arbitrary
  flatten order.
  `test_vision_encoder_is_genuinely_bidirectional_not_accidentally_causal`
  is the mirror image of `test_model.py`'s causal-masking test:
  perturbing the *first* patch must change the encoder's output at the
  *last* patch too, proving `causal=False` is genuinely wired through
  rather than silently ignored.
- **`Projector`**: a 2-layer MLP with a GELU nonlinearity mapping the
  vision encoder's output dim to the text decoder's `dim` — the
  "connector" every LLaVA-class model uses (LLaVA-1.5's own ablation
  found this beats a single linear projection, which
  `test_projector_is_nonlinear_not_a_disguised_linear_layer` checks
  directly: fit the best possible *linear* map to a few samples and
  confirm it does NOT predict fresh samples the way the real, nonlinear
  `Projector` does).

`DecoderOnlyTransformer.forward_multimodal(token_ids, image_embeds,
image_token_id)` is the splice point: every occurrence of
`image_token_id` in `token_ids` is replaced by the next projected image
embedding, and the *same* causal decoder body runs on the resulting
sequence — the decoder never needs to know which positions came from an
image. The strongest proof this actually works, not just runs:
`test_full_stack_is_trainable_gradients_flow_through_vision_and_text`
asserts every parameter across the vision encoder, the projector, AND
the text decoder receives a real gradient during a real training step —
a broken splice (e.g. an accidental `.detach()`) would silently zero out
the vision/projector gradients while still producing plausible-shaped
logits, which no shape-only test could catch.

**Honestly named simplification:** the vision encoder reuses 1D RoPE
over the flattened patch sequence — the same positional mechanism the
text decoder already has — rather than a 2D-aware scheme (2D RoPE or
learned 2D embeddings) a production vision encoder would use to encode
row/column structure. Real, deferred follow-up work, not silently
assumed equivalent.

## Try it

```bash
uv run python examples/03_train_toy_transformer.py            # dense baseline
uv run python examples/07_moe_transformer.py                   # MoE, watch the load balance itself
uv run python examples/08_long_context_rope_scaling.py         # linear vs NTK RoPE scaling, made visible
uv run python examples/09_multimodal_vision_transformer.py     # vision encoder + projector + decoder, trained
```

The first trains the real byte-level BPE tokenizer (see the
[tokenizer chapter](tokenizer.md)) on a toy corpus, feeds real token ids
into a ~142K-parameter transformer, trains for 200 steps on CPU in a few
seconds, and greedy-decodes a continuation to show the whole
tokenize → embed → attend → predict → backprop pipeline working end to
end — memorizing (intentionally, at this toy scale) the sentence it was
trained on. The second does the same training loop with an
8-expert-per-layer MoE feedforward instead, printing each layer's
per-expert token counts every 50 steps so you can watch the load
actually flatten out as `update_expert_bias()` runs after each step. The
third prints the actual per-dimension frequency ratios NTK scaling
produces (dimension 0's ratio is exactly 1.0, the lowest dimension's is
exactly `1/factor`) and the exact position-index equivalence linear
scaling produces — a training run wouldn't be the honest way to show
either property at toy scale (extending real context length needs a
real long-context fine-tuning pass this project doesn't have data or
compute for yet), but the math itself is fully visible today. The fourth
trains a vision encoder + projector + text decoder together on a
deliberately trivial but real task — a solid red image should make the
model predict one specific token, a solid blue image a different one,
with *identical* surrounding text — so getting both right after training
is only possible if the model is genuinely using the image content, not
guessing from text alone.

## What's next

§3.6a's full named architecture list (dense baseline, MoE, long-context
RoPE scaling, native multimodal input) is now built. What's still ahead:
pretraining data pipelines at real scale and a real (non-toy) training
loop with distributed checkpointing (§3.6d/F1), post-training (§3.6e),
and audio input — named in §3.6a's own multimodal line as "vision
encoder + projector; audio later," and "later" still means later.
