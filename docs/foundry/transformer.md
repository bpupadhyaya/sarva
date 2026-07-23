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

## Try it

```bash
uv run python examples/03_train_toy_transformer.py     # dense baseline
uv run python examples/07_moe_transformer.py            # MoE, watch the load balance itself
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
actually flatten out as `update_expert_bias()` runs after each step.

## What's next

Pretraining data pipelines, a real (non-toy) training loop with
checkpointing, and the remaining frontier-class extensions from §3.6a —
long-context scaling, native multimodal input — build on this baseline
rather than replacing it.
