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

## What's next

Pretraining data pipelines, a real (non-toy) training loop with
checkpointing, and the frontier-class extensions from §3.6a — MoE
routing, long-context scaling, native multimodal input — build on this
baseline rather than replacing it.
