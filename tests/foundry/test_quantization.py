"""Conformance tests for sarva_foundry.quantization — real int8
weight-only quantization (spec §3.6f). The bar isn't "produces int8
tensors of the right shape," it's three real, separately falsifiable
claims: the round-trip error is provably bounded (not just "small"),
the storage reduction is a real measured byte count (not an assumed
ratio), and a genuinely trained model's real loss moves measurably but
not catastrophically after quantization (not a no-op, not a wipeout)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sarva_foundry.model import DecoderOnlyTransformer, TransformerConfig
from sarva_foundry.quantization import (
    apply_quantized_weights,
    quantize_linear,
    quantize_model,
    quantized_linear_forward,
)
from sarva_foundry.train import Trainer
from torch import nn


def _tiny_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=50, dim=32, n_layers=3, n_heads=4, n_kv_heads=2, max_seq_len=64
    )


def test_quantize_linear_roundtrip_error_never_exceeds_half_a_scale_step():
    # Round-to-nearest's own provable bound: every element of
    # |dequantize() - original| <= scale/2 for that row, since int8
    # rounds to the nearest representable multiple of scale.
    torch.manual_seed(0)
    layer = nn.Linear(64, 32, bias=False)
    q = quantize_linear(layer)
    error = (q.dequantize() - layer.weight).abs()
    bound = (q.scale / 2).unsqueeze(-1).expand_as(error)
    assert (error <= bound + 1e-6).all()


def test_quantize_linear_handles_an_all_zero_output_row_without_dividing_by_zero():
    layer = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        layer.weight[1].zero_()
    q = quantize_linear(layer)
    assert torch.isfinite(q.scale).all()
    assert torch.equal(q.weight_int8[1], torch.zeros(4, dtype=torch.int8))


def test_quantize_linear_preserves_bias_unquantized():
    layer = nn.Linear(4, 3, bias=True)
    q = quantize_linear(layer)
    assert torch.equal(q.bias, layer.bias)


def test_quantized_linear_forward_matches_a_manual_dequantize_then_linear():
    torch.manual_seed(0)
    layer = nn.Linear(16, 8, bias=True)
    q = quantize_linear(layer)
    x = torch.randn(2, 16)
    expected = F.linear(x, q.dequantize(), q.bias)
    assert torch.equal(quantized_linear_forward(q, x), expected)


def test_quantized_storage_is_measurably_smaller_than_float32():
    # Real byte counts, not an assumed 4x — a small scale vector adds
    # real overhead, so the true ratio is somewhat below the naive
    # int8-vs-float32 ideal, especially for narrow layers.
    layer = nn.Linear(512, 512, bias=False)
    q = quantize_linear(layer)
    original_bytes = layer.weight.numel() * layer.weight.element_size()
    assert q.nbytes() < original_bytes
    assert original_bytes / q.nbytes() > 3.5


def test_quantize_model_finds_every_linear_submodule_by_its_dotted_name():
    torch.manual_seed(0)
    model = DecoderOnlyTransformer(_tiny_config())
    quantized = quantize_model(model)
    expected_names = {name for name, m in model.named_modules() if isinstance(m, nn.Linear)}
    assert set(quantized) == expected_names
    assert "layers.0.attn.wq" in quantized
    assert "layers.0.mlp.w1" in quantized
    assert "lm_head" in quantized


def test_apply_quantized_weights_preserves_tied_lm_head_and_embedding_identity():
    # transformer.py ties lm_head.weight to tok_embeddings.weight (the
    # same Parameter object). quantize_model treats lm_head as an
    # ordinary nn.Linear and quantizes it independently; the real
    # question is whether apply_quantized_weights's in-place `.data =`
    # assignment breaks that tie. It doesn't -- both names still point
    # at the identical Parameter object, so mutating one's `.data`
    # necessarily mutates the other's too. Verified directly rather
    # than assumed from how weight tying happens to be implemented.
    torch.manual_seed(0)
    model = DecoderOnlyTransformer(_tiny_config())
    assert model.lm_head.weight is model.tok_embeddings.weight

    quantized = quantize_model(model)
    apply_quantized_weights(model, quantized)

    assert model.lm_head.weight is model.tok_embeddings.weight
    assert torch.equal(model.lm_head.weight.data, quantized["lm_head"].dequantize())


def test_apply_quantized_weights_actually_changes_the_live_model_forward_pass():
    # Proves apply_quantized_weights isn't a no-op: the same input
    # produces different logits before and after.
    torch.manual_seed(0)
    model = DecoderOnlyTransformer(_tiny_config())
    model.eval()
    ids = torch.randint(0, 50, (1, 10))
    with torch.no_grad():
        before = model(ids)
    apply_quantized_weights(model, quantize_model(model))
    with torch.no_grad():
        after = model(ids)
    assert not torch.equal(before, after)


def test_quantizing_a_real_trained_model_moves_loss_measurably_but_not_catastrophically():
    # The honest bar this project holds itself to (mirrors the
    # ablation harness's "positive control" pattern): don't just check
    # the math on random weights, train a real tiny model on a real
    # objective, quantize it, and confirm the loss actually moves (this
    # isn't a no-op) while staying bounded (this isn't a destructive
    # bug masquerading as "quantization").
    torch.manual_seed(1)
    config = TransformerConfig(
        vocab_size=20, dim=16, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=16
    )
    model = DecoderOnlyTransformer(config)
    trainer = Trainer(model)
    seq_len = 8
    x = torch.arange(seq_len).unsqueeze(0) % config.vocab_size
    y = (x + 1) % config.vocab_size
    for _ in range(200):
        trainer.train_step(x, y)

    model.eval()
    with torch.no_grad():
        loss_before = F.cross_entropy(model(x).view(-1, config.vocab_size), y.view(-1)).item()

    apply_quantized_weights(model, quantize_model(model))

    with torch.no_grad():
        loss_after = F.cross_entropy(model(x).view(-1, config.vocab_size), y.view(-1)).item()

    assert loss_before < 0.05  # sanity: training genuinely converged first
    assert loss_after != loss_before  # quantization is not a no-op
    assert loss_after < loss_before + 0.5  # ...but not a catastrophic regression
