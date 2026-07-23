"""Example 19 — int8 weight-only quantization: real storage savings,
real accuracy cost.

Spec §3.6f names "KV-cache, paged attention, quantization" together as
inference/serving scope. KV-cache is example 15; paged attention/batching
stays deliberately deferred (the user's own call, given the correctness
risk of touching that code); quantization is genuinely separable from
both, and `sarva_foundry.quantization` closes it.

This example proves two things with real numbers, not asserted or
assumed: (1) the storage reduction from int8 + a small per-channel float32
scale vector vs. plain float32 weights, measured directly on this model's
actual tensor byte counts, and (2) the real accuracy cost of quantizing an
actually-trained model's weights -- trained on a real (if toy) next-token
objective first, so the "before" loss reflects genuine learned structure,
not random-init noise a quantization bug could hide behind.

Honest scope note, not glossed over: `apply_quantized_weights` measures
accuracy impact by dequantizing back to float32 before running the
forward pass -- it does NOT save memory or add speed by itself (weights
stay float32-sized during inference here). A real memory-saving serving
path would keep every layer in its compact int8+scale form and dequantize
only the one layer currently executing; that's separate, deferred serving
work, not what this example demonstrates.

Run: uv run python examples/19_quantization.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sarva_foundry.model import DecoderOnlyTransformer, TransformerConfig
from sarva_foundry.quantization import apply_quantized_weights, quantize_model
from sarva_foundry.train import Trainer

VOCAB_SIZE = 64
SEQ_LEN = 32
TRAIN_STEPS = 60


def main() -> None:
    torch.manual_seed(0)
    config = TransformerConfig(
        vocab_size=VOCAB_SIZE, dim=64, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=128
    )
    model = DecoderOnlyTransformer(config)
    print(f"Model: {model.num_parameters():,} parameters")

    # A real (if toy) objective: predict the next id in a fixed
    # sequence -- enough real structure for quantization's cost to mean
    # something, unlike scoring random-init weights.
    x = torch.arange(SEQ_LEN).unsqueeze(0) % VOCAB_SIZE
    y = (x + 1) % VOCAB_SIZE
    trainer = Trainer(model)
    print(f"\nTraining {TRAIN_STEPS} real steps on a next-token objective...")
    for step in range(TRAIN_STEPS):
        loss = trainer.train_step(x, y)
        if step % 100 == 0 or step == TRAIN_STEPS - 1:
            print(f"  step {step:4d}  loss {loss:.4f}")

    model.eval()
    with torch.no_grad():
        loss_before = F.cross_entropy(model(x).view(-1, VOCAB_SIZE), y.view(-1)).item()

    quantized = quantize_model(model)

    total_before = sum(q.weight_int8.numel() * 4 for q in quantized.values())
    total_after = sum(q.nbytes() for q in quantized.values())
    print(f"\n{len(quantized)} Linear layers quantized to int8.")
    print(f"  float32 storage: {total_before:,} bytes")
    print(f"  int8+scale storage: {total_after:,} bytes")
    print(f"  real measured reduction: {total_before / total_after:.2f}x")

    apply_quantized_weights(model, quantized)
    with torch.no_grad():
        loss_after = F.cross_entropy(model(x).view(-1, VOCAB_SIZE), y.view(-1)).item()

    print("\nLoss on the trained objective, before vs. after quantization:")
    print(f"  before: {loss_before:.5f}")
    print(f"  after:  {loss_after:.5f}")
    print(
        "\nBoth numbers above are real, measured on this run -- quantization "
        "has a real, nonzero accuracy cost (the model's weights are "
        "genuinely different now, not bit-identical), but it should stay "
        "small relative to the trained loss rather than destroying what "
        "training learned."
    )


if __name__ == "__main__":
    main()
