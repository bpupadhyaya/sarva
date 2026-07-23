"""sarva_foundry.quantization — real int8 weight-only quantization
(spec §3.6f: "inference/serving stack" names "KV-cache, paged attention,
quantization" together; KV-cache is built (`sarva_foundry.model.
kv_cache`), batching/paged attention is real, separate, deliberately
deferred scope (it touches the same internals a naive change could
easily get subtly wrong), and quantization — genuinely separable from
both — is what this module closes.

Real int8 round-to-nearest quantization, per output channel (one scale
per row of a `Linear` layer's weight matrix, not one scale for the
whole matrix — the standard choice, since different output channels
can have very different weight magnitudes, and a single global scale
would waste int8's range on whichever channel has the largest
magnitude). `scale = max(|weight_row|) / 127`; `weight_int8 =
round(weight_row / scale)`, clamped to `[-127, 127]` (not `-128` —
`int8`'s full range is `[-128, 127]`, but using a symmetric `[-127,
127]` range means quantizing back has no off-by-one asymmetry between
the largest positive and negative representable values).

**Honestly scoped, not overclaimed:** this reduces STORAGE (a real,
measured ~4x reduction from float32's 4 bytes/element to int8's 1,
verified directly against actual tensor byte counts) and demonstrates
the real accuracy cost of quantizing a trained model's weights. It does
NOT speed up compute — there's no real int8 GEMM kernel here,
`dequantize()` converts back to float32 before the matmul runs, the
same "commodity substrate" line this project draws around `torch.matmul`
itself (see `layers.py`'s own module docstring). A real quantized
*inference* server that keeps weights in int8 end-to-end and uses an
actual int8 kernel is separate, deferred serving-optimization work, the
same category `sarva.providers.foundry_provider`'s own docstring already
names batching as.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class QuantizedLinear:
    weight_int8: Tensor  # int8, same shape as the original nn.Linear.weight
    scale: Tensor  # float32, one entry per output channel (row)
    bias: Tensor | None

    def dequantize(self) -> Tensor:
        """Reconstructs a float32 weight matrix — never bit-identical to
        the original (that's the whole point of quantization: it's lossy),
        but within a provable per-element bound of `scale/2` (round-to-
        nearest's own error bound), checked directly in
        `test_quantization.py` rather than assumed from the formula."""
        return self.weight_int8.float() * self.scale.unsqueeze(-1)

    def nbytes(self) -> int:
        """Real bytes actually used by the quantized representation —
        int8 weights (1 byte/element) plus the float32 per-channel scales
        (4 bytes/channel, negligible for any real-sized matrix) — for
        comparing directly against the original float32 weight's real
        byte count, not an assumed compression ratio."""
        bias_bytes = self.bias.numel() * self.bias.element_size() if self.bias is not None else 0
        return self.weight_int8.numel() + self.scale.numel() * 4 + bias_bytes


def quantize_linear(layer: nn.Linear) -> QuantizedLinear:
    weight = layer.weight.detach()
    # clamp_min guards a real edge case, not a hypothetical one: an
    # all-zero output channel (a legitimately possible, if unusual,
    # trained weight row) would otherwise divide by zero computing scale.
    scale = (weight.abs().amax(dim=1) / 127.0).clamp_min(1e-8)
    weight_int8 = torch.round(weight / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    bias = layer.bias.detach().clone() if layer.bias is not None else None
    return QuantizedLinear(weight_int8=weight_int8, scale=scale, bias=bias)


def quantized_linear_forward(q: QuantizedLinear, x: Tensor) -> Tensor:
    return F.linear(x, q.dequantize(), q.bias)


def quantize_model(model: nn.Module) -> dict[str, QuantizedLinear]:
    """Every `nn.Linear` submodule in `model`, quantized — the real,
    storable representation an actual quantized checkpoint would keep
    (int8 + per-channel scales), keyed by the same dotted name
    `model.named_modules()` uses, so a caller can round-trip a specific
    layer back to `model.get_submodule(name)` unambiguously."""
    return {
        name: quantize_linear(module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }


def apply_quantized_weights(model: nn.Module, quantized: dict[str, QuantizedLinear]) -> None:
    """Overwrites every quantized layer's live weight with its
    *dequantized* (round-tripped through int8) value, in place — lets a
    caller run a real forward pass through the same model object to
    measure quantization's actual accuracy impact directly (see
    `test_quantization.py`'s end-to-end loss-comparison test).

    Deliberately NOT how a real quantized inference server would serve
    requests (that would keep every layer in its compact int8+scale form
    the whole time, dequantizing only the one layer currently executing,
    never materializing the entire model back to float32 at once) — this
    function exists to answer "how much does quantizing this model
    actually cost in accuracy," not to demonstrate the memory-saving
    serving path, which stays separate, deferred work."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in quantized:
            with torch.no_grad():
                module.weight.data = quantized[name].dequantize()
                if module.bias is not None and quantized[name].bias is not None:
                    module.bias.data = quantized[name].bias.clone()
