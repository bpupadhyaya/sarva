"""Example 08 — Long-context RoPE scaling: linear interpolation vs
NTK-aware scaling, made visible.

Neither technique is best demonstrated by a training run (extending
real context length takes a real long-context training/fine-tuning
pass, well beyond this project's toy scale) — the useful, honest thing
to show at this scale is the actual math each one changes. Two direct
demonstrations, each picked because the effect is actually visible at
this scale (a naive "print cos(angle) at position 50" table would show
numbers indistinguishable to 4 decimal places for the lowest-frequency
dimension — real RoPE frequencies are tiny by design, and the whole
reason long-context scaling matters is that the effect only becomes
significant over *thousands* of positions, not dozens):

1. **NTK's defining property, frequency by frequency** (position-
   independent): the per-dimension rotation rate itself, scaled vs
   unscaled. The highest-frequency dimension's ratio is exactly 1.0
   (completely unaffected); the lowest-frequency dimension's ratio is
   exactly `1/factor` (fully stretched) — visible in raw numbers, not
   asymptotically after enough positions.
2. **Linear scaling's defining property**: position compression. The
   angle at raw table index `i * factor` in a linearly-scaled table
   must exactly equal the angle at raw index `i` in the unscaled
   table — printed side by side to show the match directly.

Run: uv run python examples/08_long_context_rope_scaling.py
"""

from __future__ import annotations

import torch
from sarva_foundry.model import RopeScalingConfig, TransformerConfig
from sarva_foundry.model.layers import precompute_rope
from sarva_foundry.model.transformer import DecoderOnlyTransformer

HEAD_DIM = 16
THETA = 10000.0
FACTOR = 4.0


def _inv_freq(theta: float) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))


def main() -> None:
    print(f"head_dim={HEAD_DIM}, theta={THETA}, scaling factor={FACTOR}\n")

    # --- 1. NTK: per-dimension frequency ratio, no position needed ---
    base_freq = _inv_freq(THETA)
    ntk_theta = THETA * (FACTOR ** (HEAD_DIM / (HEAD_DIM - 2)))
    ntk_freq = _inv_freq(ntk_theta)
    ratio = ntk_freq / base_freq

    print("NTK scaling -- per-dimension rotation-rate ratio (ntk / unscaled):")
    print(f"{'dim':>5} {'unscaled freq':>15} {'ntk freq':>12} {'ratio':>8}")
    for d in range(len(base_freq)):
        print(
            f"{d:>5} {base_freq[d].item():>15.6f} "
            f"{ntk_freq[d].item():>12.6f} {ratio[d].item():>8.4f}"
        )
    print(
        f"-> dim 0 (highest frequency, most local): ratio = {ratio[0].item():.4f} "
        "(exactly 1.0 -- completely unaffected).\n"
        f"-> last dim (lowest frequency, most long-range): ratio = {ratio[-1].item():.4f} "
        f"(exactly 1/factor = {1 / FACTOR:.4f} -- fully stretched).\n"
    )

    # --- 2. Linear: position compression, shown as an exact equivalence ---
    cos_base, _ = precompute_rope(HEAD_DIM, 16, THETA)
    cos_linear, _ = precompute_rope(
        HEAD_DIM, 16 * int(FACTOR), THETA, scaling=RopeScalingConfig("linear", FACTOR)
    )
    print("Linear scaling -- angle at scaled index i*factor equals angle at unscaled index i:")
    print(f"{'i':>5} {'unscaled[i]':>14} {'linear[i*factor]':>18} {'match?':>8}")
    for i in [1, 4, 9, 15]:
        a, b = cos_base[i, 2].item(), cos_linear[i * int(FACTOR), 2].item()
        print(f"{i:>5} {a:>14.6f} {b:>18.6f} {'yes' if abs(a - b) < 1e-5 else 'no':>8}")
    print(
        "-> a linearly-scaled table covering 4x the positions rotates at "
        "raw index 4*i exactly the way the unscaled table rotates at index "
        "i -- the model sees the same rotation range, just spread over "
        "more actual tokens.\n"
    )

    # A real forward pass through a scaled model -- proves the wiring
    # (TransformerConfig.rope_scaling -> GroupedQueryAttention) actually
    # produces a runnable model, not just a standalone table function.
    config = TransformerConfig(
        vocab_size=50,
        dim=HEAD_DIM * 2,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=64,
        rope_scaling=RopeScalingConfig("ntk", FACTOR),
    )
    model = DecoderOnlyTransformer(config)
    tokens = torch.randint(0, 50, (1, 64))
    logits = model(tokens)
    print(
        f"NTK-scaled model forward pass: input {tuple(tokens.shape)} "
        f"-> logits {tuple(logits.shape)}"
    )


if __name__ == "__main__":
    main()
