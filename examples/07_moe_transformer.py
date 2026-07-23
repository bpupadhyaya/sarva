"""Example 07 — Train a Mixture-of-Experts transformer, and watch the
aux-loss-free router balance itself.

Same tiny toy-corpus setup as example 03, but `TransformerConfig.moe` is
set: every block's dense SwiGLU feedforward is replaced by a routed
Mixture-of-Experts feedforward (`sarva_foundry.model.moe`) — fine-grained
experts, an always-active shared expert, and a routing bias updated by a
fixed rule (`update_expert_bias()`) rather than an auxiliary loss term.
Watch the per-expert token counts printed every 50 steps: with the bias
update wired into the training loop, load starts skewed (routing is
driven only by a freshly-initialized, near-random gate) and visibly
flattens out over training — the aux-loss-free mechanism actually
working, not just present in the code.

Run: uv run python examples/07_moe_transformer.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sarva_foundry.model import MoEConfig, MoEFeedForward, TransformerConfig
from sarva_foundry.model.transformer import DecoderOnlyTransformer
from sarva_foundry.tokenizer import ByteLevelBPETokenizer

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox is quick and the dog is lazy",
]


def _expert_loads(model: DecoderOnlyTransformer) -> list[torch.Tensor]:
    """One load tensor per MoE layer, from each layer's most recent forward."""
    loads = []
    for layer in model.layers:
        if isinstance(layer.mlp, MoEFeedForward) and layer.mlp._last_load is not None:
            loads.append(layer.mlp._last_load)
    return loads


def main() -> None:
    torch.manual_seed(0)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(CORPUS, vocab_size=300)

    text = "the quick brown fox jumps over the lazy dog"
    ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    inputs, targets = ids[:, :-1], ids[:, 1:]

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=inputs.shape[1],
        moe=MoEConfig(n_experts=8, n_experts_per_tok=2, n_shared_experts=1),
    )
    model = DecoderOnlyTransformer(config)
    print(
        f"Model: {model.num_parameters():,} parameters, "
        f"{config.moe.n_experts} experts/layer (top-{config.moe.n_experts_per_tok} routed "
        f"+ {config.moe.n_shared_experts} shared)"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(200):
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # The aux-loss-free update: a fixed arithmetic nudge based on the
        # forward pass's realized expert load, applied AFTER the
        # optimizer step -- never part of `loss`, never touching a
        # gradient. This is what "aux-loss-free" means in practice, not
        # just in the module's docstring.
        for layer in model.layers:
            if isinstance(layer.mlp, MoEFeedForward):
                layer.mlp.update_expert_bias()

        if step % 50 == 0 or step == 199:
            loads = _expert_loads(model)
            layer0_load = loads[0].tolist() if loads else []
            print(f"step {step:3d}  loss {loss.item():.4f}  layer0 expert load {layer0_load}")

    print(
        "\nExpert load in layer 0 should visibly flatten from step 0's skew toward "
        "something closer to uniform by step 199 -- the routing bias correcting "
        "itself purely from realized load, with zero contribution to `loss` above."
    )


if __name__ == "__main__":
    main()
