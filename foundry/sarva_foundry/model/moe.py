"""sarva_foundry.model.moe — Mixture-of-Experts feedforward (spec §3.6a's
"frontier-class architecture": "fine-grained experts, shared experts,
aux-loss-free load balancing", the K3/DeepSeek-class design). Swaps in
for `SwiGLU` in a `TransformerBlock` via `TransformerConfig.moe` — the
dense baseline is completely untouched when `moe=None` (the default).

Three real, named ideas from the DeepSeek-V2/V3 line, not a generic
"MoE" strawman:

1. **Fine-grained experts**: many smaller experts (`n_experts`, each with
   a hidden dim a fraction of a dense FFN's) rather than few large ones —
   more combinations of experts can be activated per token for the same
   compute budget, empirically improving specialization.
2. **Shared experts**: `n_shared_experts` always-active experts every
   token passes through unconditionally, alongside the routed ones —
   captures common knowledge every token needs, so routed experts don't
   have to re-learn it redundantly.
3. **Aux-loss-free load balancing**: routing uses a per-expert bias added
   to the gate logits for *selection* (top-k) only, updated after the
   fact by `update_expert_bias()` based on realized load — never an
   auxiliary loss term added to the training objective. This is the
   specific innovation that makes it "aux-loss-free": a traditional
   load-balancing loss competes with the language-modeling loss for
   gradient budget; a post-hoc bias nudge doesn't touch the loss at all.

Dense expert loop (`index_add_` per expert), not scatter/gather or
grouped-GEMM kernels: correct and simple at the scale this project
trains at, honestly not what a production MoE serving stack would use —
the same "commodity substrate boundary" `layers.py` draws around
`nn.Linear`, just on the other side of it here (this module's whole
point is testing the *routing/balancing math*, not writing an efficient
CUDA kernel).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sarva_foundry.model.layers import SwiGLU, default_swiglu_hidden_dim


@dataclass
class MoEConfig:
    n_experts: int
    n_experts_per_tok: int
    n_shared_experts: int = 1
    expert_hidden_dim: int | None = None  # default: a quarter of a dense FFN's (fine-grained)
    bias_update_speed: float = 0.01  # DeepSeek-V3's gamma; how fast the routing bias corrects load

    def __post_init__(self) -> None:
        if self.n_experts_per_tok > self.n_experts:
            raise ValueError(
                f"n_experts_per_tok ({self.n_experts_per_tok}) can't exceed "
                f"n_experts ({self.n_experts})"
            )


def _route(gate_logits: Tensor, bias: Tensor, top_k: int) -> tuple[Tensor, Tensor]:
    """Selection uses `gate_logits + bias` (the load-balancing signal);
    weighting uses softmax over the *raw*, unbiased logits of just the
    selected experts, renormalized to sum to 1. Keeping these two uses of
    the logits separate is the entire aux-loss-free mechanism: the bias
    can freely push token traffic toward underloaded experts without
    ever changing how much any expert's output actually counts, which is
    what would make it equivalent to a hidden auxiliary loss term instead
    of a loss-free correction.

    Returns `(expert_indices, weights)`, each shaped `(n_tokens, top_k)`.
    """
    biased = gate_logits + bias
    _, top_idx = biased.topk(top_k, dim=-1)
    selected_logits = gate_logits.gather(-1, top_idx)
    weights = F.softmax(selected_logits, dim=-1)
    return top_idx, weights


class MoEFeedForward(nn.Module):
    def __init__(self, dim: int, config: MoEConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(dim, config.n_experts, bias=False)
        # A buffer, not a Parameter: the bias is updated by a fixed
        # arithmetic rule (update_expert_bias), never by backprop -- it
        # must never accumulate a gradient, which is exactly what
        # "aux-loss-free" means at the tensor level.
        self.register_buffer("expert_bias", torch.zeros(config.n_experts))
        self._last_load: Tensor | None = None

        hidden = config.expert_hidden_dim or max(32, default_swiglu_hidden_dim(dim) // 4)
        self.experts = nn.ModuleList([SwiGLU(dim, hidden) for _ in range(config.n_experts)])
        self.shared_experts = nn.ModuleList(
            [SwiGLU(dim, hidden) for _ in range(config.n_shared_experts)]
        )

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])  # (n_tokens, dim)

        gate_logits = self.gate(x_flat)
        top_idx, weights = _route(gate_logits, self.expert_bias, self.config.n_experts_per_tok)

        out = torch.zeros_like(x_flat)
        load = torch.zeros(self.config.n_experts, device=x.device)
        for e in range(self.config.n_experts):
            token_idx, slot_idx = (top_idx == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            expert_out = self.experts[e](x_flat[token_idx])
            w = weights[token_idx, slot_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, w * expert_out)
            load[e] = token_idx.numel()

        for shared in self.shared_experts:
            out = out + shared(x_flat)

        self._last_load = load.detach()
        return out.reshape(orig_shape)

    def update_expert_bias(self) -> None:
        """Nudge `expert_bias` toward balance using the most recent
        forward's realized expert load — the DeepSeek-V3 aux-loss-free
        update rule. Call once per training step (after `forward`,
        typically after the optimizer step); a no-op if `forward` hasn't
        run yet. Deliberately not auto-invoked by `forward` itself: the
        caller (a training loop) decides the cadence, matching this
        module's `nn.Module` contract of `forward` doing only the forward
        computation."""
        if self._last_load is None:
            return
        avg_load = self._last_load.mean()
        speed = self.config.bias_update_speed
        with torch.no_grad():
            self.expert_bias[self._last_load > avg_load] -= speed
            self.expert_bias[self._last_load < avg_load] += speed
