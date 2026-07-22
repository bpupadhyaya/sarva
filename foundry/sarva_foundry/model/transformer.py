"""The teaching-baseline dense decoder-only transformer (spec §3.6a): the
composable pre-norm residual architecture every LLaMA-class model uses —
attention and feedforward each wrapped in `x = x + sublayer(norm(x))`.
Frontier-class extensions (MoE routing, long-context scaling, native
multimodal input) build on this baseline rather than replacing it; see
BUILD-JOURNAL.md for what's implemented so far.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from sarva_foundry.model.attention import GroupedQueryAttention
from sarva_foundry.model.layers import RMSNorm, SwiGLU, default_swiglu_hidden_dim


@dataclass
class TransformerConfig:
    vocab_size: int
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    hidden_dim: int | None = None  # default: default_swiglu_hidden_dim(dim)

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")
        if self.hidden_dim is None:
            self.hidden_dim = default_swiglu_hidden_dim(self.dim)

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(
            dim=config.dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
        )
        self.mlp_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.mlp = SwiGLU(config.dim, config.hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    """Token ids in, next-token logits out. `forward` takes no attention
    mask parameter — causality is enforced inside `GroupedQueryAttention`
    unconditionally, so there is no way to accidentally call this model
    in a non-causal configuration."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        # Weight tying (Press & Wolf 2017): the embedding and unembedding
        # are the same matrix, transposed. Standard for this model scale —
        # cuts embedding-table parameters in half with no quality loss.
        self.lm_head.weight = self.tok_embeddings.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        x = self.tok_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    def num_parameters(self, include_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if not include_embedding:
            n -= self.tok_embeddings.weight.numel()
        return n
