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
from sarva_foundry.model.layers import RMSNorm, RopeScalingConfig, SwiGLU, default_swiglu_hidden_dim
from sarva_foundry.model.moe import MoEConfig, MoEFeedForward


@dataclass
class TransformerConfig:
    vocab_size: int
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rope_scaling: RopeScalingConfig | None = None  # None (default): unscaled RoPE, unchanged
    norm_eps: float = 1e-6
    hidden_dim: int | None = None  # default: default_swiglu_hidden_dim(dim)
    moe: MoEConfig | None = None  # None (default): dense SwiGLU FFN, unchanged

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
            rope_scaling=config.rope_scaling,
        )
        self.mlp_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.mlp: nn.Module
        if config.moe is not None:
            self.mlp = MoEFeedForward(config.dim, config.moe)
        else:
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
        return self._forward_embeds(self.tok_embeddings(token_ids))

    def embed_multimodal(
        self, token_ids: Tensor, image_embeds: Tensor, image_token_id: int
    ) -> Tensor:
        """Text token embeddings, with every occurrence of
        `image_token_id` in `token_ids` replaced, in row-major
        (batch-then-sequence) order, by the next row of `image_embeds`
        `(batch, n_image_tokens, dim)` — the placeholder-token splicing
        every LLaVA-class VLM uses, so the causal decoder underneath sees
        one unified sequence and never needs to know which positions
        came from an image. `image_embeds` must already be projected to
        `self.config.dim` (see `vision.Projector`) — this method does no
        projection itself. Assumes every batch item contributes the same
        number of image placeholder tokens, matching `image_embeds`'
        uniform per-item token count (one image per example, or images
        of the same patch count)."""
        x = self.tok_embeddings(token_ids)
        mask = token_ids == image_token_id
        n_placeholders = int(mask.sum())
        n_image_tokens = image_embeds.shape[0] * image_embeds.shape[1]
        if n_placeholders != n_image_tokens:
            raise ValueError(
                f"{n_placeholders} image-placeholder tokens in token_ids but "
                f"{n_image_tokens} image embeddings were provided"
            )
        x = x.clone()
        x[mask] = image_embeds.reshape(-1, image_embeds.shape[-1]).to(x.dtype)
        return x

    def forward_multimodal(
        self, token_ids: Tensor, image_embeds: Tensor, image_token_id: int
    ) -> Tensor:
        """Like `forward`, but every `image_token_id` position in
        `token_ids` is embedded from `image_embeds` instead of the token
        embedding table — the rest of the decoder (causal self-attention,
        FFN, norm, head) is exactly the same code path as text-only
        `forward`, unaware anything about the sequence came from an
        image."""
        return self._forward_embeds(self.embed_multimodal(token_ids, image_embeds, image_token_id))

    def _forward_embeds(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    def num_parameters(self, include_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if not include_embedding:
            n -= self.tok_embeddings.weight.numel()
        return n
