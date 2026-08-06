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
from sarva_foundry.model.kv_cache import KVCache
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
        # A real bug found by a fresh-eyes sweep, applying the identical
        # lens already used five times over on MoEConfig's own sibling
        # fields in moe.py (most recently n_shared_experts): range() of a
        # non-positive number is silently empty in Python, so
        # DecoderOnlyTransformer.__init__'s `[TransformerBlock(config)
        # for _ in range(config.n_layers)]` never raised for n_layers <= 0
        # -- it silently built a model with ZERO transformer blocks
        # instead, just an embedding table, a final norm, and a tied
        # output head. forward() still ran and returned plausible-shaped
        # logits from a model that never computes attention or a
        # feedforward pass at all. Reachable through the identical
        # untrusted `config.json` path as MoEConfig's own fields (see
        # foundry_provider.py's own docstring: "a checkpoint saved
        # mid-crash, a hand-edited/rolled-back config.json, or a
        # training-script bug computing this field incorrectly").
        # Confirmed live two ways: direct construction with n_layers=0
        # built successfully with 0 blocks instead of raising; a real
        # trained checkpoint's config.json hand-edited to n_layers=0 and
        # reloaded produced a confusing raw RuntimeError deep inside
        # load_state_dict naming implementation-detail state-dict keys,
        # instead of a clean error at construction time naming the real
        # cause. Unlike n_shared_experts, zero is never legitimate here --
        # a transformer needs at least one block to do anything -- so this
        # uses the same `<= 0` check dim/n_heads-adjacent fields use, not
        # n_shared_experts' negative-only check.
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")

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

    def forward(self, x: Tensor, cache: KVCache | None = None, layer_idx: int = 0) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cache=cache, layer_idx=layer_idx)
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

    def forward(self, token_ids: Tensor, cache: KVCache | None = None) -> Tensor:
        """`cache=None` (the default, and every call site before this
        parameter existed) is exactly the original behavior: every
        position in `token_ids` is recomputed from scratch. Passing a
        `KVCache` (see `sarva_foundry.model.kv_cache`) makes `token_ids`
        mean "the NEW tokens since the cache was last advanced" rather
        than the full sequence — the caller's responsibility, matching
        how `generate_with_cache` (`sarva_foundry.inference`) uses this."""
        return self._forward_embeds(self.tok_embeddings(token_ids), cache=cache)

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
        # A real bug found by a fresh-eyes sweep: this docstring's own
        # "assumes every batch item contributes the same number of
        # image placeholder tokens" claim was never actually checked --
        # the count above is an AGGREGATE (`mask.sum()` and `shape[0] *
        # shape[1]` are both scalars), so it passes whenever the totals
        # happen to match, even when individual batch items' own
        # placeholder counts don't match `image_embeds`' uniform
        # per-item count. Confirmed live: a 2-item batch with 1
        # placeholder in item 0 and 3 in item 1 (total 4) against
        # `image_embeds` shaped (2, 2, dim) (also 4 total) passed this
        # check silently -- `x[mask] = image_embeds.reshape(-1, ...)`
        # then spliced row-major across the FLATTENED mask, so item 1's
        # own first placeholder position received item 0's unused
        # second image embedding instead of any of item 1's real
        # embeddings, with every following position shifted the same
        # way -- cross-example contamination with no exception raised
        # anywhere, silently corrupting both training and inference.
        # Checked per row, not just in aggregate: every batch item must
        # independently match `image_embeds`' own per-item count for
        # the row-major splice below to actually line up positions with
        # the images they're meant to represent.
        per_item_placeholders = mask.sum(dim=1)
        if not bool((per_item_placeholders == image_embeds.shape[1]).all()):
            raise ValueError(
                f"each batch item must contribute exactly {image_embeds.shape[1]} "
                "image-placeholder tokens (image_embeds' own per-item count), got "
                f"per-item counts {per_item_placeholders.tolist()}"
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

    def _forward_embeds(self, x: Tensor, cache: KVCache | None = None) -> Tensor:
        new_len = x.shape[1]
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, cache=cache, layer_idx=layer_idx)
        if cache is not None:
            # Advanced exactly once per forward call, after every layer
            # has written its own key/value at the SAME starting offset
            # (`cache.seq_len` read once at the top of each layer's
            # attention call) -- advancing per-layer instead would shift
            # every subsequent layer's write position by however many
            # layers came before it, corrupting the cache.
            cache.advance(new_len)
        x = self.norm(x)
        return self.lm_head(x)

    def num_parameters(self, include_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if not include_embedding:
            n -= self.tok_embeddings.weight.numel()
        return n
