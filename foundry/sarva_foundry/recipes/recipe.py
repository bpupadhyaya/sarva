"""sarva_foundry.recipes — named, costed training configs (spec §3.6h:
"named, costed configs: laptop-125M -> 1B -> 7B -> 70B"). Named directly
in the design doc's own repo-structure diagram
(`foundry/recipes/  # named, costed configs...`); this closes that gap.

A `Recipe` bundles a real `TransformerConfig` architecture with the
training hyperparameters (token budget, batch size, learning rate,
warmup) that go with it at that scale, plus `compute_estimate()` — a
real FLOPs-based cost estimate, not a fabricated dollar figure. The
formula (`6 * N * D`, N = non-embedding-ish total params, D = training
tokens) is the standard dense-transformer training-FLOPs approximation
from Kaplan et al. 2020 / Hoffmann et al. 2022 ("Chinchilla") — well
known, checkable, not invented for this project. `compute_estimate`
takes hardware throughput (FLOP/s) and price ($/hour) as EXPLICIT
caller-supplied arguments rather than hardcoding a specific GPU/price —
GPU pricing and availability change constantly, and this project's
standing practice (see the model registry's refusal to add
OpenAI/Google pricing without verified-current data) is to never
present an unverified number as current fact. "Costed" here means "you
can compute a real cost from real inputs," not "we assert a fixed
dollar amount."

`Recipe.param_count()` is computed analytically from the architecture's
own weight-matrix shapes (the exact same shapes `DecoderOnlyTransformer`
actually allocates — attention's four projections, SwiGLU's three
matrices, the tied embedding table) rather than by instantiating the
model, which is the point: the `large`/`xlarge` recipes below are
FAR too big to instantiate on a laptop (confirmed empirically —
constructing an ~70B-parameter model here was killed by the OS for
using too much memory, not a hypothetical concern). The formula itself
is verified exact, not assumed, against two recipes small enough to
actually instantiate (`small` and `medium` below) — both match
`DecoderOnlyTransformer(config).num_parameters()` bit-for-bit;
`tests/foundry/test_recipes.py` pins this.
"""

from __future__ import annotations

from dataclasses import dataclass

from sarva_foundry.model import TransformerConfig


@dataclass(frozen=True)
class ComputeEstimate:
    train_flops: float
    gpu_hours: float
    estimated_cost_usd: float


@dataclass(frozen=True)
class Recipe:
    name: str
    description: str
    model: TransformerConfig
    total_tokens: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    # Whether this project's own tests/examples actually instantiate and
    # train this recipe's model, vs. it being an architecture-only
    # specification too large to run here (see module docstring) --
    # stated explicitly rather than left for a reader to guess.
    runnable_here: bool

    def param_count(self) -> int:
        """Exact analytic parameter count from the architecture's own
        weight shapes -- see module docstring for why this isn't just
        `DecoderOnlyTransformer(self.model).num_parameters()` at every
        scale, and `test_recipes.py` for where it's verified exact
        against a real instantiated model."""
        cfg = self.model
        head_dim = cfg.dim // cfg.n_heads
        # `__post_init__` always resolves `hidden_dim` to a concrete value
        # (the default when unset) -- never None on a constructed config.
        hidden_dim = cfg.hidden_dim
        assert hidden_dim is not None
        # wq, wo: dim -> dim each. wk, wv: dim -> n_kv_heads*head_dim each.
        attn = 2 * cfg.dim * cfg.dim + 2 * cfg.dim * (cfg.n_kv_heads * head_dim)
        # SwiGLU: three dim<->hidden_dim matrices (gate, up, down).
        mlp = 3 * cfg.dim * hidden_dim
        norms = 2 * cfg.dim  # attn_norm + mlp_norm weight vectors
        per_layer = attn + mlp + norms
        # Tied embedding/unembedding (counted once) + final norm.
        return cfg.n_layers * per_layer + cfg.vocab_size * cfg.dim + cfg.dim

    def compute_estimate(self, flops_per_second: float, dollars_per_hour: float) -> ComputeEstimate:
        """`flops_per_second`/`dollars_per_hour` describe the hardware
        the caller intends to train on -- e.g. a specific GPU's published
        (dense, non-sparse) FLOP/s figure and its current market $/hour,
        both of which the caller must supply themselves (see module
        docstring for why this function doesn't assume any specific
        hardware or price)."""
        flops = 6.0 * self.param_count() * self.total_tokens
        gpu_hours = flops / flops_per_second / 3600.0
        return ComputeEstimate(
            train_flops=flops,
            gpu_hours=gpu_hours,
            estimated_cost_usd=gpu_hours * dollars_per_hour,
        )
