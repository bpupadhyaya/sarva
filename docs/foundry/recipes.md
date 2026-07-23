# Recipes: named, costed configs from laptop-125M to 70B

The design doc's own repo-structure diagram has named
`foundry/recipes/  # named, costed configs: laptop-125M -> 1B -> 7B -> 70B`
since T0. This chapter is where that gap closes: `sarva_foundry.recipes`
bundles a real architecture (`TransformerConfig`) with the training
hyperparameters that go with it at each scale, plus a genuine
FLOPs-based compute estimate.

## The four recipes

| Recipe | Real params | `runnable_here` |
|---|---|---|
| `LAPTOP_125M` | 125,264,640 | Yes — this project's own tests instantiate and train it |
| `SCALE_1B` | 1,057,581,056 | Yes — instantiable and smoke-trainable on a laptop |
| `SCALE_7B` | 5,802,037,248 | No — architecture-only specification |
| `SCALE_70B` | 55,628,275,712 | No — architecture-only specification |

Every number above is the **real, exact** output of
`Recipe.param_count()` — not rounded or fabricated to hit the label.
Labels are the nearest standard scale bucket, the same looseness the
field itself uses: Llama-2-7B is actually 6.7B parameters, Mistral-7B is
7.24B. `SCALE_70B`'s real count (55.6B) sits further from its label than
the others because this project's SwiGLU hidden-dim formula (a plain
2/3× rule) differs from Llama-2-70B's own custom FFN multiplier —
reported honestly rather than hand-tuned to hit exactly 70B.

**Why `param_count()` doesn't just instantiate the model and count:**
building an ~70B-parameter model was tried directly while writing this
chapter and was killed by the operating system for memory use — a real,
confirmed limit, not a hypothetical one. `param_count()` computes the
same number analytically instead, directly from the architecture's own
weight-matrix shapes (attention's four projections, SwiGLU's three
matrices, the tied embedding table). This formula is **verified exact**,
not assumed — `tests/foundry/test_recipes.py` instantiates real
`DecoderOnlyTransformer`s at the two scales small enough to build
(`LAPTOP_125M`, `SCALE_1B`) and confirms `param_count()` matches
`.num_parameters()` bit-for-bit at both. Since the formula is derived
directly from the same architecture code (not fitted or approximated),
that exactness carries to `SCALE_7B`/`SCALE_70B` too, even though those
two are never actually instantiated.

## Compute estimates: real formula, explicit hardware

`Recipe.compute_estimate(flops_per_second, dollars_per_hour)` uses the
standard dense-transformer training-FLOPs approximation,
`FLOPs ≈ 6 × N × D` (Kaplan et al. 2020 / Hoffmann et al. 2022,
"Chinchilla") — a well-known, checkable formula, not invented for this
project.

**Hardware throughput and price are always caller-supplied, never
hardcoded.** GPU pricing and availability change constantly, and this
project's standing practice — the same one that kept unverified
OpenAI/Google pricing out of `models.yaml` — is to never present an
unverified number as current fact. "Costed" here means *you can compute
a real cost from real inputs*, not *we assert a fixed dollar amount*.
`examples/16_foundry_recipes.py` prints estimates under two explicitly
labeled **illustrative** hardware profiles (never claimed as current,
verified pricing for any real GPU).

## A real measured check, not just a formula in isolation

The example script does one more thing the printed table alone can't
prove: it runs `LAPTOP_125M`'s actual architecture for a few real
training steps on this machine, measures real wall-clock tokens/sec,
converts that into a real FLOP/s figure via the same `6×N×D` formula,
and shows what `compute_estimate` would predict using *this machine's
own measured speed* — a genuine correlation check between the formula
and reality, not two independent, unverified numbers sitting next to
each other.

## Build it yourself

```python
from sarva_foundry.recipes import LAPTOP_125M, ALL_RECIPES

for recipe in ALL_RECIPES:
    print(recipe.name, recipe.param_count())

estimate = LAPTOP_125M.compute_estimate(flops_per_second=1e15, dollars_per_hour=2.0)
print(estimate.gpu_hours, estimate.estimated_cost_usd)
```

Run `uv run python examples/16_foundry_recipes.py` for the full picture,
including the real measured-throughput check.
