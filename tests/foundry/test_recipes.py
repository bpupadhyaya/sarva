"""Conformance tests for sarva_foundry.recipes (spec §3.6h: "named,
costed configs: laptop-125M -> 1B -> 7B -> 70B"). The property that
actually matters isn't "the dataclass holds the fields it's declared
to" -- it's that `Recipe.param_count()`'s analytic formula produces the
EXACT same number a real instantiated model would, verified directly
against real models at the two scales small enough to instantiate on a
laptop, not assumed to generalize from the formula looking right on
paper."""

from __future__ import annotations

from sarva_foundry.model import DecoderOnlyTransformer
from sarva_foundry.recipes import ALL_RECIPES, LAPTOP_125M, SCALE_1B, SCALE_7B, SCALE_70B
from sarva_foundry.recipes.recipe import Recipe


def test_laptop_125m_param_count_matches_a_real_instantiated_model():
    real = DecoderOnlyTransformer(LAPTOP_125M.model).num_parameters()
    assert LAPTOP_125M.param_count() == real
    assert real == 125_264_640


def test_1b_param_count_matches_a_real_instantiated_model():
    real = DecoderOnlyTransformer(SCALE_1B.model).num_parameters()
    assert SCALE_1B.param_count() == real
    assert real == 1_057_581_056


def test_7b_and_70b_param_counts_are_pinned_exact_without_instantiating():
    # Deliberately NOT instantiated -- building an ~70B-parameter model
    # was tried directly while writing this recipe and was killed by the
    # OS for memory use on a laptop (see recipes/named.py's docstring).
    # These are the exact values the same verified-correct formula
    # (test_recipes.py's other two tests confirm it's exact, not
    # approximate) produces for these configs.
    assert SCALE_7B.param_count() == 5_802_037_248
    assert SCALE_70B.param_count() == 55_628_275_712


def test_runnable_here_flags_match_what_this_project_actually_runs():
    assert LAPTOP_125M.runnable_here is True
    assert SCALE_1B.runnable_here is True
    assert SCALE_7B.runnable_here is False
    assert SCALE_70B.runnable_here is False


def test_compute_estimate_uses_the_standard_6nd_flops_formula_exactly():
    recipe = Recipe(
        name="toy",
        description="",
        model=LAPTOP_125M.model,
        total_tokens=1000,
        batch_size=1,
        learning_rate=1e-3,
        warmup_steps=0,
        runnable_here=True,
    )
    n = recipe.param_count()
    estimate = recipe.compute_estimate(flops_per_second=1e15, dollars_per_hour=2.0)

    assert estimate.train_flops == 6.0 * n * 1000
    expected_hours = estimate.train_flops / 1e15 / 3600.0
    assert estimate.gpu_hours == expected_hours
    assert estimate.estimated_cost_usd == expected_hours * 2.0


def test_compute_estimate_scales_linearly_with_token_budget():
    small_budget = Recipe(
        name="a",
        description="",
        model=LAPTOP_125M.model,
        total_tokens=1_000_000,
        batch_size=1,
        learning_rate=1e-3,
        warmup_steps=0,
        runnable_here=True,
    )
    double_budget = Recipe(
        name="b",
        description="",
        model=LAPTOP_125M.model,
        total_tokens=2_000_000,
        batch_size=1,
        learning_rate=1e-3,
        warmup_steps=0,
        runnable_here=True,
    )
    e1 = small_budget.compute_estimate(flops_per_second=1e15, dollars_per_hour=2.0)
    e2 = double_budget.compute_estimate(flops_per_second=1e15, dollars_per_hour=2.0)
    assert e2.train_flops == 2 * e1.train_flops
    assert e2.estimated_cost_usd == 2 * e1.estimated_cost_usd


def test_all_recipes_lists_every_named_recipe_in_scale_order():
    assert [r.name for r in ALL_RECIPES] == ["laptop-125M", "1B", "7B", "70B"]
