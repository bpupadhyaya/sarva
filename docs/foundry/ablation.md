# The ablation harness: trustworthy architecture comparisons

The design doc's own §3 sentence is specific about what this repo is
supposed to let a researcher do: "architecture is composable
[via `TransformerConfig`]... + an **ablation harness** so researchers
can test *new* ideas at small scale with trustworthy comparisons. This
is what 'advance LLMs, not just train them' means concretely."

Until now, that harness didn't exist — confirmed by grep before
starting. Two other docstrings in this codebase cite *other people's*
published ablations (LLaVA-1.5's connector design in `vision.py`, the
long-standing SwiGLU-vs-ReLU comparison in `layers.py`), but Sarva had
no way to actually *run* one of its own. Every "the architecture is
composable" claim had been asserted, never exercised as a real
head-to-head comparison.

## "Trustworthy" is the word that matters

A single, single-seed training run's final loss is genuinely noisy —
weight initialization and data ordering interact with the specific
seed, so comparing two architectures via one run each risks a
conclusion that's really just seed luck dressed up as a finding.
`sarva_foundry.ablation.run_ablation` controls for the two confounds a
naive comparison misses:

- **Identical data, identical order.** Every `AblationArm` trains
  against the same tokenized corpus, chunked the same way, and every
  training step pulls the exact same chunk indices — proven directly in
  `tests/foundry/test_ablation.py`, not just claimed: two arms given the
  *same* model config and the *same* seed produce bit-identical final
  losses, which is only possible if they really did see identical data
  in identical order throughout.
- **Multiple seeds, not one.** Every arm trains across several seeds
  (three by default) and reports **mean and standard deviation**, never
  a single point estimate treated as ground truth.

## Honestly scoped: a real signal, not a fabricated p-value

`AblationResult.is_difference_trustworthy(arm_a, arm_b)` reports one
specific, real, defensible thing: whether the two arms' mean final
losses differ by more than their *combined* standard deviation. That is
**not** a formal hypothesis test or a p-value — computing a genuine
Welch's t-test needs a real t-distribution CDF (an incomplete beta
function this project hasn't implemented), and this project doesn't
approximate statistics it hasn't actually built any more than it
fabricates benchmark numbers or GPU pricing elsewhere in the codebase.
What it does report is checked directly: a deliberately obvious capacity
gap (an 8-dim, 1-layer model vs. a 48-dim, 2-layer one) is correctly
flagged trustworthy; a deliberately marginal change (a purely cosmetic
head-count difference at the same total dimension) is correctly flagged
*not* trustworthy.

## Two real comparisons, run end to end

`examples/18_ablation_harness.py` runs both kinds of result on purpose,
so the harness's honesty is visible, not just asserted:

1. **A positive control** (tiny vs. a reasonably sized model) — the
   harness correctly reports this as trustworthy: the loss gap is far
   larger than the run-to-run noise across seeds.
2. **A genuine architecture question** (dense SwiGLU vs. MoE
   feedforward, the two feedforward options `TransformerConfig` already
   composes between) — at toy scale and this training budget, both
   essentially memorize the small corpus, and the harness honestly
   reports the difference is **not** trustworthy. A real result, either
   way, is more useful than a massaged "winner."

```python
from sarva_foundry.ablation import AblationArm, run_ablation

result = run_ablation(
    arms=[
        AblationArm(name="dense", model_config=dense_config),
        AblationArm(name="moe", model_config=moe_config),
    ],
    token_ids=token_ids,
    seq_len=32,
    batch_size=4,
    steps=150,
    seeds=[0, 1, 2],
)

for arm in result.ranked():
    print(arm.name, arm.mean_final_loss, arm.stdev_final_loss)
print(result.is_difference_trustworthy("dense", "moe"))
```

Run `uv run python examples/18_ablation_harness.py` for the full,
real, printed comparison.
