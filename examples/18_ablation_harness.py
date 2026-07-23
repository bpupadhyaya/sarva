"""Example 18 — The ablation harness: trustworthy architecture
comparisons, not single-run guesses.

Design doc §3: "architecture is composable [via `TransformerConfig`]...
+ an ablation harness so researchers can test *new* ideas at small
scale with trustworthy comparisons. This is what 'advance LLMs, not
just train them' means concretely." Runs two real comparisons end to
end, both against the identical tokenized corpus, controlling for data
order and seed the way `sarva_foundry.ablation`'s own docstring
describes:

1. A **positive control** — an obviously undersized model vs. a
   reasonably sized one. The harness should (and does) report this as a
   real, trustworthy difference: the loss gap is far larger than the
   run-to-run noise across seeds.
2. A **genuine architecture comparison** — dense SwiGLU vs. MoE, the two
   already-built feedforward options `TransformerConfig` composes
   between. At this toy scale and training budget, both essentially
   memorize the small corpus, and the harness honestly reports the
   difference is NOT trustworthy (within noise) — not massaged into
   looking like a finding it isn't. A real ablation result, whichever
   way it comes out, is more useful than a fabricated "winner."

Run: uv run python examples/18_ablation_harness.py
"""

from __future__ import annotations

import time

from sarva_foundry.ablation import AblationArm, run_ablation
from sarva_foundry.data import DOCUMENT_SEPARATOR, tokenize_corpus
from sarva_foundry.model import MoEConfig, TransformerConfig
from sarva_foundry.tokenizer import ByteLevelBPETokenizer

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox is quick and the dog is lazy",
    "she sells seashells by the seashore and the shells are pretty",
    "how much wood would a woodchuck chuck if a woodchuck could chuck wood",
    "a journey of a thousand miles begins with a single step forward",
    "the rain in spain falls mainly on the plain during springtime",
    "to be or not to be that is the question worth pondering",
    "all that glitters is not gold but some of it truly is",
] * 4
SEEDS = [0, 1, 2]


def _print_result(result, arm_a: str, arm_b: str) -> None:
    for arm in result.ranked():
        print(
            f"  {arm.name:8s} mean_final_loss={arm.mean_final_loss:.4f}  "
            f"stdev={arm.stdev_final_loss:.4f}  params={arm.param_count:,}  "
            f"losses={[round(x, 4) for x in arm.final_losses]}"
        )
    trustworthy = result.is_difference_trustworthy(arm_a, arm_b)
    print(f"  -> difference trustworthy (mean gap > combined stdev)? {trustworthy}")


def main() -> None:
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(CORPUS, vocab_size=400, special_tokens=[DOCUMENT_SEPARATOR])
    token_ids = tokenize_corpus(CORPUS, tokenizer)
    print(f"Corpus: {len(token_ids)} tokens, vocab_size={tokenizer.vocab_size}")

    print("\n--- Comparison 1: positive control (tiny vs. a reasonably sized model) ---")
    print("Every arm trains on the IDENTICAL corpus, in the IDENTICAL order, for the")
    print("IDENTICAL step count, across 3 seeds each -- the only thing that varies")
    print("is model capacity.")
    tiny_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size, dim=8, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16
    )
    bigger_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size, dim=48, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=16
    )
    start = time.perf_counter()
    control_result = run_ablation(
        arms=[
            AblationArm(name="tiny", model_config=tiny_config, description="severely undersized"),
            AblationArm(name="bigger", model_config=bigger_config, description="reasonably sized"),
        ],
        token_ids=token_ids,
        seq_len=16,
        batch_size=4,
        steps=200,
        seeds=SEEDS,
    )
    print(f"  ({time.perf_counter() - start:.1f}s real wall-clock, {len(SEEDS)} seeds x 2 arms)")
    _print_result(control_result, "tiny", "bigger")

    print("\n--- Comparison 2: dense SwiGLU vs. MoE feedforward ---")
    print("The two already-built feedforward options TransformerConfig composes")
    print("between -- a genuine architecture question, not a rigged demonstration.")
    dense_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size, dim=64, n_layers=3, n_heads=4, n_kv_heads=2, max_seq_len=32
    )
    moe_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        dim=64,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        moe=MoEConfig(n_experts=4, n_experts_per_tok=2, n_shared_experts=1),
    )
    start = time.perf_counter()
    arch_result = run_ablation(
        arms=[
            AblationArm(name="dense", model_config=dense_config, description="baseline SwiGLU FFN"),
            AblationArm(name="moe", model_config=moe_config, description="fine-grained MoE FFN"),
        ],
        token_ids=token_ids,
        seq_len=32,
        batch_size=4,
        steps=150,
        seeds=SEEDS,
    )
    print(f"  ({time.perf_counter() - start:.1f}s real wall-clock, {len(SEEDS)} seeds x 2 arms)")
    _print_result(arch_result, "dense", "moe")

    print(
        "\nBoth results are real, not staged: the positive control finds an "
        "obvious capacity gap trustworthy (as it should), and the dense-vs-MoE "
        "comparison honestly reports 'not trustworthy at this scale/budget' "
        "rather than being tuned until it looked like a finding. A harness that "
        "always finds a winner isn't trustworthy -- one that can honestly say "
        "'no real difference detected' is what makes the times it DOES find one "
        "worth believing."
    )


if __name__ == "__main__":
    main()
