"""sarva.eval.benchmarks — small, real, hand-verified benchmark sets
bundled with the harness. Deliberately not a claim to GSM8K-scale
coverage or difficulty — the same "crudest real thing, not a fabricated
placeholder" honesty this project applies everywhere (the corpus
pipeline's length filter, the degraders' metadata-only reports): ten
genuinely correct arithmetic problems, each answer computed and checked
by hand, not generated and assumed right.
"""

from __future__ import annotations

from sarva.eval.harness import Benchmark, BenchmarkCase

ARITHMETIC = Benchmark(
    name="arithmetic",
    cases=[
        BenchmarkCase(
            id="add-1", prompt="What is 17 + 26? Answer with just the number.", expected="43"
        ),
        BenchmarkCase(
            id="add-2", prompt="What is 148 + 275? Answer with just the number.", expected="423"
        ),
        BenchmarkCase(
            id="sub-1", prompt="What is 92 - 47? Answer with just the number.", expected="45"
        ),
        BenchmarkCase(
            id="sub-2", prompt="What is 500 - 233? Answer with just the number.", expected="267"
        ),
        BenchmarkCase(
            id="mul-1", prompt="What is 13 * 12? Answer with just the number.", expected="156"
        ),
        BenchmarkCase(
            id="mul-2", prompt="What is 9 * 8? Answer with just the number.", expected="72"
        ),
        BenchmarkCase(
            id="div-1", prompt="What is 144 / 12? Answer with just the number.", expected="12"
        ),
        BenchmarkCase(
            id="div-2", prompt="What is 81 / 9? Answer with just the number.", expected="9"
        ),
        BenchmarkCase(
            id="word-1",
            prompt=(
                "A train travels 60 miles per hour for 3 hours. "
                "How many miles does it travel in total? Answer with just the number."
            ),
            expected="180",
        ),
        BenchmarkCase(
            id="word-2",
            prompt=(
                "A basket has 24 apples. If you split them equally among 6 people, "
                "how many apples does each person get? Answer with just the number."
            ),
            expected="4",
        ),
    ],
)
