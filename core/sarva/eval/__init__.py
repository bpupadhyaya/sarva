"""sarva.eval — benchmark harness (spec §3.6g)."""

from sarva.eval.benchmarks import ARITHMETIC
from sarva.eval.harness import (
    Benchmark,
    BenchmarkCase,
    BenchmarkReport,
    CaseResult,
    contains_match,
    exact_match,
    run_benchmark,
)

__all__ = [
    "ARITHMETIC",
    "Benchmark",
    "BenchmarkCase",
    "BenchmarkReport",
    "CaseResult",
    "contains_match",
    "exact_match",
    "run_benchmark",
]
