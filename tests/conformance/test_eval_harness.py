"""Conformance tests for sarva.eval — the benchmark harness (spec §3.6g).

Runs against MockProvider (no network, no API key) — the harness itself
is what's under test here, not any real model's actual accuracy. Real
provider grading is exercised by whoever runs `sarva eval` with a
configured API key, same split as every other live-only concern in this
project (tests/live/).
"""

from __future__ import annotations

from sarva.eval.benchmarks import ARITHMETIC
from sarva.eval.harness import (
    Benchmark,
    BenchmarkCase,
    contains_match,
    exact_match,
    run_benchmark,
)
from sarva.providers.mock import MockProvider, ScriptedTurn


def test_exact_match_grader():
    case = BenchmarkCase(id="c1", prompt="p", expected="42")
    assert exact_match("42", case)
    assert exact_match(" 42 ", case)  # whitespace-insensitive
    assert exact_match("42", BenchmarkCase(id="c2", prompt="p", expected="42"))
    assert not exact_match("the answer is 42", case)


def test_contains_match_grader():
    case = BenchmarkCase(id="c1", prompt="p", expected="42")
    assert contains_match("the answer is 42.", case)
    assert contains_match("42", case)
    assert not contains_match("the answer is 43", case)


def test_arithmetic_benchmark_is_bundled_and_has_real_cases():
    assert ARITHMETIC.name == "arithmetic"
    assert len(ARITHMETIC.cases) == 10
    assert len({c.id for c in ARITHMETIC.cases}) == 10  # every id unique


async def test_run_benchmark_scores_correct_and_incorrect_cases():
    benchmark = Benchmark(
        name="tiny",
        cases=[
            BenchmarkCase(id="a", prompt="2+2?", expected="4"),
            BenchmarkCase(id="b", prompt="3+3?", expected="6"),
        ],
    )
    provider = MockProvider(script=[ScriptedTurn(text="4"), ScriptedTurn(text="wrong")])

    report = await run_benchmark(benchmark, provider, model="mock")

    assert report.benchmark_name == "tiny"
    assert report.model == "mock"
    assert [r.correct for r in report.results] == [True, False]
    assert report.accuracy == 0.5


async def test_run_benchmark_on_all_correct_gives_full_accuracy():
    benchmark = Benchmark(name="tiny", cases=[BenchmarkCase(id="a", prompt="2+2?", expected="4")])
    provider = MockProvider(script=[ScriptedTurn(text="4")])
    report = await run_benchmark(benchmark, provider, model="mock")
    assert report.accuracy == 1.0


async def test_run_benchmark_records_provider_errors_as_incorrect_not_a_crash():
    benchmark = Benchmark(
        name="tiny",
        cases=[
            BenchmarkCase(id="a", prompt="2+2?", expected="4"),
            BenchmarkCase(id="b", prompt="3+3?", expected="6"),
        ],
    )
    provider = MockProvider(script=[ScriptedTurn(error="rate limited"), ScriptedTurn(text="6")])

    report = await run_benchmark(benchmark, provider, model="mock")

    assert report.results[0].correct is False
    assert "rate limited" in report.results[0].output
    # The failure on case "a" must not prevent case "b" from running.
    assert report.results[1].correct is True


def test_empty_benchmark_report_has_zero_accuracy_not_a_zerodivisionerror():
    from sarva.eval.harness import BenchmarkReport

    report = BenchmarkReport(benchmark_name="empty", model="mock", results=[])
    assert report.accuracy == 0.0


async def test_run_benchmark_accepts_a_custom_grader():
    benchmark = Benchmark(
        name="tiny", cases=[BenchmarkCase(id="a", prompt="2+2?", expected="four")]
    )
    provider = MockProvider(script=[ScriptedTurn(text="4")])

    # Under the default contains_match grader, "4" doesn't satisfy "four".
    default_report = await run_benchmark(benchmark, provider, model="mock")
    assert default_report.accuracy == 0.0

    # A custom grader that treats any non-empty output as correct.
    always_correct = lambda output, case: bool(output)  # noqa: E731
    provider2 = MockProvider(script=[ScriptedTurn(text="4")])
    custom_report = await run_benchmark(benchmark, provider2, model="mock", grader=always_correct)
    assert custom_report.accuracy == 1.0
