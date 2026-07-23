"""Conformance tests for sarva.eval — the benchmark harness (spec §3.6g).

Runs against MockProvider (no network, no API key) — the harness itself
is what's under test here, not any real model's actual accuracy. Real
grading against Anthropic/OpenAI/Google/Ollama is exercised by whoever
runs `sarva eval` with a configured API key or a reachable local
server, the same live-only split as everywhere else in this project
(tests/live/). Foundry is the one provider that doesn't belong in that
bucket -- it needs no API key or network, so
`test_foundry_provider.py`'s own `run_benchmark()` test exercises it
for real, right here in this automated suite.
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


def test_contains_match_does_not_false_positive_on_a_longer_wrong_number():
    # A real bug this pins: naive substring matching graded a genuinely
    # WRONG answer ("89") as correct for an expected answer of "9",
    # since "9" is a literal substring of "89". Word-boundary matching
    # fixes it. Found by actually running `sarva eval --model mock` and
    # getting a measured 30% instead of the honest 0% every prior claim
    # in this project had assumed without re-checking the real number.
    case = BenchmarkCase(id="c1", prompt="p", expected="9")
    assert not contains_match("The answer is 89", case)
    assert not contains_match("19 apples", case)
    assert contains_match("The answer is 9.", case)
    assert contains_match("9", case)


def test_arithmetic_benchmark_is_bundled_and_has_real_cases():
    assert ARITHMETIC.name == "arithmetic"
    assert len(ARITHMETIC.cases) == 10
    assert len({c.id for c in ARITHMETIC.cases}) == 10  # every id unique


def test_arithmetic_case_expected_answers_never_appear_in_their_own_prompt():
    # A real, previously-undetected flaw: div-1/div-2 used a perfect
    # square as the dividend with its own square root as the divisor
    # (144 / 12, 81 / 9), so the correct answer was already sitting in
    # the prompt text verbatim -- MockProvider's own prompt echo passed
    # grading without computing anything. This is the structural
    # invariant that flaw violated; pinned directly so no future case
    # can reintroduce it silently.
    import re

    for case in ARITHMETIC.cases:
        pattern = r"\b" + re.escape(case.expected) + r"\b"
        assert not re.search(pattern, case.prompt), (
            f"{case.id}: expected answer {case.expected!r} appears in its own prompt"
        )


async def test_mock_provider_scores_zero_on_the_real_bundled_arithmetic_benchmark():
    # The actual honest-0%-for-mock claim this project has repeated
    # throughout, verified for real at the integration level (not just
    # the grader in isolation) -- this genuinely failed with a measured
    # 30% before contains_match's word-boundary fix and the div-1/div-2
    # case fix above.
    report = await run_benchmark(ARITHMETIC, MockProvider(), model="mock")
    assert report.accuracy == 0.0


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
