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
from sarva.multimodal.content import Message, TextBlock
from sarva.providers.base import DoneEvent, StopReason, Usage
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


def test_contains_match_is_not_sign_blind():
    # A real bug found by actually running this against a model that
    # reverses subtraction operand order (a common weak-model mistake):
    # "9" is a non-word character to \b, so a word boundary already
    # exists between a minus sign and the digits that follow it -- the
    # old pattern matched "45" inside a wrong "-45" just as readily as
    # inside a correct "45". Concretely reachable via this project's own
    # bundled ARITHMETIC benchmark: sub-1 expects "45" for "92 - 47", and
    # computing 47 - 92 = -45 used to score full credit for a
    # numerically wrong answer.
    case = BenchmarkCase(id="sub-1", prompt="p", expected="45")
    assert not contains_match("The answer is -45", case)
    assert not contains_match("-45", case)
    assert contains_match("The answer is 45", case)  # the real fix must not break this


def test_contains_match_still_recognizes_a_genuinely_negative_expected_answer():
    # A second, related bug found while verifying the sign-blindness fix
    # above: \b never fires between two non-word characters (a space
    # and a leading "-"), so a genuinely negative `expected` value never
    # matched at all under the old \b-based pattern -- not reachable via
    # any bundled ARITHMETIC case today (none has a negative expected
    # answer), but a real, separate defect in the same boundary logic.
    case = BenchmarkCase(id="neg-1", prompt="p", expected="-5")
    assert contains_match("The answer is -5", case)
    assert contains_match("-5", case)
    assert not contains_match("The answer is -55", case)  # still rejects a longer wrong number


def test_contains_match_is_not_decimal_blind():
    # A third bug in the same boundary logic, found by a much later
    # fresh-eyes sweep: neither side of the word-boundary pattern
    # excluded "." -- not \w, so it never blocked a match on either
    # side, the same shape as the digit- and sign-adjacency bugs above.
    # Confirmed live: expected="9" matched inside "0.9" (a decimal point
    # immediately before the match), and expected="12" matched inside
    # "12.5" (a decimal point immediately after it) -- both wrong
    # answers scored correct=True. Concretely reachable via this
    # project's own bundled ARITHMETIC division cases (div-1: 84/7=12,
    # div-2: 45/5=9), the ones a real weaker model is most likely to
    # answer with a decimal instead of an integer.
    case_9 = BenchmarkCase(id="div-2", prompt="p", expected="9")
    assert not contains_match("0.9", case_9)
    assert not contains_match("The answer is 0.9", case_9)
    assert contains_match("9", case_9)  # the real fix must not break this

    case_12 = BenchmarkCase(id="div-1", prompt="p", expected="12")
    assert not contains_match("12.5", case_12)
    assert contains_match("12", case_12)
    # A bare trailing period (ordinary sentence punctuation, not a
    # decimal continuation) must still match -- only a period followed
    # by another digit is a genuine decimal, not just any period.
    assert contains_match("The answer is 12.", case_12)

    # A genuinely decimal expected answer must still match itself exactly.
    case_decimal = BenchmarkCase(id="dec-1", prompt="p", expected="0.9")
    assert contains_match("0.9", case_decimal)
    assert contains_match("The answer is 0.9.", case_decimal)


def test_contains_match_does_not_reject_an_integer_answer_formatted_as_a_float():
    # A fourth bug, introduced by the decimal-blindness fix above and
    # found by a much later fresh-eyes sweep: the fix's own lookahead
    # rejected ANY digit right after the decimal point, including an
    # all-zero continuation that's numerically the exact same value,
    # not a genuinely different one. Confirmed live via this project's
    # own bundled ARITHMETIC division cases again: a model answering
    # "12.0" for div-1 (84 / 7, expected "12") or "9.0" for div-2
    # (45 / 5, expected "9") -- an entirely ordinary response shape,
    # since many models default to float-style output for division --
    # was scored correct=False despite being mathematically exactly
    # right, silently under-reporting real accuracy with no error or
    # warning. The opposite direction from every prior bug pinned above
    # (those all inflated a wrong answer's score; this one deflates a
    # right answer's).
    case_9 = BenchmarkCase(id="div-2", prompt="p", expected="9")
    assert contains_match("9.0", case_9)
    assert contains_match("The answer is 9.00", case_9)

    case_12 = BenchmarkCase(id="div-1", prompt="p", expected="12")
    assert contains_match("12.0", case_12)
    assert contains_match("The answer is 12.0", case_12)
    # A genuinely different decimal value must still be rejected -- the
    # widened lookahead must not accidentally accept any decimal at all.
    assert not contains_match("12.5", case_12)
    assert not contains_match("12.05", case_12)
    assert not contains_match("12.10", case_12)


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


class _EmptyStreamProvider:
    """A minimal, structurally-valid third-party Provider (this module's
    own docstring names grading third-party models a first-class use
    case) whose generate() ends without ever yielding a DoneEvent or
    StreamErrorEvent on its 2nd call -- exactly what an ordinary network
    drop mid-stream does to a not-as-defensively-written adapter, not a
    contrived shape."""

    name = "flaky"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request):
        self.calls += 1
        if self.calls == 2:
            return
        yield DoneEvent(
            stop_reason=StopReason.END_TURN,
            message=Message(role="assistant", content=[TextBlock(text="4")]),
            usage=Usage(),
        )

    async def close(self) -> None:
        return None


async def test_run_benchmark_records_a_stream_that_ends_without_a_doneevent_as_incorrect():
    # A real bug found by giving sarva.providers.base.complete()'s own
    # callers a fresh-eyes sweep: a provider generator ending without
    # ever yielding a DoneEvent/StreamErrorEvent raised a bare
    # RuntimeError, not a ProviderError -- run_benchmark only ever
    # catches ProviderError around each case, so this crashed the WHOLE
    # benchmark run and discarded every already-graded case's result,
    # directly contradicting this module's own "one bad case shouldn't
    # hide every other case's real result" contract (see the
    # rate-limited-provider test above, which this mirrors for a
    # different failure shape).
    benchmark = Benchmark(
        name="tiny",
        cases=[
            BenchmarkCase(id="a", prompt="2+2?", expected="4"),
            BenchmarkCase(id="b", prompt="3+3?", expected="4"),
        ],
    )

    report = await run_benchmark(benchmark, _EmptyStreamProvider(), model="flaky")

    assert report.results[0].correct is True
    assert report.results[1].correct is False
    assert "DoneEvent" in report.results[1].output


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
