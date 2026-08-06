"""sarva.eval.harness — a benchmark harness that grades any
`Provider`-conforming model with the same yardstick (spec §3.6g:
"benchmark harness shared with the registry (grades our models and
third-party models with the same yardstick)").

Deliberately built against the `Provider` protocol, not against any
specific backend: the same `run_benchmark()` call grades Anthropic,
OpenAI, Google, Ollama, the offline Mock provider, and a foundry-trained
model identically — `sarva.providers.foundry_provider.FoundryProvider`
plugs models trained by `sarva_foundry` into this same registry as
first-class citizens, and `tests/conformance/test_foundry_provider.py`
runs a real one through this exact harness. Reuses
`sarva.providers.base.complete()` (the existing "drain the stream, get
the DoneEvent" helper) rather than re-implementing stream draining.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from pydantic import BaseModel

from sarva.multimodal.content import Message, TextBlock
from sarva.providers.base import GenerateRequest, Provider, ProviderError, complete


class BenchmarkCase(BaseModel):
    model_config = {"frozen": True}
    id: str
    prompt: str
    expected: str


class CaseResult(BaseModel):
    model_config = {"frozen": True}
    case_id: str
    output: str
    correct: bool


class BenchmarkReport(BaseModel):
    model_config = {"frozen": True}
    benchmark_name: str
    model: str
    results: list[CaseResult]

    @property
    def accuracy(self) -> float:
        """0.0 for an empty result set rather than raising — an empty
        benchmark is a degenerate but not erroneous case for a caller
        that filters cases dynamically."""
        if not self.results:
            return 0.0
        return sum(r.correct for r in self.results) / len(self.results)


class Benchmark(BaseModel):
    model_config = {"frozen": True}
    name: str
    cases: list[BenchmarkCase]


Grader = Callable[[str, BenchmarkCase], bool]


def exact_match(output: str, case: BenchmarkCase) -> bool:
    return output.strip().lower() == case.expected.strip().lower()


def contains_match(output: str, case: BenchmarkCase) -> bool:
    """The default grader: real models rarely answer with *only* the
    expected string (they explain, they add punctuation/units) — an
    exact-match grader would mostly measure formatting luck rather than
    correctness. Substring matching against the expected answer is the
    same forgiving-but-still-objective yardstick most small benchmark
    harnesses use for short factual/numeric answers.

    Word-boundary matched (`\\b...\\b`), not a raw substring check — a
    real bug found by actually running this against `sarva eval --model
    mock`, not a hypothetical: naive `in` matching graded wrong numeric
    answers as correct purely because the right digits happened to
    appear inside a longer wrong number (e.g. expected `"9"` matched
    inside a wrong `"89"`), and — worse — this project's own bundled
    `ARITHMETIC` cases have prompts whose numbers already contain the
    expected digit as a substring (`"What is 81 / 9?"` contains `"9"`;
    `"...24 apples..."` contains `"4"`), so MockProvider's own prompt
    echo was silently graded "correct" on several cases. This wasn't a
    hypothetical risk: `sarva eval --model mock` measurably reported
    30% accuracy before this fix, not the honest 0% every prior claim
    in this project assumed without re-checking the real number.

    A real, later bug in the same function, sign-blindness: `\\b`
    treats `-` as a non-word character, so a word boundary already
    exists between a minus sign and the digits that follow it -- the
    pattern above matched `"45"` inside a wrong `"-45"` just as readily
    as inside a correct `"45"`. Confirmed live and concretely reachable
    via this project's own bundled `ARITHMETIC` benchmark: `sub-1`
    expects `"45"` for `"92 - 47"`, and a model that reverses operand
    order (a common weak-model mistake -- computing `47 - 92 = -45`
    instead) got full credit for a numerically wrong answer.

    Fixed by replacing `\\b` with explicit lookaround that treats `-`
    as significant rather than relying on word/non-word transitions:
    `(?<![\\w-])` before the pattern rejects a match immediately
    preceded by a digit/letter *or* a minus sign (closing the
    sign-blind gap above), and `(?!\\w)` after it rejects a match
    immediately followed by a digit/letter (preserving the original
    digit-adjacency fix -- `"9"` still can't match inside `"89"`).
    This also fixes a second, related bug the sign-blind one's own
    `\\b`-based pattern had and this project's tests hadn't caught
    yet: a genuinely negative `expected` value (e.g. `"-5"`) never
    matched at all under the old pattern, for the identical reason --
    `\\b` doesn't fire between two non-word characters (a space and a
    leading `-`), so `contains_match` would have scored a correctly
    negative model answer as wrong. Not reachable via any bundled
    `ARITHMETIC` case today (none has a negative `expected`), but a
    real, separate defect in the same boundary logic, fixed by the
    same change rather than left in place alongside the sign-blindness
    fix.

    A third bug in the same boundary logic, found by a much later
    fresh-eyes sweep: decimal-point adjacency. Neither side of the
    boundary excluded `.` -- not `\\w`, so it never blocked a match on
    either side, the same shape as the digit- and sign-adjacency bugs
    above. Confirmed live: `expected="9"` matched inside `"0.9"` (a
    decimal point immediately *before* the match), and `expected="12"`
    matched inside `"12.5"` (a decimal point immediately *after* it) --
    both wrong answers scored `correct=True`. Concretely reachable via
    this project's own bundled `ARITHMETIC` division cases (`div-1:
    84/7=12`, `div-2: 45/5=9`), the ones a real weaker model is most
    likely to answer with a decimal instead of an integer. Fixed by
    adding `.` to the lookbehind's excluded-character set (mirroring
    `-`) and adding a second lookahead, `(?!\\.\\d)`, that only rejects
    a match followed by a decimal point *and then a digit* -- not by a
    bare trailing `.`, which stays allowed so an ordinary
    sentence-ending period (`"The answer is 12."`) still matches
    correctly; only a genuine decimal continuation (`"12.5"`) is now
    rejected.

    A fourth bug, introduced by the third bug's own fix and found by a
    much later fresh-eyes sweep: the decimal-continuation lookahead
    rejected *any* digit right after the decimal point, including an
    all-zero continuation that's numerically the exact same value, not
    a genuinely different one. Confirmed live via this project's own
    bundled `ARITHMETIC` division cases again: a model answering
    `"12.0"` for `div-1` (`84 / 7`, expected `"12"`) or `"9.0"` for
    `div-2` (`45 / 5`, expected `"9"`) -- an entirely ordinary response
    shape, since many models default to float-style output for division
    -- was scored `correct=False` despite being mathematically exactly
    right, silently under-reporting real accuracy with no error or
    warning. The opposite direction from every prior bug in this
    function (those all inflated a wrong answer's score; this one
    deflates a right answer's). Fixed by widening the lookahead so it
    still rejects a decimal point followed by any digit sequence
    containing a genuine nonzero digit (`"12.5"`, `"12.05"`, `"12.10"`
    all still correctly rejected), but no longer rejects one followed
    only by zeros (`"12.0"`, `"12.00"`), since those represent the
    identical value.

    A fifth bug in the same boundary logic, found by a much later
    fresh-eyes sweep: the comma thousands-separator, exactly the same
    shape as the `-`/`.` gaps above, was never excluded either. `,` is
    not `\\w`, so neither side of the boundary blocked it -- confirmed
    live, both directions: `expected="200"` matched inside a wrong
    `"1,200"` (the comma right before the match wasn't excluded by the
    lookbehind, the same way `-` wasn't before its own fix), and
    `expected="12"` matched inside a wrong `"12,000"` (the comma right
    after wasn't excluded by the lookahead). Comma-grouped formatting
    for any answer >= 1000 is completely ordinary model output, not a
    contrived case -- and unlike the fourth bug's all-zero decimal
    continuation, a comma-then-digits continuation is *never* the same
    value as the bare match (`"12,000"` is really twelve thousand, not
    twelve), so there's no all-zero-style carve-out needed here: this
    lookahead rejects unconditionally whenever a digit follows the
    comma. Fixed by adding `,` to the lookbehind's excluded-character
    set (mirroring `-`/`.`) and a third lookahead, `(?!,\\d)`, that
    rejects a match immediately followed by a comma and then a digit --
    an ordinary trailing comma before a space or word (`"12, not 13"`)
    stays allowed, only a genuine thousands-continuation is rejected.

    A sixth bug, the mirror image of the fourth (decimal all-zero) one,
    found by a much later fresh-eyes sweep: the fifth bug's own fix
    closed the false-positive direction (a comma-grouped WRONG answer,
    e.g. `"12,000"` matching a shorter `expected="12"`) but introduced
    no corresponding handling for the false-negative direction -- a
    comma-grouped CORRECT answer never matches `expected` at all, since
    the literal substring search has no way to see "1200" inside the
    text "1,200"; the comma sits right in the middle of what would
    otherwise be an exact match. Confirmed live: `expected="1200"`
    against the entirely ordinary model output `"1,200"` (thousands-
    grouped formatting for any answer >= 1000 is completely standard,
    not a contrived shape -- the same real-world formatting habit the
    fifth bug's own fix was written to reject when the VALUE differs)
    scored `correct=False` for a numerically exact answer, silently
    under-reporting real accuracy with no error or warning -- the same
    "deflates a right answer" failure shape as the fourth bug, just for
    thousands-commas instead of trailing decimal zeros. Fixed by
    stripping genuine thousands-separator commas (a comma between two
    digits, followed by exactly three digits with no fourth digit right
    after) out of the text being searched, but only when `expected`
    itself is a plain (optionally negative) integer -- a non-numeric
    expected answer (e.g. a yes/no or word-based benchmark case) is left
    untouched, so this can't introduce a spurious match where none of
    the digit-boundary logic above is even in play. This normalization
    happens before the existing boundary pattern runs, so every prior
    fix's own digit/sign/decimal/comma-adjacency guard still applies
    identically to the now comma-free text -- a comma-grouped WRONG
    answer (the fifth bug's own case) still correctly fails to match,
    since stripping its separator doesn't change the underlying value
    being searched for."""
    expected = case.expected.strip()
    text = output.strip()
    if re.fullmatch(r"-?\d+", expected):
        text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    pattern = r"(?<![\w.,-])" + re.escape(expected) + r"(?!\w)(?!\.\d*[1-9])(?!,\d)"
    return re.search(pattern, text, re.IGNORECASE) is not None


async def run_benchmark(
    benchmark: Benchmark,
    provider: Provider,
    model: str,
    grader: Grader = contains_match,
) -> BenchmarkReport:
    """Run every case in `benchmark` as an independent, single-turn
    request against `provider`/`model`. A case whose request fails
    (rate limit, auth, any `ProviderError`) is scored incorrect with the
    error recorded as its output, rather than aborting the whole run —
    one bad case shouldn't hide every other case's real result."""
    results: list[CaseResult] = []
    for case in benchmark.cases:
        request = GenerateRequest(
            model=model,
            messages=[Message(role="user", content=[TextBlock(text=case.prompt)])],
        )
        try:
            done = await complete(provider, request)
            output = done.message.text()
        except ProviderError as e:
            results.append(CaseResult(case_id=case.id, output=f"[error: {e}]", correct=False))
            continue
        results.append(CaseResult(case_id=case.id, output=output, correct=grader(output, case)))
    return BenchmarkReport(benchmark_name=benchmark.name, model=model, results=results)
