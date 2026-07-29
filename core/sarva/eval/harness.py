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
    fix."""
    pattern = r"(?<![\w-])" + re.escape(case.expected.strip()) + r"(?!\w)"
    return re.search(pattern, output.strip(), re.IGNORECASE) is not None


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
