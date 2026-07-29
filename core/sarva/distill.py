"""sarva.distill — synthetic-data generation: frontier-as-teacher via
the provider layer (spec §3.6c: "synthetic-data generation
(frontier-as-teacher via the provider layer)").

Built against the `Provider` protocol, not any specific backend — the
same abstraction `sarva.eval.harness.run_benchmark` already uses to
grade every model with the same yardstick, reused here to *generate*
data with any model instead. Reuses `sarva.providers.base.complete()`
(the "drain the stream, get the DoneEvent" helper) rather than
reimplementing stream draining, same pattern as the eval harness.

Deliberately produces a plain, package-agnostic format (JSON records
with `prompt`/`completion`/`model` fields), not `sarva_foundry.train.
sft.SFTExample` objects directly: `core` has no dependency on
`sarva_foundry`, and `sarva_foundry` has no dependency on `core` (their
`pyproject.toml`s name completely disjoint dependency sets — `torch`/
`numpy` vs. `anthropic`/`openai`/`google-genai`/etc.) — a caller who
wants to turn this output into foundry SFT training data reads the
JSONL and constructs `SFTExample(prompt=r.prompt, response=r.completion)`
directly, one line of glue code in the caller, not a package-level
import in either direction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sarva.atomic_write import atomic_write_text
from sarva.multimodal.content import Message, TextBlock
from sarva.providers.base import GenerateRequest, Provider, ProviderError, complete


@dataclass(frozen=True)
class DistillationRecord:
    prompt: str
    completion: str
    model: str


class DistillationError(Exception):
    """Raised when generation fails partway through a batch. Carries
    every record already generated (`partial_records`) so a caller can
    save the real, already-paid-for completions instead of losing them
    -- a real bug found by actually running a provider that fails on
    prompt N of a larger batch: before this, the exception propagated
    straight out of `distill()` with the `records` list holding every
    prior success simply discarded, since nothing was returned or
    persisted until the whole loop finished. The same "don't throw away
    expensive work already done" concern `cli.py`'s own `_distill()`
    already names for the output-file write failing -- this closes the
    identical gap on the generation side instead."""

    def __init__(self, message: str, partial_records: list[DistillationRecord]) -> None:
        super().__init__(message)
        self.partial_records = partial_records


async def distill(
    prompts: list[str],
    provider: Provider,
    model: str,
    system: str | None = None,
) -> list[DistillationRecord]:
    """Generate one completion per prompt from `provider`/`model` — the
    frontier-as-teacher step. Unlike `run_benchmark` (which scores a
    failing case as incorrect and keeps going, since one bad benchmark
    case shouldn't hide every other case's real result), a `ProviderError`
    here still propagates rather than being silently skipped:
    distillation output becomes training data, so a silently-missing or
    garbage record is a worse outcome than a loud failure a caller can
    retry or investigate. It propagates as `DistillationError`, though,
    not the raw `ProviderError` -- carrying every completion already
    generated so a caller isn't forced to throw away real, potentially
    expensive API calls just because a later one in the same batch
    failed."""
    records: list[DistillationRecord] = []
    for prompt in prompts:
        request = GenerateRequest(
            model=model,
            system=system,
            messages=[Message(role="user", content=[TextBlock(text=prompt)])],
        )
        try:
            done = await complete(provider, request)
        except ProviderError as e:
            raise DistillationError(
                f"generation failed after {len(records)}/{len(prompts)} prompts: {e}",
                partial_records=records,
            ) from e
        records.append(
            DistillationRecord(prompt=prompt, completion=done.message.text(), model=model)
        )
    return records


def save_jsonl(records: list[DistillationRecord], path: Path) -> None:
    """Atomic write, not a direct `path.open("w")`: confirmed live that
    `"w"` mode truncates the file to 0 bytes the instant it's opened,
    before a single record is written -- a crash between that open and
    the write completing destroys every previously-saved distillation
    record, real generated data that cost real provider API calls to
    produce (see `DistillationError`'s own docstring on not throwing
    away expensive completions). Same shared fix as `sarva.config`/
    `sarva.memory.session`/`WriteFileTool` -- see `sarva.atomic_write`."""
    content = "".join(json.dumps(asdict(record)) + "\n" for record in records)
    atomic_write_text(path, content)


def load_jsonl(path: Path) -> list[DistillationRecord]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(DistillationRecord(**json.loads(line)))
    return records
