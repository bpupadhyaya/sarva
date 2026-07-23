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

from sarva.multimodal.content import Message, TextBlock
from sarva.providers.base import GenerateRequest, Provider, complete


@dataclass(frozen=True)
class DistillationRecord:
    prompt: str
    completion: str
    model: str


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
    here propagates: distillation output becomes training data, so a
    silently-missing or garbage record is a worse outcome than a loud
    failure a caller can retry or investigate."""
    records: list[DistillationRecord] = []
    for prompt in prompts:
        request = GenerateRequest(
            model=model,
            system=system,
            messages=[Message(role="user", content=[TextBlock(text=prompt)])],
        )
        done = await complete(provider, request)
        records.append(
            DistillationRecord(prompt=prompt, completion=done.message.text(), model=model)
        )
    return records


def save_jsonl(records: list[DistillationRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record)) + "\n")


def load_jsonl(path: Path) -> list[DistillationRecord]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(DistillationRecord(**json.loads(line)))
    return records
