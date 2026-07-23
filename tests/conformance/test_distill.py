"""Conformance tests for sarva.distill — synthetic-data generation
(spec §3.6c). Runs against MockProvider (no network, no API key) — the
generation/serialization machinery is what's under test here, not any
real frontier model's actual output quality. Real distillation runs
need a configured provider and are the caller's to exercise live, same
split as every other live-only concern in this project.
"""

from __future__ import annotations

import pytest
from sarva.distill import DistillationRecord, distill, load_jsonl, save_jsonl
from sarva.providers.base import ProviderError
from sarva.providers.mock import MockProvider, ScriptedTurn


async def test_distill_generates_one_record_per_prompt_in_order():
    provider = MockProvider(script=[ScriptedTurn(text="Paris"), ScriptedTurn(text="four")])
    records = await distill(
        ["what is the capital of France?", "what is 2+2?"], provider, model="mock"
    )

    assert records == [
        DistillationRecord(
            prompt="what is the capital of France?", completion="Paris", model="mock"
        ),
        DistillationRecord(prompt="what is 2+2?", completion="four", model="mock"),
    ]


async def test_distill_records_carry_the_model_id_used():
    provider = MockProvider(script=[ScriptedTurn(text="hi")])
    records = await distill(["hello"], provider, model="claude-opus-4-8")
    assert records[0].model == "claude-opus-4-8"


async def test_distill_propagates_a_provider_error_rather_than_masking_it():
    # Unlike run_benchmark (which scores a failing case as incorrect and
    # continues), distillation output becomes training data -- a silent
    # or garbage record is worse than a loud failure. A ProviderError on
    # any prompt must propagate, not get swallowed into a record.
    provider = MockProvider(script=[ScriptedTurn(error="rate limited")])
    with pytest.raises(ProviderError):
        await distill(["hello"], provider, model="mock")


async def test_distill_passes_the_system_prompt_through():
    # A scripted MockProvider always returns the same text regardless of
    # input, so this checks the request shape reaches the provider
    # without erroring when a system prompt is supplied -- the real
    # per-request translation is covered by each adapter's own tests.
    provider = MockProvider(script=[ScriptedTurn(text="ok")])
    records = await distill(["hello"], provider, model="mock", system="You are terse.")
    assert records[0].completion == "ok"


def test_jsonl_round_trip(tmp_path):
    records = [
        DistillationRecord(prompt="p1", completion="c1", model="m"),
        DistillationRecord(prompt="p2", completion='c2 with "quotes" and\nnewline', model="m"),
    ]
    path = tmp_path / "out.jsonl"
    save_jsonl(records, path)
    loaded = load_jsonl(path)
    assert loaded == records


def test_jsonl_is_genuinely_line_delimited(tmp_path):
    records = [DistillationRecord(prompt="p", completion="c", model="m") for _ in range(3)]
    path = tmp_path / "out.jsonl"
    save_jsonl(records, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text(
        '{"prompt": "p", "completion": "c", "model": "m"}\n\n'
        '{"prompt": "p2", "completion": "c2", "model": "m"}\n',
        encoding="utf-8",
    )
    loaded = load_jsonl(path)
    assert len(loaded) == 2
