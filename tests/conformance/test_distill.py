"""Conformance tests for sarva.distill — synthetic-data generation
(spec §3.6c). Runs against MockProvider (no network, no API key) — the
generation/serialization machinery is what's under test here, not any
real frontier model's actual output quality. Real distillation runs
need a configured provider and are the caller's to exercise live, same
split as every other live-only concern in this project.
"""

from __future__ import annotations

import os

import pytest
from sarva.distill import DistillationError, DistillationRecord, distill, load_jsonl, save_jsonl
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
    # any prompt must propagate (wrapped as DistillationError, not
    # swallowed into a record).
    provider = MockProvider(script=[ScriptedTurn(error="rate limited")])
    with pytest.raises(DistillationError):
        await distill(["hello"], provider, model="mock")


async def test_distill_error_carries_every_record_already_generated():
    # A real bug found by actually running a provider that fails on
    # prompt N of a larger batch: before this, the exception propagated
    # straight out of distill() with the records list holding every
    # already-generated (real, potentially expensive) completion simply
    # discarded -- never returned, never persisted. DistillationError
    # now carries those already-succeeded records so a caller isn't
    # forced to throw away real work just because a later prompt in the
    # same batch failed.
    provider = MockProvider(
        script=[
            ScriptedTurn(text="Paris"),
            ScriptedTurn(text="four"),
            ScriptedTurn(error="rate limited"),
        ]
    )

    with pytest.raises(DistillationError) as excinfo:
        await distill(
            ["capital of France?", "2+2?", "this one fails", "never reached"],
            provider,
            model="mock",
        )

    partial = excinfo.value.partial_records
    assert partial == [
        DistillationRecord(prompt="capital of France?", completion="Paris", model="mock"),
        DistillationRecord(prompt="2+2?", completion="four", model="mock"),
    ]
    assert "2/4" in str(excinfo.value)


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


def test_save_jsonl_does_not_destroy_the_previous_file_if_interrupted_mid_write(
    tmp_path, monkeypatch
):
    # A real bug found by actually simulating an interrupted write: this
    # used to open the real output file directly with "w" mode, which
    # truncates it to 0 bytes immediately -- before a single record is
    # written. A crash mid-write destroyed every previously-saved
    # distillation record, real generated data that cost real provider
    # API calls to produce -- confirmed live. Simulated here by making
    # os.replace() raise partway through a second save.
    path = tmp_path / "out.jsonl"
    first = [DistillationRecord(prompt="p1", completion="c1", model="m")]
    save_jsonl(first, path)

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("simulated crash during os.replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)

    second = [DistillationRecord(prompt="p2", completion="c2", model="m")]
    with pytest.raises(OSError):
        save_jsonl(second, path)

    assert load_jsonl(path) == first


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text(
        '{"prompt": "p", "completion": "c", "model": "m"}\n\n'
        '{"prompt": "p2", "completion": "c2", "model": "m"}\n',
        encoding="utf-8",
    )
    loaded = load_jsonl(path)
    assert len(loaded) == 2
