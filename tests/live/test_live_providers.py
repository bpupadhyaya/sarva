"""Live conformance tests for real provider adapters.

Skipped by default (pyproject `addopts = "-m 'not live'"`). Run explicitly
with: `uv run pytest tests/live -m live`. These exercise the SAME contract
the mock provider is held to in tests/conformance/test_provider.py — the
mock isn't a substitute for validating the real adapters, just the thing
that lets the rest of the suite run without credentials or a local runtime.
"""

from __future__ import annotations

import os

import pytest
from sarva.multimodal.content import Message, TextBlock
from sarva.providers.base import DoneEvent, GenerateRequest, StopReason
from sarva.providers.ollama_provider import OllamaProvider


def _has_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _has_google_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _ollama_reachable() -> bool:
    import httpx

    try:
        httpx.get(
            os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/tags",
            timeout=1.0,
        )
        return True
    except httpx.HTTPError:
        return False


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.skipif(not _has_anthropic_key(), reason="ANTHROPIC_API_KEY not set")
async def test_anthropic_terminal_event_law():
    from sarva.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider()
    req = GenerateRequest(
        model="claude-haiku-4-5",
        messages=[Message(role="user", content=[TextBlock(text="Say 'hi' and nothing else.")])],
    )
    events = [e async for e in provider.generate(req)]
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].stop_reason == StopReason.END_TURN
    assert events[-1].usage.output_tokens > 0
    await provider.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.skipif(not _ollama_reachable(), reason="no Ollama server reachable")
async def test_ollama_terminal_event_law():
    provider = OllamaProvider()
    req = GenerateRequest(
        model="ollama/qwen3:8b",
        messages=[Message(role="user", content=[TextBlock(text="Say 'hi' and nothing else.")])],
    )
    events = [e async for e in provider.generate(req)]
    assert isinstance(events[-1], DoneEvent)
    await provider.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.skipif(not _has_openai_key(), reason="OPENAI_API_KEY not set")
async def test_openai_terminal_event_law():
    from sarva.providers.openai_provider import OpenAIProvider

    # No verified-current OpenAI model id is wired into models.yaml yet
    # (see openai_provider.py's module docstring) -- overridable via
    # OPENAI_TEST_MODEL so this test stays runnable without editing code
    # once a real model id is known at run time.
    model = os.environ.get("OPENAI_TEST_MODEL", "gpt-4o-mini")
    provider = OpenAIProvider()
    req = GenerateRequest(
        model=model,
        messages=[Message(role="user", content=[TextBlock(text="Say 'hi' and nothing else.")])],
    )
    events = [e async for e in provider.generate(req)]
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].stop_reason == StopReason.END_TURN
    assert events[-1].usage.output_tokens > 0
    await provider.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.skipif(not _has_google_key(), reason="GEMINI_API_KEY/GOOGLE_API_KEY not set")
async def test_google_terminal_event_law():
    from sarva.providers.google_provider import GoogleProvider

    # No verified-current Gemini model id is wired into models.yaml yet
    # (see google_provider.py's module docstring) -- overridable via
    # GOOGLE_TEST_MODEL so this test stays runnable without editing code
    # once a real model id is known at run time.
    model = os.environ.get("GOOGLE_TEST_MODEL", "gemini-2.0-flash")
    provider = GoogleProvider()
    req = GenerateRequest(
        model=model,
        messages=[Message(role="user", content=[TextBlock(text="Say 'hi' and nothing else.")])],
    )
    events = [e async for e in provider.generate(req)]
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].stop_reason == StopReason.END_TURN
    assert events[-1].usage.output_tokens > 0
    await provider.close()
