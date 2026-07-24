"""sarva.providers.registry — the model registry and router.

The registry is data (`models.yaml`); the router is a small policy that
picks a model per task class from that data. This is the mechanism that
lets Sarva absorb a new frontier model as a one-entry registry change
instead of a code change anywhere else in the system.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml

from sarva.multimodal.content import Modality
from sarva.providers.base import ModelInfo


class TaskClass(StrEnum):
    MAIN = "main"  # the primary agent loop
    SUBTASK = "subtask"  # cheap delegated work
    ESCALATION = "escalation"  # hardest problems (two-strikes)
    VISION = "vision"
    AUDIO = "audio"


class UnknownModelError(Exception):
    """Raised when an explicit model override doesn't name a registered
    model. Deliberately NOT a LookupError subclass (unlike the "no
    available model for this task" case `pick()` raises below) -- a
    caller who named a specific model and got it wrong must see a clear
    error, not have `AgentLoop`'s modality-degradation fallback silently
    catch it and substitute a different model for the one they actually
    asked for. See `AgentLoop.run()`'s own handling of this."""


class Registry:
    def __init__(self, models: dict[str, ModelInfo]):
        self._models = models

    @classmethod
    def load(cls, path: Path) -> Registry:
        raw = yaml.safe_load(path.read_text())
        models = {m["id"]: ModelInfo.model_validate(m) for m in raw["models"]}
        return cls(models)

    def get(self, model_id: str) -> ModelInfo:
        return self._models[model_id]

    def all(self) -> list[ModelInfo]:
        return list(self._models.values())

    def register(self, model: ModelInfo) -> None:
        """Adds (or replaces) one entry at runtime -- used for models
        discovered dynamically rather than declared in `models.yaml`, e.g.
        a user's own locally-trained foundry checkpoints (see
        `sarva.runtime`), which can't have a fixed static entry since the
        set of checkpoints varies per install."""
        self._models[model.id] = model


def load_routing(path: Path) -> dict[TaskClass, list[str]]:
    raw = yaml.safe_load(path.read_text())
    return {TaskClass(k): v for k, v in raw["routing"].items()}


class Router:
    """Policy: data-driven default candidates per TaskClass with a
    modality-aware, availability-aware fallback. `pick()` returns the first
    candidate that (a) exists in the registry, (b) supports the modalities
    the caller needs, and (c) is available (API key present / local runtime
    up / etc., as tracked by the caller in `available`)."""

    def __init__(
        self,
        registry: Registry,
        routing: dict[TaskClass, list[str]],
        available: set[str],
    ):
        self.registry = registry
        self.routing = routing
        self.available = available

    def pick(
        self,
        task: TaskClass,
        needs: set[Modality] = frozenset({Modality.TEXT}),
        override: str | None = None,
    ) -> ModelInfo:
        if override:
            try:
                return self.registry.get(override)
            except KeyError:
                raise UnknownModelError(
                    f"unknown model {override!r} -- see 'sarva models' for the full list"
                ) from None
        for mid in self.routing.get(task, []):
            if mid not in self.available:
                continue
            m = self.registry.get(mid)
            if needs <= m.capabilities.modalities_in:
                return m
        raise LookupError(f"no available model for {task} needing {needs}")
