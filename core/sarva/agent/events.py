"""sarva.agent.events — the agent loop's state machine and event vocabulary.

Every skin (CLI, server, desktop) consumes only `AgentEvent`s — none of them
reach into loop internals. The state machine is explicit data (`LEGAL`) so
the loop's control flow stays inspectable rather than implicit in code paths.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from sarva.agent.budget import Spend
from sarva.multimodal.content import Message, ToolCallBlock, ToolResultBlock
from sarva.providers.base import ProviderEvent


class AgentState(StrEnum):
    INIT = "init"
    CALLING_MODEL = "calling_model"
    RUNNING_TOOLS = "running_tools"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    DONE = "done"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    BUDGET_EXCEEDED = "budget_exceeded"


TERMINAL = {
    AgentState.DONE,
    AgentState.FAILED,
    AgentState.INTERRUPTED,
    AgentState.BUDGET_EXCEEDED,
}

LEGAL: dict[AgentState, set[AgentState]] = {
    AgentState.INIT: {AgentState.CALLING_MODEL},
    AgentState.CALLING_MODEL: {
        AgentState.RUNNING_TOOLS,
        AgentState.DONE,
        AgentState.FAILED,
        AgentState.INTERRUPTED,
        AgentState.BUDGET_EXCEEDED,
    },
    AgentState.RUNNING_TOOLS: {
        AgentState.AWAITING_CONFIRMATION,
        AgentState.CALLING_MODEL,
        AgentState.FAILED,
        AgentState.INTERRUPTED,
        AgentState.BUDGET_EXCEEDED,
    },
    AgentState.AWAITING_CONFIRMATION: {
        AgentState.RUNNING_TOOLS,
        AgentState.INTERRUPTED,
    },
}


class _AEvent(BaseModel):
    model_config = {"frozen": True}
    ts: float = Field(default_factory=time.time)


class StateChangedEvent(_AEvent):
    type: Literal["state_changed"] = "state_changed"
    state: AgentState
    detail: str | None = None


class ModelStreamEvent(_AEvent):
    """Provider deltas passed through for live rendering."""

    type: Literal["model_stream"] = "model_stream"
    event: ProviderEvent


class ToolStartedEvent(_AEvent):
    type: Literal["tool_started"] = "tool_started"
    call: ToolCallBlock


class ToolFinishedEvent(_AEvent):
    type: Literal["tool_finished"] = "tool_finished"
    result: ToolResultBlock
    seconds: float


class NeedsConfirmationEvent(_AEvent):
    type: Literal["needs_confirmation"] = "needs_confirmation"
    call: ToolCallBlock


class RunDoneEvent(_AEvent):
    type: Literal["run_done"] = "run_done"
    state: AgentState
    final_message: Message | None
    spend: Spend


AgentEvent = Annotated[
    (
        StateChangedEvent
        | ModelStreamEvent
        | ToolStartedEvent
        | ToolFinishedEvent
        | NeedsConfirmationEvent
        | RunDoneEvent
    ),
    Field(discriminator="type"),
]


class AgentResult(BaseModel):
    model_config = {"frozen": True}
    state: AgentState
    final_message: Message | None
    spend: Spend
    run_dir: str
