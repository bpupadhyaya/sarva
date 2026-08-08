"""sarva.server.schemas — request/response models for the HTTP/WS API."""

from __future__ import annotations

from pydantic import BaseModel

from sarva.agent.budget import Spend


class ChatRequest(BaseModel):
    message: str
    session: str | None = None
    image_base64: str | None = None
    image_media_type: str | None = None
    model: str | None = None
    verify: bool = False


class ChatResponse(BaseModel):
    state: str
    message: str | None
    spend: Spend
    detail: str | None = None


class ModelInfoOut(BaseModel):
    id: str
    display_name: str
    available: bool


class SaveConfigRequest(BaseModel):
    """Only the four provider-key names `sarva.config` knows about are
    accepted — an explicit allowlist (validated in the route handler, not
    just documented here) rather than writing arbitrary caller-supplied
    keys straight into a config file the backend later trusts.

    A real bug found by a fresh-eyes sweep: this docstring has said
    "four" since it was written, but only three fields were ever
    declared here -- `google_api_key` (the exact env-var name Google's
    own SDK docs have historically used, and already a first-class
    entry in `sarva.config.KNOWN_KEYS`/`get_env`'s own Gemini fallback,
    and already correctly reported by `sarva config show`) was missing.
    Because Pydantic ignores unknown fields by default, POSTing
    `google_api_key` didn't error -- it silently vanished with a `200
    OK`, the exact "user thought they sent it, it got silently dropped"
    failure mode this project explicitly guards against elsewhere (e.g.
    `--mcp-header` parsing rejects rather than drops a malformed
    entry). Confirmed live before this fix: `saved config contents: {}`
    after a real `POST /config` with `google_api_key` set."""

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    google_api_key: str | None = None
    # Not a provider key -- `sarva.config.KNOWN_KEYS`/`run_diagnostics`
    # deliberately never lists this one, since it doesn't select which
    # model answers. An optional upgrade for `WebSearchTool` (sarva.agent.
    # tools): unset, the tool already works for free via DuckDuckGo; set,
    # it switches to the paid Brave Search index instead. Accepted here
    # for the same reason the four provider keys are: `save_config` is
    # generic, but this route's own explicit allowlist is what actually
    # decides which caller-supplied fields get persisted.
    brave_api_key: str | None = None


class DoctorCheckOut(BaseModel):
    name: str
    ok: bool
    detail: str
