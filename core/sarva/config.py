"""sarva.config — a real, persistent config file for provider API keys.

Closes a gap the desktop app's own promised first-run flow depends on:
the design doc's own T4 definition of done and the README's own
quickstart text both promise a guided first run that offers "paste an
API key" as an alternative to a local model — but until now there was
nowhere for a pasted key to actually go. `sarva.runtime`'s availability
checks, and every provider's SDK client, only ever looked at real
process environment variables; a key entered once in a UI had no way to
survive past that single process's lifetime.

`~/.sarva/config.json` — the same `~/.sarva/` home this project already
uses for session storage (`sarva.memory.session`). A flat dict of
provider env-var names to values (e.g. `{"ANTHROPIC_API_KEY": "sk-..."}`),
deliberately the exact same names `sarva.runtime` already checks via
`os.environ`, so nothing downstream needs a second, parallel notion of
"which key is this."

**Precedence, stated explicitly and tested, not left implicit:** a real
environment variable always wins over a saved config value. A user who
explicitly exported a key in their shell almost certainly means for it
to take effect for that session; silently overriding an explicit env
var with a stale saved file would be a confusing, hard-to-debug
surprise, the same category of "don't guess when you don't have to"
principle this project applies elsewhere (e.g. session-name validation
rejecting rather than silently sanitizing).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".sarva" / "config.json"

# The exact env-var names sarva.runtime checks -- kept here as the one
# place both sides agree on the set of names this module manages.
KNOWN_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")


def load_config(path: Path | None = None) -> dict[str, str]:
    """Returns `{}` if no config file exists yet — a fresh install with
    nothing saved is the expected common case, not an error."""
    path = path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_config(values: dict[str, str], path: Path | None = None) -> None:
    """Merges `values` into whatever's already saved (a caller setting
    only `ANTHROPIC_API_KEY` doesn't wipe out a previously saved
    `OPENAI_API_KEY`), then writes the whole file back."""
    path = path or DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_config(path)
    existing.update(values)
    path.write_text(json.dumps(existing, indent=2))


def get_env(name: str, path: Path | None = None) -> str | None:
    """What `sarva.runtime` should treat env-var `name` as being set to:
    a real process environment variable if set, else whatever's saved in
    the config file, else `None`. Every provider-key check in
    `sarva.runtime` goes through this instead of `os.environ.get`
    directly, so config-file support can't accidentally be forgotten at
    a new call site."""
    env_value = os.environ.get(name)
    if env_value:
        return env_value
    return load_config(path).get(name)
