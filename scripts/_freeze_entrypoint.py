"""scripts/_freeze_entrypoint.py — the actual script PyInstaller freezes.

Not `.venv/bin/sarva` (the console-script entry point `uv sync` installs
from `core/pyproject.toml`'s `[project.scripts]`): that launcher is a
plain readable Python script with a shebang line on macOS/Linux, but a
*compiled* .exe stub on Windows — PyInstaller can analyze the former but
can't use the latter as a script argument at all. This tiny wrapper is a
real .py file on every platform, so freeze-server.sh points PyInstaller
at this instead, sidestepping the platform difference entirely.
"""

from sarva.cli import app

if __name__ == "__main__":
    app()
