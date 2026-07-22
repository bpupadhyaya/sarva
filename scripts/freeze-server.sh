#!/usr/bin/env bash
# scripts/freeze-server.sh — freeze the Python backend into a standalone
# executable (PyInstaller --onefile) for use as a Tauri sidecar, so the
# desktop app can start its own backend with no system Python and no
# terminal — the one-click, no-terminal experience the project promises
# non-developer users.
#
# PyInstaller's import analysis only discovers Python code. It does NOT
# discover non-Python data files the app reads at runtime (the model
# registry YAML, the pre-built web UI), so those are bundled explicitly
# with --add-data. Both core/sarva/runtime.py and core/sarva/server/app.py
# resolve these paths as `Path(__file__).parent / ...`, which is exactly
# the layout --add-data recreates inside the frozen bundle — if either
# module's data-loading path changes, update the --add-data targets here
# to match.
#
# Run ./scripts/build-web.sh first if the web UI source has changed —
# this script freezes whatever is currently in core/sarva/server/static/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$REPO_ROOT/apps/desktop/src-tauri/bin"

# Tauri sidecar binaries must be suffixed with the Rust target triple.
TARGET_TRIPLE="$(rustc -vV 2>/dev/null | sed -n 's/host: //p')"
if [ -z "$TARGET_TRIPLE" ]; then
  echo "error: rustc not found — needed to determine the sidecar target triple" >&2
  exit 1
fi

echo "==> Freezing sarva-server ($TARGET_TRIPLE)"
cd "$REPO_ROOT"
if [ ! -x "$REPO_ROOT/.venv/bin/pyinstaller" ]; then
  echo "error: .venv/bin/pyinstaller not found — run 'uv sync --all-packages --group dev' first" >&2
  exit 1
fi
"$REPO_ROOT/.venv/bin/pyinstaller" --onefile --name sarva-server \
  --distpath "$REPO_ROOT/build/freeze/dist" \
  --workpath "$REPO_ROOT/build/freeze/work" \
  --specpath "$REPO_ROOT/build/freeze" \
  --add-data "$REPO_ROOT/core/sarva/providers/data:sarva/providers/data" \
  --add-data "$REPO_ROOT/core/sarva/server/static:sarva/server/static" \
  --noconfirm \
  .venv/bin/sarva

mkdir -p "$DIST_DIR"
cp "$REPO_ROOT/build/freeze/dist/sarva-server" "$DIST_DIR/sarva-server-$TARGET_TRIPLE"
chmod +x "$DIST_DIR/sarva-server-$TARGET_TRIPLE"

echo "==> Done. $DIST_DIR/sarva-server-$TARGET_TRIPLE is ready to bundle as a Tauri sidecar."
