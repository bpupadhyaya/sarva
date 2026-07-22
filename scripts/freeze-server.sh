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

# On Windows, this script runs under Git Bash (MSYS2), whose automatic
# POSIX<->Windows path conversion is inconsistent in a way two separate
# confirmed CI failures exposed: left enabled, it mangles --add-data's
# semicolon-joined SRC;DEST value (converts D:/a/sarva/sarva/core/... into
# \\d\\a\\sarva\\sarva\\core\\...); disabled outright (MSYS_NO_PATHCONV=1),
# plain single-path arguments like the script path stop being converted
# at all, so PyInstaller — a native Windows program with no idea what
# MSYS's internal /d/a/... paths mean — reports them as not existing.
# Rather than fight that heuristic either way, disable it and resolve
# every path PyInstaller receives to native Windows form ourselves via
# `cygpath`, which only exists under Git Bash/MSYS in the first place —
# harmless on macOS/Linux where $NATIVE_ROOT just equals $REPO_ROOT.
export MSYS_NO_PATHCONV=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$REPO_ROOT/apps/desktop/src-tauri/bin"
if command -v cygpath >/dev/null 2>&1; then
  NATIVE_ROOT="$(cygpath -m "$REPO_ROOT")"
else
  NATIVE_ROOT="$REPO_ROOT"
fi

# Tauri sidecar binaries must be suffixed with the Rust target triple.
TARGET_TRIPLE="$(rustc -vV 2>/dev/null | sed -n 's/host: //p')"
if [ -z "$TARGET_TRIPLE" ]; then
  echo "error: rustc not found — needed to determine the sidecar target triple" >&2
  exit 1
fi

# uv venvs use .venv/bin on macOS/Linux and .venv/Scripts on Windows, and
# every executable in it — including the frozen binary PyInstaller itself
# produces — gains a .exe suffix on Windows. PyInstaller's --add-data
# separator is also platform-dependent (os.pathsep: ':' on POSIX, ';' on
# Windows) — get any of these wrong and the freeze either can't find
# pyinstaller at all, or silently mis-parses the --add-data paths.
EXE_SUFFIX=""
ADD_DATA_SEP=":"
if [ -x "$REPO_ROOT/.venv/Scripts/pyinstaller.exe" ]; then
  VENV_BIN="$REPO_ROOT/.venv/Scripts"
  EXE_SUFFIX=".exe"
  ADD_DATA_SEP=";"
elif [ -x "$REPO_ROOT/.venv/bin/pyinstaller" ]; then
  VENV_BIN="$REPO_ROOT/.venv/bin"
else
  echo "error: pyinstaller not found in .venv — run 'uv sync --all-packages --group dev' first" >&2
  exit 1
fi

echo "==> Freezing sarva-server ($TARGET_TRIPLE)"
cd "$REPO_ROOT"
# Freezes scripts/_freeze_entrypoint.py, not the installed `sarva`
# console-script launcher: that launcher is a plain readable .py file on
# macOS/Linux but a *compiled* .exe stub on Windows (confirmed via a real
# Windows CI failure: "Script file '...\\sarva.exe' does not exist" —
# PyInstaller can't treat a compiled binary as an analyzable script at
# all). The wrapper is a real .py file on every platform, sidestepping
# the difference entirely.
"$VENV_BIN/pyinstaller$EXE_SUFFIX" --onefile --name sarva-server \
  --distpath "$NATIVE_ROOT/build/freeze/dist" \
  --workpath "$NATIVE_ROOT/build/freeze/work" \
  --specpath "$NATIVE_ROOT/build/freeze" \
  --add-data "$NATIVE_ROOT/core/sarva/providers/data${ADD_DATA_SEP}sarva/providers/data" \
  --add-data "$NATIVE_ROOT/core/sarva/server/static${ADD_DATA_SEP}sarva/server/static" \
  --noconfirm \
  "$NATIVE_ROOT/scripts/_freeze_entrypoint.py"

mkdir -p "$DIST_DIR"
cp "$REPO_ROOT/build/freeze/dist/sarva-server$EXE_SUFFIX" "$DIST_DIR/sarva-server-$TARGET_TRIPLE$EXE_SUFFIX"
chmod +x "$DIST_DIR/sarva-server-$TARGET_TRIPLE$EXE_SUFFIX"

echo "==> Done. $DIST_DIR/sarva-server-$TARGET_TRIPLE$EXE_SUFFIX is ready to bundle as a Tauri sidecar."
