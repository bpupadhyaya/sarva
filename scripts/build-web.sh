#!/usr/bin/env bash
# scripts/build-web.sh — build the web UI and copy it into the Python
# package so `sarva serve` ships a complete browser experience without
# requiring Node at install time.
#
# Run this after any change under apps/desktop/src/ or apps/desktop/*.json,
# and before committing — core/sarva/server/static/ is checked into git
# deliberately (see BUILD-JOURNAL.md), so a stale build is a real risk this
# script exists specifically to prevent. A real release pipeline should
# call this automatically instead of relying on a human remembering to.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_DIR="$REPO_ROOT/apps/desktop"
STATIC_DIR="$REPO_ROOT/core/sarva/server/static"

echo "==> Building web UI ($WEB_DIR)"
cd "$WEB_DIR"
npm install
npm run build

echo "==> Copying dist/ -> $STATIC_DIR"
rm -rf "${STATIC_DIR:?}"/*
mkdir -p "$STATIC_DIR"
cp -r dist/. "$STATIC_DIR/"

echo "==> Done. core/sarva/server/static/ is up to date — remember to 'git add' it."
