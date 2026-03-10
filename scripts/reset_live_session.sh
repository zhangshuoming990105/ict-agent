#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_ROOT="$ROOT/.live_session"
SCRIPTS="$ROOT/scripts"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reset_live_session.sh

Behavior:
  1. close any running live sessions (graceful first, force-stop on timeout)
  2. remove all .live_session runtime state

This returns the repo to a clean live-session test baseline.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

mkdir -p "$STATE_ROOT"

bash "$SCRIPTS/cleanup_live_session.sh" --stop-running >/dev/null
bash "$SCRIPTS/kill_lingering_live_session_processes.sh" >/dev/null

rm -rf "$STATE_ROOT"/session_*
rm -f "$STATE_ROOT"/stdout.log "$STATE_ROOT"/pid "$STATE_ROOT"/fifo_keeper.pid "$STATE_ROOT"/stdin.fifo
rmdir "$STATE_ROOT" 2>/dev/null || true

echo "[reset_live_session] reset complete."
