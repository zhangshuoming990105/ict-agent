#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
SESSION_ID="${ICT_AGENT_SESSION_ID:-0}"
TIMEOUT="${TIMEOUT:-15}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/close_session.sh [--session-id ID] [--timeout SEC]

Behavior:
  1. send "quit" to the target session
  2. wait for the process to exit cleanly
  3. if still running after timeout, fall back to live_session.sh stop
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-id)
      SESSION_ID="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

status_cmd() {
  bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" status
}

is_running() {
  status_cmd | awk -F= '/^running=/{print $2}' | grep -q '^true$'
}

if ! is_running; then
  echo "[close_session] session_id=${SESSION_ID} already stopped."
  exit 0
fi

bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" send "quit" >/dev/null 2>&1 || true

deadline=$(( $(date +%s) + TIMEOUT ))
while true; do
  if ! is_running; then
    echo "[close_session] session_id=${SESSION_ID} closed cleanly."
    exit 0
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    break
  fi
  sleep 1
done

bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" stop >/dev/null 2>&1 || true

if is_running; then
  echo "[close_session] session_id=${SESSION_ID} failed to stop." >&2
  exit 1
fi

echo "[close_session] session_id=${SESSION_ID} force-stopped after timeout."
