#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_ID="${SESSION_ID:-0}"
LOG=""
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"
MESSAGES=(
  "现在几点？"
  "你是谁？一句话回答。"
  "1加1等于几？"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-id)
      SESSION_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

LOG="$ROOT/.live_session/session_${SESSION_ID}/stdout.log"

cd "$ROOT"

cleanup() {
  bash scripts/live_session.sh --session-id "$SESSION_ID" stop >/dev/null 2>&1 || true
}

ready_count() {
  local n
  n=$(grep -c '>>> Ready for input.' "$LOG" 2>/dev/null) || n=0
  echo "$n"
}

wait_ready() {
  local expected=$1
  local deadline=$(( $(date +%s) + WAIT_TIMEOUT ))
  while true; do
    if [ "$(ready_count)" -ge "$expected" ]; then
      return 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "[basic_live_session] timeout waiting for ready signal >= $expected" >&2
      return 1
    fi
    sleep 2
  done
}

send_and_wait() {
  local msg="$1"
  local before
  before=$(ready_count)
  bash scripts/live_session.sh --session-id "$SESSION_ID" send "$msg" >/dev/null
  wait_ready $((before + 1))
}

trap cleanup EXIT

bash scripts/live_session.sh --session-id "$SESSION_ID" stop >/dev/null 2>&1 || true

bash scripts/live_session.sh --session-id "$SESSION_ID" start
wait_ready 1

for msg in "${MESSAGES[@]}"; do
  send_and_wait "$msg"
done

bash scripts/live_session.sh --session-id "$SESSION_ID" send "quit" >/dev/null || true
sleep 1
bash scripts/live_session.sh --session-id "$SESSION_ID" stop >/dev/null 2>&1 || true

echo "[basic_live_session] passed (${#MESSAGES[@]} messages, session_id=${SESSION_ID}). log: $LOG"
