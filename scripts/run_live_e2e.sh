#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_ID="${SESSION_ID:-0}"
LOG=""
SCRIPTS="$ROOT/scripts"
MODEL="gpt-oss-120b"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-id) SESSION_ID="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

LOG="$ROOT/.live_session/session_${SESSION_ID}/stdout.log"

fail() { echo "[FAIL] $*" >&2; FAILURES=$((FAILURES + 1)); }

WAIT_TIMEOUT="${WAIT_TIMEOUT:-120}"
FAILURES=0

cleanup() {
  # Kill agent by PID first so stop doesn't block on fifo write (agent may not be reading on exit).
  local pid_file="$ROOT/.live_session/session_${SESSION_ID}/pid"
  local wrapper_file="$ROOT/.live_session/session_${SESSION_ID}/wrapper.pid"
  if [[ -f "$pid_file" ]]; then
    local pid; pid=$(cat "$pid_file" 2>/dev/null)
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  fi
  if [[ -f "$wrapper_file" ]]; then
    local wpid; wpid=$(cat "$wrapper_file" 2>/dev/null)
    [[ -n "$wpid" ]] && kill "$wpid" 2>/dev/null || true
  fi
  sleep 1
  timeout 5 bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" stop >/dev/null 2>&1 || true
}

wait_ready() {
  local expected=$1
  local timeout="${2:-$WAIT_TIMEOUT}"
  local deadline=$(( $(date +%s) + timeout ))
  local count
  while true; do
    count=$(grep -c '>>> Ready for input.' "$LOG" 2>/dev/null) || count=0
    [ "$count" -ge "$expected" ] && return 0
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "[run_live_e2e] timeout waiting for ready count>=$expected (got $count)" >&2
      return 1
    fi
    sleep 2
  done
}

trap cleanup EXIT

bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" stop >/dev/null 2>&1 || true
bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" start --model "$MODEL" --compact-model "$MODEL"
wait_ready 1

bash "$SCRIPTS/run_20_turns.sh" --session-id "$SESSION_ID"

wait_ready 23 300

READY_COUNT=$(grep -c '>>> Ready for input.' "$LOG" 2>/dev/null) || READY_COUNT=0
if [ "$READY_COUNT" -lt 23 ]; then
  fail "Ready signals: $READY_COUNT < 23"
fi

for tool in write_file read_file run_shell; do
  if ! grep -q "Calling tool: $tool" "$LOG" 2>/dev/null; then
    fail "Tool NOT called: $tool"
  fi
done

if ! grep -q "Compacting .* old messages" "$LOG" 2>/dev/null; then
  fail "Compact was never attempted"
fi

if ! grep -q "Compressed .* old messages" "$LOG" 2>/dev/null && \
   ! grep -q "rate.limit\|429\|retrying" "$LOG" 2>/dev/null; then
  fail "Compact failed for unexpected reason"
fi

if ! grep -q '"role":' "$LOG" 2>/dev/null; then
  fail "/debug raw produced no JSON output"
fi

bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" send "quit" >/dev/null || true

if [ "$FAILURES" -eq 0 ]; then
  echo "[run_live_e2e] passed (session_id=${SESSION_ID}). log: $LOG"
else
  echo "[run_live_e2e] failed ($FAILURES, session_id=${SESSION_ID}). log: $LOG" >&2
fi

exit "$FAILURES"
