#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_ID="${SESSION_ID:-0}"
LOG=""
SCRIPTS="$ROOT/scripts"
FAILURES=0

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

fail() { echo "[FAIL] $*" >&2; FAILURES=$((FAILURES + 1)); }

cleanup() {
  bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" stop >/dev/null 2>&1 || true
}

ready_count() {
  local n
  n=$(grep -c '>>> Ready for input.' "$LOG" 2>/dev/null) || n=0
  echo "$n"
}

wait_ready() {
  local expected=$1
  local timeout="${2:-60}"
  local deadline=$(( $(date +%s) + timeout ))
  while true; do
    [ "$(ready_count)" -ge "$expected" ] && return 0
    [ "$(date +%s)" -ge "$deadline" ] && { echo "[TIMEOUT] waited ${timeout}s"; return 1; }
    sleep 2
  done
}

send() {
  local msg="$1"
  local expected=$2
  bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" send "$msg" >/dev/null
  wait_ready "$expected" || true
}

trap cleanup EXIT

bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" stop >/dev/null 2>&1 || true
bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" start --model mco-4
wait_ready 1 30

send "请记住这个数字：42。然后告诉我它是奇数还是偶数。" 2
send "/set-model gpt-oss-120b" 3
send "/tokens" 4
send "你之前记住的那个数字，乘以2是多少？" 5

bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" send "quit" >/dev/null || true

if ! grep -q "Model switched: mco-4 -> gpt-oss-120b" "$LOG" 2>/dev/null; then
  fail "Model switch log message NOT found"
fi

if ! grep -q "Model:.*gpt-oss-120b" "$LOG" 2>/dev/null; then
  fail "/tokens does not show gpt-oss-120b"
fi

if ! grep -q "84" "$LOG" 2>/dev/null; then
  fail "Follow-up answer does not contain 84"
fi

if ! grep -q "42" "$LOG" 2>/dev/null; then
  fail "42 not found - context may have been lost"
fi

if [ "$FAILURES" -eq 0 ]; then
  echo "[test_set_model] passed (session_id=${SESSION_ID}). log: $LOG"
else
  echo "[test_set_model] failed ($FAILURES, session_id=${SESSION_ID}). log: $LOG" >&2
fi
exit "$FAILURES"
