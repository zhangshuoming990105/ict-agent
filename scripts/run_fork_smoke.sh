#!/usr/bin/env bash
# Smoke test for fork skill. Prefer scripts/run_fork_smoke.py (same style as run_multi_fork_test.py).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_ID="${SESSION_ID:-0}"
LOG=""
SCRIPTS="$ROOT/scripts"
MODEL="${MODEL:-gpt-oss-120b}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-id) SESSION_ID="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

LOG="$ROOT/.live_session/session_${SESSION_ID}/stdout.log"
FAILURES=0
fail() { echo "[FAIL] $*" >&2; FAILURES=$((FAILURES + 1)); }
WAIT_TIMEOUT="${WAIT_TIMEOUT:-90}"

cleanup() {
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
      echo "[run_fork_smoke] timeout waiting for ready count>=$expected (got $count)" >&2
      return 1
    fi
    sleep 2
  done
}

trap cleanup EXIT

bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" stop >/dev/null 2>&1 || true
bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" start --model "$MODEL" --compact-model "$MODEL"
wait_ready 1

bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" send '/run scout list all Python files under src/ict_agent and report how many'
wait_ready 2

if ! grep -q '\[fork:scout\]' "$LOG" 2>/dev/null; then
  fail "Log does not contain [fork:scout]"
fi
if ! grep -q 'Calling tool: list_directory\|Calling tool: read_file\|Calling tool: search_files' "$LOG" 2>/dev/null; then
  fail "Log does not show scout tool calls (list_directory/read_file/search_files)"
fi
if ! grep -q 'Fork result injected\|\[subagent scout\]' "$LOG" 2>/dev/null; then
  fail "Log does not show fork result injected or [subagent scout]"
fi

bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" send 'Based on the scout result above, how many Python files were reported? Reply briefly with the number or a one-line summary.'
wait_ready 3

if ! grep -q 'Assistant' "$LOG" 2>/dev/null; then
  fail "Log does not show any Assistant reply"
fi
if ! grep -q 'Ready for input' "$LOG" 2>/dev/null; then
  fail "Log does not show Ready for input"
fi
ready_count=$(grep -c '>>> Ready for input.' "$LOG" 2>/dev/null) || ready_count=0
if [ "$ready_count" -lt 3 ]; then
  fail "Expected 3 Ready signals (start, after /run, after follow-up); got $ready_count"
fi

bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" send '/fork-status'
sleep 2
if ! grep -q 'No fork subagents running\|all finished' "$LOG" 2>/dev/null; then
  fail "Fork status should report no running subagents after /run and follow-up (check for leftover threads)"
fi

if [ "$FAILURES" -eq 0 ]; then
  echo "[run_fork_smoke] passed (session_id=${SESSION_ID}). log: $LOG"
else
  echo "[run_fork_smoke] failed ($FAILURES, session_id=${SESSION_ID}). log: $LOG" >&2
fi
exit "$FAILURES"
