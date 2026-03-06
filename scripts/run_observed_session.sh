#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
SESSION_ID="${SESSION_ID:-0}"
LOG=""
WAIT_TIMEOUT="${WAIT_TIMEOUT:-180}"
POLL_INTERVAL="${POLL_INTERVAL:-1}"
KEEP_SESSION=0
MESSAGES_FILE=""
LAST_PRINTED_LINE=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_observed_session.sh [--session-id ID] [--messages-file FILE] [--keep-session] [-- agent args...]

Examples:
  bash scripts/run_observed_session.sh
  bash scripts/run_observed_session.sh --messages-file scripts/sample_messages.txt
  bash scripts/run_observed_session.sh -- --model mco-4
  bash scripts/run_observed_session.sh --messages-file prompts.txt -- --task level1/001

Behavior:
  - starts a clean live session
  - streams newly appended session log lines to stdout in real time
  - sends one message at a time and waits for the next '>>> Ready for input.'
  - does not assume expected answers in advance
  - stops the session automatically unless --keep-session is set
EOF
}

cleanup() {
  if [ "$KEEP_SESSION" -eq 0 ]; then
    bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" stop >/dev/null 2>&1 || true
  fi
}

ready_count() {
  local n
  n=$(grep -c '>>> Ready for input.' "$LOG" 2>/dev/null) || n=0
  echo "$n"
}

log_line_count() {
  if [ -f "$LOG" ]; then
    wc -l < "$LOG"
  else
    echo 0
  fi
}

print_new_log_lines() {
  local total
  total=$(log_line_count)
  if [ "$total" -gt "$LAST_PRINTED_LINE" ]; then
    sed -n "$((LAST_PRINTED_LINE + 1)),$((total))p" "$LOG"
    LAST_PRINTED_LINE=$total
  fi
}

wait_ready_and_stream() {
  local expected=$1
  local timeout="${2:-$WAIT_TIMEOUT}"
  local deadline=$(( $(date +%s) + timeout ))
  while true; do
    print_new_log_lines
    if [ "$(ready_count)" -ge "$expected" ]; then
      return 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "[run_observed_session] timeout waiting for ready signal >= $expected" >&2
      return 1
    fi
    sleep "$POLL_INTERVAL"
  done
}

send_and_observe() {
  local msg="$1"
  local before
  before=$(ready_count)
  bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" send "$msg" >/dev/null
  wait_ready_and_stream $((before + 1))
}

run_interactive() {
  while true; do
    printf '\nobserver> '
    IFS= read -r msg || break
    if [ -z "${msg}" ]; then
      continue
    fi
    if [ "$msg" = "quit" ] || [ "$msg" = "exit" ]; then
      bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" send "quit" >/dev/null || true
      sleep 1
      print_new_log_lines
      return 0
    fi
    send_and_observe "$msg"
  done
}

run_file() {
  while IFS= read -r msg || [ -n "$msg" ]; do
    if [ -z "$msg" ]; then
      continue
    fi
    send_and_observe "$msg"
  done < "$MESSAGES_FILE"
  bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" send "quit" >/dev/null || true
  sleep 1
  print_new_log_lines
}

AGENT_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-id)
      SESSION_ID="$2"
      shift 2
      ;;
    --messages-file)
      MESSAGES_FILE="$2"
      shift 2
      ;;
    --keep-session)
      KEEP_SESSION=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      AGENT_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cd "$ROOT"
trap cleanup EXIT

LOG="$ROOT/.live_session/session_${SESSION_ID}/stdout.log"

bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" stop >/dev/null 2>&1 || true
bash "$SCRIPTS/live_session.sh" --session-id "$SESSION_ID" start "${AGENT_ARGS[@]}"
LAST_PRINTED_LINE=0
wait_ready_and_stream 1 60

if [ -n "$MESSAGES_FILE" ]; then
  run_file
else
  run_interactive
fi

print_new_log_lines
echo "[run_observed_session] finished (session_id=${SESSION_ID}). log: $LOG"
