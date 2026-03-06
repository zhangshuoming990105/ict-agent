#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_ROOT="$ROOT/.live_session"
SCRIPTS="$ROOT/scripts"
SESSION_ID=""
REMOVE_LOGS=0
STOP_RUNNING=0
REMOVE_EMPTY_DIRS=1

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cleanup_live_session.sh [--session-id ID] [--stop-running] [--remove-logs]

Behavior:
  - cleans stale live-session metadata under .live_session
  - by default does NOT stop running sessions
  - by default preserves stdout.log files

Options:
  --session-id ID   clean only one session directory
  --stop-running    close running sessions before cleanup
  --remove-logs     also remove stdout.log files for stopped sessions
EOF
}

is_session_running() {
  local sid="$1"
  local pid_file="$STATE_ROOT/session_${sid}/pid"
  if [[ ! -f "$pid_file" ]]; then
    return 1
  fi
  local pid
  pid="$(<"$pid_file")"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

cleanup_legacy_root_state() {
  rm -f "$STATE_ROOT/pid" "$STATE_ROOT/fifo_keeper.pid"
  if [[ -p "$STATE_ROOT/stdin.fifo" ]]; then
    rm -f "$STATE_ROOT/stdin.fifo"
  fi
  if [[ "$REMOVE_LOGS" -eq 1 ]]; then
    rm -f "$STATE_ROOT/stdout.log"
  fi
}

cleanup_one_session() {
  local sid="$1"
  local dir="$STATE_ROOT/session_${sid}"
  [[ -d "$dir" ]] || return 0

  if is_session_running "$sid"; then
    if [[ "$STOP_RUNNING" -eq 1 ]]; then
      bash "$SCRIPTS/close_session.sh" --session-id "$sid" >/dev/null
    else
      echo "[cleanup_live_session] skip running session_id=${sid}" >&2
      return 0
    fi
  fi

  rm -f "$dir/pid" "$dir/fifo_keeper.pid"
  if [[ -p "$dir/stdin.fifo" ]]; then
    rm -f "$dir/stdin.fifo"
  fi
  if [[ "$REMOVE_LOGS" -eq 1 ]]; then
    rm -f "$dir/stdout.log"
  fi

  if [[ "$REMOVE_EMPTY_DIRS" -eq 1 ]]; then
    rmdir "$dir" 2>/dev/null || true
  fi

  echo "[cleanup_live_session] cleaned session_id=${sid}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-id)
      SESSION_ID="$2"
      shift 2
      ;;
    --remove-logs)
      REMOVE_LOGS=1
      shift
      ;;
    --stop-running)
      STOP_RUNNING=1
      shift
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

mkdir -p "$STATE_ROOT"
cleanup_legacy_root_state

if [[ -n "$SESSION_ID" ]]; then
  cleanup_one_session "$SESSION_ID"
  exit 0
fi

shopt -s nullglob
dirs=("$STATE_ROOT"/session_*)
shopt -u nullglob

for dir in "${dirs[@]}"; do
  [[ -d "$dir" ]] || continue
  sid="${dir##*/session_}"
  cleanup_one_session "$sid"
done
