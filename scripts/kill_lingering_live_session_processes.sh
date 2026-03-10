#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

collect_wrapper_pids() {
  ps -axo pid=,command= | awk '
    /bash scripts\/live_session\.sh( --session-id [0-9]+)? start/ && $0 !~ /awk/ {
      print $1
    }
  '
}

collect_child_pids() {
  local wrapper_pids="$1"
  if [[ -z "$wrapper_pids" ]]; then
    return 0
  fi
  ps -axo pid=,ppid=,command= | awk -v wrappers="$wrapper_pids" '
    BEGIN {
      split(wrappers, arr, " ")
      for (i in arr) {
        if (arr[i] != "") ids[arr[i]] = 1
      }
    }
    ($2 in ids) && $0 !~ /awk/ {
      print $1
    }
  '
}

wrapper_pids="$(collect_wrapper_pids | xargs 2>/dev/null || true)"
child_pids="$(collect_child_pids "${wrapper_pids}" | xargs 2>/dev/null || true)"

all_pids="$(printf '%s %s\n' "$child_pids" "$wrapper_pids" | xargs 2>/dev/null || true)"

if [[ -z "${all_pids}" ]]; then
  echo "[kill_lingering_live_session_processes] no lingering live-session processes found."
  exit 0
fi

kill ${all_pids} 2>/dev/null || true
sleep 1
kill -9 ${all_pids} 2>/dev/null || true

echo "[kill_lingering_live_session_processes] cleaned pids: ${all_pids}"
