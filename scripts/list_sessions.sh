#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_ROOT="$ROOT/.live_session"
CURRENT_SESSION_ID="${ICT_AGENT_SESSION_ID:-0}"

echo "current_session_id=${CURRENT_SESSION_ID}"

if [ ! -d "$STATE_ROOT" ]; then
  echo "no session state directory"
  exit 0
fi

shopt -s nullglob
dirs=("$STATE_ROOT"/session_*)
shopt -u nullglob

if [ "${#dirs[@]}" -eq 0 ]; then
  echo "no sessions found"
  exit 0
fi

for dir in "${dirs[@]}"; do
  [ -d "$dir" ] || continue
  sid="${dir##*/session_}"
  pid_file="$dir/pid"
  fifo="$dir/stdin.fifo"
  log="$dir/stdout.log"
  running="false"
  pid=""
  if [ -f "$pid_file" ]; then
    pid="$(<"$pid_file")"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      running="true"
    fi
  fi
  echo "session_id=${sid} running=${running} pid=${pid:-none} fifo=${fifo} log=${log}"
done
