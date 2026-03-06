#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_ID="${ICT_AGENT_SESSION_ID:-0}"
TIMEOUT="${TIMEOUT:-60}"
POLL_INTERVAL="${POLL_INTERVAL:-1}"
AFTER_LINES=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/wait_for_session_reply.sh --session-id ID [--after-lines N] [--timeout SEC]

Behavior:
  - waits until the target session produces a new assistant reply
  - if --after-lines is given, only accepts replies that appear after that log line count
  - prints the latest assistant reply block when found
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-id)
      SESSION_ID="$2"
      shift 2
      ;;
    --after-lines)
      AFTER_LINES="$2"
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

LOG="$ROOT/.live_session/session_${SESSION_ID}/stdout.log"
if [[ ! -f "$LOG" ]]; then
  echo "session log not found: $LOG" >&2
  exit 1
fi

deadline=$(( $(date +%s) + TIMEOUT ))

extract_latest_assistant_after_line() {
  python - "$LOG" "$AFTER_LINES" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
after_lines = int(sys.argv[2])
lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

blocks = []
current = []
capturing = False
start_line = None
stop_prefixes = ("  [tokens:", ">>> Ready for input.", "You:", "  -> Calling tool:", "  <- Result:")

for idx, line in enumerate(lines, start=1):
    if line.startswith("Assistant:"):
        if current and start_line is not None:
            blocks.append((start_line, current))
        current = [line]
        start_line = idx
        capturing = True
        continue
    if capturing:
        if line.startswith(stop_prefixes):
            if current and start_line is not None:
                blocks.append((start_line, current))
            current = []
            start_line = None
            capturing = False
        else:
            current.append(line)

if current and start_line is not None:
    blocks.append((start_line, current))

for start, block in reversed(blocks):
    if start > after_lines:
        print("\n".join(block))
        sys.exit(0)

sys.exit(1)
PY
}

while true; do
  if reply="$(extract_latest_assistant_after_line)"; then
    printf '%s\n' "$reply"
    exit 0
  fi
  if [[ "$(date +%s)" -ge "$deadline" ]]; then
    echo "timeout waiting for new assistant reply in session ${SESSION_ID}" >&2
    exit 1
  fi
  sleep "$POLL_INTERVAL"
done
