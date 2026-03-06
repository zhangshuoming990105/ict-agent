#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_ID="${ICT_AGENT_SESSION_ID:-0}"
LINES="${LINES:-80}"
ASSISTANT_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/read_session_output.sh [--session-id ID] [--lines N] [--assistant-only]

Examples:
  bash scripts/read_session_output.sh --session-id 1
  bash scripts/read_session_output.sh --session-id 1 --assistant-only
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-id)
      SESSION_ID="$2"
      shift 2
      ;;
    --lines)
      LINES="$2"
      shift 2
      ;;
    --assistant-only)
      ASSISTANT_ONLY=1
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

LOG="$ROOT/.live_session/session_${SESSION_ID}/stdout.log"
if [ ! -f "$LOG" ]; then
  echo "session log not found: $LOG" >&2
  exit 1
fi

if [ "$ASSISTANT_ONLY" -eq 0 ]; then
  tail -n "$LINES" "$LOG"
  exit 0
fi

python - "$LOG" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

blocks = []
current = []
capturing = False
stop_prefixes = ("  [tokens:", ">>> Ready for input.", "You:", "  -> Calling tool:", "  <- Result:")

for line in lines:
    if line.startswith("Assistant:"):
        if current:
            blocks.append(current)
        current = [line]
        capturing = True
        continue
    if capturing:
        if line.startswith(stop_prefixes):
            blocks.append(current)
            current = []
            capturing = False
        else:
            current.append(line)

if current:
    blocks.append(current)

if not blocks:
    print("No assistant output found.")
else:
    print("\n".join(blocks[-1]))
PY
