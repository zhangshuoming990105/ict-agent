#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="$ROOT/logs"
ALL_LOGS=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/clean_logs.sh [--all]

Behavior:
  - by default removes only live-session persistent logs under logs/session_*
  - with --all, removes every file and subdirectory under logs/
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      ALL_LOGS=1
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

mkdir -p "$LOG_ROOT"

if [[ "$ALL_LOGS" -eq 1 ]]; then
  rm -rf "$LOG_ROOT"/*
  echo "[clean_logs] removed all logs under $LOG_ROOT"
  exit 0
fi

rm -rf "$LOG_ROOT"/session_*
echo "[clean_logs] removed session logs under $LOG_ROOT/session_*"
