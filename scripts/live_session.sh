#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_ID="${ICT_AGENT_SESSION_ID:-0}"
TTL="${ICT_AGENT_SESSION_TTL:-3600}"
STATE_ROOT="${ROOT_DIR}/.live_session"
STATE_DIR=""
FIFO_PATH=""
LOG_PATH=""
PID_PATH=""
KEEPER_PID_PATH=""
WRAPPER_PID_PATH=""
TTL_PID_PATH=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/live_session.sh [--session-id ID] [--ttl SEC] start [agent args...]
  bash scripts/live_session.sh [--session-id ID] send "<message>"
  bash scripts/live_session.sh [--session-id ID] status
  bash scripts/live_session.sh [--session-id ID] stop
  bash scripts/live_session.sh [--session-id ID] paths
EOF
}

setup_paths() {
  STATE_DIR="${STATE_ROOT}/session_${SESSION_ID}"
  FIFO_PATH="${STATE_DIR}/stdin.fifo"
  LOG_PATH="${STATE_DIR}/stdout.log"
  PID_PATH="${STATE_DIR}/pid"
  KEEPER_PID_PATH="${STATE_DIR}/fifo_keeper.pid"
  WRAPPER_PID_PATH="${STATE_DIR}/wrapper.pid"
  TTL_PID_PATH="${STATE_DIR}/ttl.pid"
}

parse_global_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --session-id)
        SESSION_ID="${2:-}"
        shift 2
        ;;
      --ttl)
        TTL="${2:-}"
        shift 2
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        break
        ;;
    esac
  done
  setup_paths
  REMAINING_ARGS=("$@")
}

ensure_state_dir() {
  mkdir -p "${STATE_DIR}"
}

is_running() {
  if [[ ! -f "${PID_PATH}" ]]; then
    return 1
  fi
  local pid
  pid="$(<"${PID_PATH}")"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && \
    ps -p "${pid}" -o command= 2>/dev/null | grep -q 'main.py'
}

start_fifo_keeper() {
  (
    exec 9<>"${FIFO_PATH}"
    while :; do
      sleep 3600
    done
  ) &
  echo "$!" > "${KEEPER_PID_PATH}"
}

stop_fifo_keeper() {
  if [[ -f "${KEEPER_PID_PATH}" ]]; then
    local kpid
    kpid="$(<"${KEEPER_PID_PATH}")"
    if [[ -n "${kpid}" ]]; then
      kill "${kpid}" 2>/dev/null || true
    fi
    rm -f "${KEEPER_PID_PATH}"
  fi
}

stop_ttl_timer() {
  if [[ -f "${TTL_PID_PATH}" ]]; then
    local tpid
    tpid="$(<"${TTL_PID_PATH}")"
    if [[ -n "${tpid}" ]]; then
      kill "${tpid}" 2>/dev/null || true
    fi
    rm -f "${TTL_PID_PATH}"
  fi
}

cleanup_stale_session_files() {
  rm -f "${PID_PATH}" "${KEEPER_PID_PATH}" "${WRAPPER_PID_PATH}" "${TTL_PID_PATH}"
  if [[ -p "${FIFO_PATH}" ]]; then
    rm -f "${FIFO_PATH}"
  fi
}

start_ttl_timer() {
  if [[ "${TTL}" -le 0 ]]; then
    return 0
  fi
  (
    sleep "${TTL}"
    bash "${ROOT_DIR}/scripts/close_session.sh" --session-id "${SESSION_ID}" >/dev/null 2>&1 || true
  ) &
  echo "$!" > "${TTL_PID_PATH}"
}

read_pid_file_or_unknown() {
  local file="$1"
  if [[ -f "${file}" ]]; then
    local value
    value="$(<"${file}")"
    if [[ -n "${value}" ]]; then
      echo "${value}"
      return 0
    fi
  fi
  echo "unknown"
}

require_running() {
  if ! is_running; then
    echo "No live session running for session_id=${SESSION_ID}." >&2
    echo "Start one with: bash scripts/live_session.sh --session-id ${SESSION_ID} start ..." >&2
    exit 1
  fi
}

cmd_start() {
  ensure_state_dir
  if is_running; then
    echo "A live session is already running for session_id=${SESSION_ID} (pid $(<"${PID_PATH}"))." >&2
    exit 1
  fi

  rm -f "${FIFO_PATH}" "${LOG_PATH}" "${PID_PATH}" "${WRAPPER_PID_PATH}" "${TTL_PID_PATH}"
  mkfifo "${FIFO_PATH}"
  start_fifo_keeper

  (
    cd "${ROOT_DIR}"
    ICT_AGENT_SESSION_ID="${SESSION_ID}" \
    ICT_AGENT_SESSION_TTL="${TTL}" \
    PYTHONUNBUFFERED=1 stdbuf -oL -eL \
      python -u main.py "$@" < "${FIFO_PATH}" > >(tee -a "${LOG_PATH}") 2>&1 &
    py_pid=$!
    echo "${py_pid}" > "${PID_PATH}"
    wait "${py_pid}"
  ) &
  local wrapper_pid=$!
  echo "${wrapper_pid}" > "${WRAPPER_PID_PATH}"
  local deadline=$(( $(date +%s) + 5 ))
  while [[ ! -f "${PID_PATH}" ]]; do
    if [[ "$(date +%s)" -ge "${deadline}" ]]; then
      break
    fi
    sleep 0.1
  done
  start_ttl_timer

  echo "session_id=${SESSION_ID}"
  echo "started_pid=$(read_pid_file_or_unknown "${PID_PATH}")"
  echo "wrapper_pid=${wrapper_pid}"
  echo "ttl=${TTL}"
  echo "fifo=${FIFO_PATH}"
  echo "log=${LOG_PATH}"
}

cmd_send() {
  require_running
  local msg="${*:-}"
  if [[ -z "${msg}" ]]; then
    echo "send requires a non-empty message." >&2
    exit 1
  fi
  printf '%s\n' "${msg}" > "${FIFO_PATH}"
  echo "sent: ${msg}"
}

cmd_status() {
  if is_running; then
    echo "running=true"
    echo "session_id=${SESSION_ID}"
    echo "pid=$(<"${PID_PATH}")"
    if [[ -f "${WRAPPER_PID_PATH}" ]]; then
      echo "wrapper_pid=$(<"${WRAPPER_PID_PATH}")"
    fi
    if [[ -f "${TTL_PID_PATH}" ]]; then
      echo "ttl_pid=$(<"${TTL_PID_PATH}")"
    fi
    echo "ttl=${TTL}"
  else
    echo "running=false"
    stop_fifo_keeper
    stop_ttl_timer
    cleanup_stale_session_files
  fi
  echo "fifo=${FIFO_PATH}"
  echo "log=${LOG_PATH}"
}

cmd_stop() {
  if is_running; then
    printf 'quit\n' > "${FIFO_PATH}" || true
    local pid
    pid="$(<"${PID_PATH}")"
    local wrapper_pid=""
    if [[ -f "${WRAPPER_PID_PATH}" ]]; then
      wrapper_pid="$(<"${WRAPPER_PID_PATH}")"
    fi
    stop_ttl_timer
    local deadline=$(( $(date +%s) + 5 ))
    while is_running; do
      if [[ "$(date +%s)" -ge "${deadline}" ]]; then
        break
      fi
      sleep 1
    done
    if is_running; then
      kill "${pid}" 2>/dev/null || true
    fi
    if [[ -n "${wrapper_pid}" ]] && kill -0 "${wrapper_pid}" 2>/dev/null; then
      kill "${wrapper_pid}" 2>/dev/null || true
    fi
    stop_fifo_keeper
    cleanup_stale_session_files
    echo "stopped"
  else
    echo "no running session"
    stop_fifo_keeper
    stop_ttl_timer
    cleanup_stale_session_files
  fi
}

cmd_paths() {
  echo "session_id=${SESSION_ID}"
  echo "fifo=${FIFO_PATH}"
  echo "log=${LOG_PATH}"
  echo "pid_file=${PID_PATH}"
  echo "keeper_pid_file=${KEEPER_PID_PATH}"
  echo "wrapper_pid_file=${WRAPPER_PID_PATH}"
  echo "ttl_pid_file=${TTL_PID_PATH}"
}

main() {
  local sub
  parse_global_args "$@"
  sub="${REMAINING_ARGS[0]:-}"
  case "${sub}" in
    start)
      cmd_start "${REMAINING_ARGS[@]:1}"
      ;;
    send)
      cmd_send "${REMAINING_ARGS[@]:1}"
      ;;
    status)
      cmd_status
      ;;
    stop)
      cmd_stop
      ;;
    paths)
      cmd_paths
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
