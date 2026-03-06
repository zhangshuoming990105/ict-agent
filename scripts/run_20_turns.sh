#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_ID="${SESSION_ID:-0}"
LOG=""
TURN_TIMEOUT="${TURN_TIMEOUT:-90}"
SKIPPED=0

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

ready_count() {
  local n
  n=$(grep -c '>>> Ready for input.' "$LOG" 2>/dev/null) || n=0
  echo "$n"
}

wait_ready() {
  local expected=$1
  local deadline=$(( $(date +%s) + TURN_TIMEOUT ))
  while true; do
    [ "$(ready_count)" -ge "$expected" ] && return 0
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "    [TIMEOUT] turn did not complete within ${TURN_TIMEOUT}s - skipping"
      SKIPPED=$(( SKIPPED + 1 ))
      return 1
    fi
    sleep 2
  done
}

send() {
  local msg="$1"
  local expected=$2
  bash "$ROOT/scripts/live_session.sh" --session-id "$SESSION_ID" send "$msg" >/dev/null
  wait_ready "$expected" || true
}

send "现在几点？用工具查一下" 2
send "在 temp/t20 下写一个 hello.c，打印 Hello World" 3
send "在 temp/t20 下写一个 factorial.c，计算 0 到 10 的阶乘" 4
send "读 temp/t20/hello.c 和 temp/t20/factorial.c 确认内容" 5
send "用 gcc 编译 temp/t20/hello.c 生成 temp/t20/hello" 6
send "用 gcc 编译 temp/t20/factorial.c 生成 temp/t20/factorial" 7
send "运行 temp/t20/hello" 8
send "运行 temp/t20/factorial" 9
send "在 temp/t20 下写一个 fib.py，计算斐波那契数列前 20 项并打印" 10
send "运行 temp/t20/fib.py" 11
send "对 temp/t20/hello.c 用 append 模式追加一行注释 /* compiled and tested */" 12
send "读 temp/t20/hello.c 确认追加成功" 13
send "重新编译 temp/t20/hello.c，再运行一次" 14
send "在 temp/t20 下写一个 primes.py，输出 100 以内的质数" 15
send "运行 temp/t20/primes.py" 16
send "列出 temp/t20 目录下所有文件" 17
send "计算 sqrt(2) + pi 的值，用 calculator 工具" 18
send "在 temp/t20 下写一个 stats.py，生成 20 个随机数并计算均值和标准差" 19
send "运行 temp/t20/stats.py" 20

bash "$ROOT/scripts/live_session.sh" --session-id "$SESSION_ID" send "/tokens" >/dev/null
wait_ready 21

bash "$ROOT/scripts/live_session.sh" --session-id "$SESSION_ID" send "/compact high" >/dev/null
wait_ready 22

bash "$ROOT/scripts/live_session.sh" --session-id "$SESSION_ID" send "/debug raw" >/dev/null
wait_ready 23

echo "[run_20_turns] done (skipped=$SKIPPED, session_id=${SESSION_ID})."
