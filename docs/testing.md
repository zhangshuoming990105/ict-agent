# 测试名单与层级 (Test Roster by Level)

## 层级说明

| Level | 说明 | 依赖 | CI 策略 |
|-------|------|------|---------|
| **Unit** | 纯逻辑单测，无网络/进程 | 无 | 必须全部通过 |
| **Integration (mock)** | 集成测，用 mock API | 无 | 必须全部通过 |
| **Integration (real_api)** | 真实 live session + 真实 LLM API | 需 API/模型、环境变量 | 按条件 skip，满足时再跑 |

---

## 1. Unit 测试 (`tests/unit/`)

无外部依赖，仅需 `PYTHONPATH=src`（或 `pip install -e .`）。

| 文件 | 测试数 | 说明 |
|------|--------|------|
| `test_commands.py` | 5 | `/set-model`, `/run`, `/fork` 等命令行为 |
| `test_enhancements.py` | 28 | 6 大增强功能: workspace=cwd, 大输出落盘, max_tokens, 动态 schema, streaming, 沙箱 |
| `test_fork_tools.py` | 8 | fork 工具：fork_subagent、drain、status、wait |
| `test_preemption.py` | 1 | 抢占标志 roundtrip |
| `test_skills.py` | 3 | skills 加载、选择、fork 排除 |
| `test_task_manager.py` | 3 | 本地 task 解析、prompt 加载、workspace summary |
| **合计** | **48** | |

```bash
PYTHONPATH=src python -m pytest tests/unit/ -v
```

---

## 2. Integration 测试 - Mock API (`tests/integration_mock_api/`)

使用 mock API，无真实 LLM 调用。

| 文件 | 测试数 | 说明 |
|------|--------|------|
| `test_runtime_smoke.py` | 1 | 工具调用执行（process_tool_calls） |
| **合计** | **1** | |

```bash
PYTHONPATH=src python -m pytest tests/integration_mock_api/ -v
```

---

## 3. Integration 测试 - Real API (`tests/integration_real_api/`)

依赖：**真实 LLM API**（如 gpt-oss-120b）、可用的 `main.py` live session。  
通过环境变量控制是否执行：**未设置 `ICT_AGENT_RUN_REAL_API=1` 时自动 skip**。

| 文件 | 测试数 | 说明 | 依赖 |
|------|--------|------|------|
| `test_live_session_smoke.py` | 2 | `test_live_session_smoke`: 跑 `run_live_e2e.py`（多轮对话、工具） | live session + API |
| | | `test_fork_skill_smoke`: 跑 `run_fork_smoke.py`（/run scout + 主副上下文） | live session + API |
| `test_enhancements_real.py` | 8 | streaming、大输出、沙箱、workspace 等新功能的真实 API 验证 | API (无需 live session) |
| **合计** | **10** | | |

```bash
# 不跑 real_api（默认 skip）
PYTHONPATH=src python -m pytest tests/ -v -m "not real_api"

# 跑 real_api（需配置 API 且设置环境变量）
ICT_AGENT_RUN_REAL_API=1 PYTHONPATH=src python -m pytest tests/integration_real_api/ -v
```

---

## 脚本级“测试”（未纳入 pytest）

以下由 real_api 用例或人工调用，不单独列在 pytest 名单中：

| 脚本 | 说明 |
|------|------|
| `scripts/run_fork_smoke.py` | /run scout 烟雾；被 `test_fork_skill_smoke` 调用 |
| `scripts/run_live_e2e.py` | 多轮 e2e；被 `test_live_session_smoke` 调用 |
| `scripts/run_enhancements_e2e.py` | 增强功能 e2e（workspace、大输出、safe-shell）；可独立运行 |
| `scripts/run_multi_fork_test.py` | 多 fork（2 个 qa）手动/可选验证 |
| `scripts/test_basic_live_session.sh` | 基础 live 会话 |

---

## 汇总

| Level | 目录 | 测试数 | CI 要求 |
|-------|------|--------|---------|
| Unit | `tests/unit/` | 48 | ✅ 必须通过 |
| Integration mock | `tests/integration_mock_api/` | 1 | ✅ 必须通过 |
| Integration real_api | `tests/integration_real_api/` | 10 | ⏭️ 按条件 skip（无 API 时跳过） |
| **总计** | | **59** | |

CI 策略：**每次 MR/PR 跑 Unit + Integration mock（49 个）必须绿；real_api 在未配置或未显式启用时不跑（skip），在满足依赖的环境（如 nightly 或手动触发）下再跑。**

---

## CI/CD 说明

### 什么时候会跑 real API 测试？

唯一条件：**环境变量 `ICT_AGENT_RUN_REAL_API=1`**。  
代码里通过 `@pytest.mark.skipif(os.getenv("ICT_AGENT_RUN_REAL_API") != "1", ...)` 控制，不设或不是 `"1"` 时这两个用例会直接 **skip**。

本地跑 real API 测试：

```bash
ICT_AGENT_RUN_REAL_API=1 PYTHONPATH=src python -m pytest tests/integration_real_api/ -v
```

同时需配置至少一个 API Key（与 `main.py` 一致）：`KSYUN_API_KEY` 或 `INFINI_API_KEY`。  
每个脚本有**超时**（默认 120 秒），超时未结束会报 `TimeoutExpired` 而非卡死；可设环境变量 `ICT_AGENT_REAL_API_TIMEOUT`（秒）覆盖。

### 为何 real API 测试会“卡死”？

- **原因**：主 agent 在等 LLM 返回时，原先用 `while not async_call.done.wait(0.1)` 无限轮询，**没有整体超时**。若 API 一直不返回（网络挂起、服务慢/不可用），主线程会一直等，不会打印下一次 `>>> Ready for input.`，脚本端的 `wait_ready` 就会等到自己的 120/180s 才报错，看起来像卡死。
- **修复**：在 `agent_loop.py` 主循环里为单轮模型调用增加 **120 秒** 上限；超时后置 `async_call.error = TimeoutError(...)` 并 `done.set()`，走原有异常分支，结束本 turn 并继续（或退出），脚本能在 120s 内要么通过要么明确超时失败。fork 子 agent 使用 `async_call.done.wait(timeout=300)`。

### 默认流水线（每次 push/PR）

- **Workflow**：`.github/workflows/test.yml`
- **行为**：不设置 `ICT_AGENT_RUN_REAL_API`，所以 10 个 real_api 用例被 skip，只跑 49 个 unit + mock，必须全过。
- **不跑** real API，因此不需要在 GitHub 配置任何 secret。

### 在 GitHub 里跑 real API 测试（可选）

单独 workflow：**`.github/workflows/test-real-api.yml`**，不会在每次 push 时跑，避免阻塞、耗时长、占 API 额度。

- **何时触发**
  - **手动**：GitHub 仓库 → Actions → 选 "Tests (Real API)" → "Run workflow"。
  - **定时**（可选）：默认每天 UTC 2:00（约北京时间 10:00）跑一次；可在 workflow 里改或删 `schedule`。
- **前置条件**
  - 在 **Settings → Secrets and variables → Actions** 里添加至少一个 **Secret**：`KSYUN_API_KEY` 或 `INFINI_API_KEY`。
  - 若都没配，该 workflow 的 job 会直接 skip（不会报错）。
- **跑的内容**：`pytest tests/integration_real_api/`，约 10 个用例（live e2e + fork smoke + 8 个增强功能），整体约几分钟，带 15 分钟 timeout。

这样 PR 只依赖 40s 的 unit+mock；real API 只在需要时手动或定时跑，并依赖你在 GitHub 配好的 API Key。
