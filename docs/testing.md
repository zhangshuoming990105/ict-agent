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
| `test_fork_tools.py` | 8 | fork 工具：fork_subagent、drain、status、wait |
| `test_preemption.py` | 1 | 抢占标志 roundtrip |
| `test_skills.py` | 3 | skills 加载、选择、fork 排除 |
| `test_task_manager.py` | 3 | 本地 task 解析、prompt 加载、workspace summary |
| **合计** | **20** | |

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
| `test_live_session_smoke.py` | 2 | `test_live_session_smoke`: 跑 `run_live_e2e.sh`（多轮对话、工具、/debug） | live session + API |
| | | `test_fork_skill_smoke`: 跑 `run_fork_smoke.sh`（/run scout + 主副上下文） | live session + API |
| **合计** | **2** | | |

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
| `scripts/run_fork_smoke.sh` | /run scout 烟雾；被 `test_fork_skill_smoke` 调用 |
| `scripts/run_live_e2e.sh` | 多轮 e2e；被 `test_live_session_smoke` 调用 |
| `scripts/run_multi_fork_test.py` | 多 fork（2 个 qa）手动/可选验证 |
| `scripts/test_basic_live_session.sh` | 基础 live 会话 |

---

## 汇总

| Level | 目录 | 测试数 | CI 要求 |
|-------|------|--------|---------|
| Unit | `tests/unit/` | 20 | ✅ 必须通过 |
| Integration mock | `tests/integration_mock_api/` | 1 | ✅ 必须通过 |
| Integration real_api | `tests/integration_real_api/` | 2 | ⏭️ 按条件 skip（无 API 时跳过） |
| **总计** | | **23** | |

CI 策略：**每次 MR/PR 跑 Unit + Integration mock（21 个）必须绿；real_api 在未配置或未显式启用时不跑（skip），在满足依赖的环境（如 nightly 或手动触发）下再跑。**

---

## CI/CD 说明

- **默认流水线**（`.github/workflows/test.yml`）：对每次 push/PR 跑 `pytest tests/`。不设置 `ICT_AGENT_RUN_REAL_API`，因此 2 个 real_api 用例会被 **skip**，只要求 21 个 unit + mock 通过。
- **若要在 CI 里跑 real_api**：在 workflow 中增加环境变量 `ICT_AGENT_RUN_REAL_API=1`，并配置好 API 相关 secret（如 `OPENAI_API_KEY` 或你们用的模型 endpoint 等）。可单独做一个 workflow（如 `test-real-api.yml`）或 manual dispatch，用 secret 注入后再跑，避免在普通 PR 里依赖真实 API。
