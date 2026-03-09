# 测试名单与层级 (Test Roster by Level)

## 层级说明

| Level | 说明 | 依赖 | CI 策略 |
|-------|------|------|---------|
| **Unit** | 纯逻辑单测，无网络/进程 | 无（bubblewrap 可选，sandbox 活体测试需要） | 必须全部通过 |
| **Integration (mock)** | 集成测，用 mock API | 无 | 必须全部通过 |
| **Integration (real_api)** | 真实 streaming + live session | 需 API/模型、环境变量 | 按条件 skip，满足时再跑 |

---

## 1. Unit 测试 (`tests/unit/`)

无外部依赖，仅需 `PYTHONPATH=src`（或 `pip install -e .`）。

| 文件 | 测试数 | 说明 |
|------|--------|------|
| `test_commands.py` | 5 | `/set-model`, `/run`, `/fork` 等命令行为 |
| `test_edit_file.py` | 13 | edit_file 工具、edit_diff 工具（fuzzy match、行尾、BOM） |
| `test_enhancements.py` | 29 | 6 大增强功能: workspace=cwd, 大输出落盘, max_tokens, 动态 schema, streaming(mock), 沙箱(含 bwrap/seatbelt 活体) |
| `test_fork_tools.py` | 8 | fork 工具：fork_subagent、drain、status、wait |
| `test_preemption.py` | 1 | 抢占标志 roundtrip |
| `test_skills.py` | 3 | skills 加载、选择、fork 排除 |
| `test_task_manager.py` | 3 | 本地 task 解析、prompt 加载、workspace summary |
| **合计** | **49** | (sandbox 活体测试在无 bwrap/seatbelt 时 skip) |

```bash
PYTHONPATH=src python -m pytest tests/unit/ -v
```

---

## 2. Integration 测试 - Mock API (`tests/integration_mock_api/`)

使用 mock API，无真实 LLM 调用。

| 文件 | 测试数 | 说明 |
|------|--------|------|
| `test_runtime_smoke.py` | 2 | 工具调用执行（calculator、edit_file） |
| **合计** | **1** | |

```bash
PYTHONPATH=src python -m pytest tests/integration_mock_api/ -v
```

---

## 3. Integration 测试 - Real API (`tests/integration_real_api/`)

依赖：**真实 LLM API**（Anthropic mcs-1 + OpenAI gpt-oss-120b）。
通过环境变量控制：**未设置 `ICT_AGENT_RUN_REAL_API=1` 时自动 skip**。

只保留**必须调用真实 API** 的测试。不需要 API 的功能（workspace、大输出、动态 schema、shell 安全、沙箱）已在 unit 测试中完整覆盖。

| 文件 | 测试数 | 功能 | 说明 |
|------|--------|------|------|
| `test_enhancements_real.py` | 6 | Dual-provider streaming | Anthropic: text + tool_call + prompt_caching; OpenAI: text + tool_call; /model switch dispatch |
| `test_live_session_smoke.py` | 1 | Agent loop e2e | `test_mixed_e2e_smoke`: 3 轮对话 (calculator+time, write+shell+sandbox, /tokens) |
| **合计** | **7** | | |

```bash
# 不跑 real_api（默认 skip）
PYTHONPATH=src python -m pytest tests/ -v -m "not real_api"

# 跑 real_api（需配置 API 且设置环境变量）
ICT_AGENT_RUN_REAL_API=1 PYTHONPATH=src python -m pytest tests/integration_real_api/ -v
```

---

## 脚本级"测试"（未纳入 pytest CI，可手动运行）

| 脚本 | Session ID | 说明 |
|------|-----------|------|
| `scripts/run_mixed_e2e.py` | 4 | 3 轮混合 e2e (calculator, time, write, shell+sandbox)；被 `test_mixed_e2e_smoke` 调用 |
| `scripts/run_live_e2e.py` | 0 | 5 轮 e2e (time, write, read, compile, run)；手动运行 |
| `scripts/run_fork_smoke.py` | 1 | /run scout 烟雾；手动运行 |
| `scripts/run_enhancements_e2e.py` | 2 | 增强功能 e2e（workspace、大输出、safe-shell）；手动运行 |
| `scripts/run_multi_fork_test.py` | 3 | 多 fork（2 个 qa）手动验证 |

---

## 汇总

| Level | 目录 | 测试数 | CI 要求 |
|-------|------|--------|---------|
| Unit | `tests/unit/` | 62 | ✅ 必须通过 |
| Integration mock | `tests/integration_mock_api/` | 2 | ✅ 必须通过 |
| Integration real_api | `tests/integration_real_api/` | 7 | ⏭️ 按条件 skip（无 API 时跳过） |
| **总计** | | **71** | |

CI 策略：**每次 push/PR 跑 Unit + Integration mock（64 个）必须绿；real_api 仅在 commit message 含 `[real-api]` 或手动触发时跑 7 个测试。**

---

## CI/CD 说明

### 默认流水线（每次 push/PR）

- **Workflow**：`.github/workflows/test.yml`
- **行为**：跑 64 个 unit + mock 测试，real_api 被 skip。安装 bubblewrap 以支持 sandbox 单元测试。
- **不需要** 配置任何 API secret。

### Real API 测试（按需）

- **Workflow**：`.github/workflows/test-real-api.yml`
- **触发条件**：
  - commit message 含 `[real-api]`：`git commit -m "feat: xxx [real-api]" && git push`
  - 手动：GitHub Actions → "Tests (Real API)" → "Run workflow"
- **前置条件**：Settings → Secrets → 添加 `KSYUN_API_KEY` 或 `INFINI_API_KEY`
- **内容**：3 个测试（2 streaming + 1 live session），约 20 秒，带 15 分钟 timeout
- **系统依赖**：自动安装 bubblewrap（sandbox live session 测试需要）

### 本地运行

```bash
# Unit + mock（无需 API）
python -m pytest tests/unit tests/integration_mock_api -v        # 64 tests

# Real API（需 API key）
ICT_AGENT_RUN_REAL_API=1 python -m pytest tests/integration_real_api -v  # 7 tests

# 手动 e2e 脚本
python scripts/run_mixed_e2e.py -v          # 3 轮混合
python scripts/run_live_e2e.py -v           # 5 轮完整
python scripts/run_fork_smoke.py -v         # fork/scout
```
