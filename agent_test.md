# ICT Agent Live E2E Test

本文档定义 `ict-agent` 的 live e2e test 标准流程，供人类和 AI agent 共同使用。

若你需要更完整的 session 运维说明，包括：

- 可靠关闭
- 清理 `.live_session`
- 一键 reset
- 多 session
- agent-to-agent 驱动

请看：`LIVE_SESSION.md`

补充说明：

- 默认主 agent 现在是 general-purpose agent
- CUDA 约束应通过 task context / `cuda` skill 动态叠加
- session orchestration 能力应通过 `session` skill 动态叠加

## 目的

- 统一入口：人类和 AI 都按同一套流程测试
- 真实环境：单进程、FIFO 注入、完整日志
- 可观测：所有输出进入 `.live_session/stdout.log` 和 `logs/<timestamp>.log`

## 前置条件

```bash
cd ict-agent
pip install -e ".[dev]"
export KSYUN_API_KEY="your-key"
```

## 标准流程

### 1. 启动会话

```bash
bash scripts/live_session.sh stop || true
bash scripts/live_session.sh start
```

可选参数示例：

```bash
bash scripts/live_session.sh start --task level1/001 --preempt-shell-kill
bash scripts/live_session.sh --session-id 7 start
```

## 多 Session

`live_session.sh` 现在支持 `--session-id <id>`，默认是 `0`。
它也支持 `--ttl <seconds>`，默认自动关闭时间是 `3600` 秒。

这意味着你可以并行启动多个互不干扰的主 agent，例如：

```bash
bash scripts/live_session.sh --session-id 0 start
bash scripts/live_session.sh --session-id 1 start
bash scripts/live_session.sh --session-id 2 start
```

每个 session 都有独立的：

- FIFO
- pid 文件
- `.live_session/session_<id>/stdout.log`
- `logs/session_<id>/...`

这让人类、测试脚本或其他 agent 都可以把自己绑定到某一个指定 session，而不会互相干扰。

### 2. 发送输入

```bash
bash scripts/live_session.sh send "<消息内容>"
```

### 3. 判断当前 turn 是否完成

关键信号：log 中出现 `>>> Ready for input.`。

### 4. 查看日志

- 实时流：`.live_session/stdout.log`
- 持久化：`logs/<timestamp>.log`

### 5. 结束会话

```bash
bash scripts/live_session.sh send "quit"
```

若你想更可靠地关闭一个 session：

```bash
bash scripts/close_session.sh --session-id 0
```

若你想清理 `.live_session` 下的陈旧状态文件：

```bash
bash scripts/cleanup_live_session.sh
```

如果你想单独清理持久化 session 日志：

```bash
bash scripts/clean_logs.sh
```

可选地连 stopped session 的 live log 一起清掉：

```bash
bash scripts/cleanup_live_session.sh --remove-logs
```

## 推荐场景

### 场景 A：基础 slash 命令

- `/help`
- `/tokens`
- `/skills`
- `/preempt`

### 场景 B：任务加载与工具调用

```bash
bash scripts/live_session.sh send "/task load level1/001"
bash scripts/live_session.sh send "start"
```

### 场景 C：20 轮多工具回归

```bash
bash scripts/run_live_e2e.sh
```

### 场景 D：模型切换与上下文保持

```bash
bash scripts/test_set_model.sh
```

## 对 AI Agent 的约定

当用户说“跑 live e2e test”或“按 agent_test 测试”时：

1. 启动一个长生命周期 session
2. 每发送一条消息，都等待 `>>> Ready for input.` 再继续
3. 测完后检查日志中是否记录了 slash 命令、工具调用和最终回复
4. 若目标不明确，默认至少执行基础 slash 命令和一组工具调用验证
