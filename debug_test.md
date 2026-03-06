# Debug Test Guide

这份文档面向后续开发者和其他 LLM agent，说明：

- 如何用 **live session** 而不是一次性脚本来调试 `ict-agent`
- 如何验证 provider / model 切换
- 如何在切模型后检查上下文是否仍然保留

## 基本原则

调试 `ict-agent` 时，**优先使用已经搭好的 live session 机制**，不要默认自己重新发明一个临时 REPL。

推荐流程：

1. 先重置现场
2. 启动一个 live session
3. 通过 `send` 逐轮发消息
4. 观察 `.live_session/session_<id>/stdout.log`
5. 用 `/tokens`、`/debug`、`/model` 等命令检查状态

注意：

- `reset_live_session.sh` 现在只重置 live-session 运行现场
- 如果你还想删除持久化日志，再单独执行 `bash scripts/clean_logs.sh`

## 每次调试前先 reset

```bash
cd ict-agent
bash scripts/reset_live_session.sh
```

## 启动一个调试 session

默认 session：

```bash
bash scripts/live_session.sh start
```

指定 session：

```bash
bash scripts/live_session.sh --session-id 3 start
```

## 发送输入

```bash
bash scripts/live_session.sh send "请记住这个数字：42"
bash scripts/live_session.sh send "/tokens"
bash scripts/live_session.sh send "/debug"
```

## 查看实时输出

```bash
tail -f .live_session/session_0/stdout.log
```

## 调试 provider / model

### 查看 provider 选项

```bash
python main.py --list-providers
```

### 查看某个 provider 的模型

```bash
python main.py --provider ksyun --list-models
python main.py --provider infini --list-models
```

### 运行中切模型

现在推荐用：

```bash
bash scripts/live_session.sh send "/model glm-5"
```

兼容旧写法：

```bash
bash scripts/live_session.sh send "/set-model glm-5"
```

查看当前模型：

```bash
bash scripts/live_session.sh send "/model"
bash scripts/live_session.sh send "/tokens"
```

## 验证“切模型后上下文是否延续”的推荐步骤

例如你想验证：

- 默认模型：`mco-4`
- 再依次切：
  - `mcs-1`
  - `glm-5`
  - `deepseek-v3.2`
  - `gpt-oss-120b`
  - `qwen3-coder-480b-a35b-instruct`

建议按下面的 live-session 流程做：

### Step 1：建立上下文

```bash
bash scripts/live_session.sh send "请记住这个数字：42。然后告诉我它是奇数还是偶数。"
```

### Step 2：切模型

```bash
bash scripts/live_session.sh send "/model mcs-1"
bash scripts/live_session.sh send "/model glm-5"
bash scripts/live_session.sh send "/model deepseek-v3.2"
bash scripts/live_session.sh send "/model gpt-oss-120b"
bash scripts/live_session.sh send "/model qwen3-coder-480b-a35b-instruct"
```

### Step 3：每次切换后验证上下文

```bash
bash scripts/live_session.sh send "你之前记住的那个数字，乘以2是多少？"
```

如果它还能稳定回答 `84`，说明上下文保留是正常的。

## 推荐的真实调试套路

如果你是另一个 LLM agent，在不确定系统行为时，优先按下面套路：

1. `reset_live_session.sh`
2. `live_session.sh start`
3. `live_session.sh send "..."` 逐轮发消息
4. 看 `.live_session/session_<id>/stdout.log`
5. 用 `/tokens`、`/debug`、`/model`
6. 结束后用 `close_session.sh`

## 不要做的事

- 不要默认跳过 live session 直接假设内部状态
- 不要在没看 log 的情况下猜测 agent 是否执行了工具
- 不要把 `\debug` 当标准写法，虽然现在兼容，但优先用 `/debug`
- 不要在复杂调试前忘记 `reset_live_session.sh`
