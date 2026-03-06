---
name: shell
description: Command-line execution skill for compilation, verification, profiling, and debugging commands.
tools:
  - run_shell
  - shell_policy_status
triggers:
  - shell
  - command
  - terminal
  - bash
  - compile
  - run
  - execute
always_on: false
---

# Shell Skill

Use `run_shell` for command execution. Common CUDA workflow commands:

```bash
bash utils/compile.sh
python -m utils.verification
python -m utils.profiling
```

When shell safety is on, `run_shell` asks for confirmation for unknown commands.
