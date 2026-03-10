---
name: core
description: General utility skill for basic math, time, and fallback assistant behavior.
tools:
  - calculator
  - get_current_time
  - shell_policy_status
triggers:
  - calculate
  - math
  - number
  - time
  - clock
  - policy
always_on: true
---

# Core Skill

Use utility tools when they clearly help:
- `calculator` for arithmetic and math expressions
- `get_current_time` for date and time queries
- `shell_policy_status` when the user asks about shell permissions

Keep answers concise and grounded in tool output.
