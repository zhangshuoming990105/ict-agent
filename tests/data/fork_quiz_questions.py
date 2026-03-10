"""16 simple questions for multi-fork QA tests. Each subagent can be assigned a subset."""

QUESTIONS = [
    "How many letter r in the word strawberry? Reply with one number.",
    "What is the current date and time? Use get_current_time if needed.",
    "What is 15 + 27? Reply with one number.",
    "What is 100 divided by 4? Reply with one number.",
    "How many vowels in the word hello? Reply with one number.",
    "What is 7 times 8? Reply with one number.",
    "What is the capital of France? One word.",
    "How many letters in the word python? One number.",
    "What is 2 to the power of 10? One number.",
    "What color is the sky on a clear day? One word.",
    "How many legs does a spider have? One number.",
    "What is 17 minus 9? One number.",
    "What is the largest planet in our solar system? One word.",
    "How many continents are there? One number.",
    "What is 144 divided by 12? One number.",
    "What is 9 times 7? One number.",
]


def partition_questions(n_subagents: int) -> list[list[str]]:
    """Split QUESTIONS into n_subagents groups (by index). Returns list of n_subagents lists."""
    assert 1 <= n_subagents <= 16
    q = list(QUESTIONS)
    size = len(q) // n_subagents
    remainder = len(q) % n_subagents
    groups = []
    start = 0
    for i in range(n_subagents):
        count = size + (1 if i < remainder else 0)
        groups.append(q[start : start + count])
        start += count
    return groups


def format_task_for_agent(questions: list[str], agent_index: int) -> str:
    """Format a subset of questions as a single task string for one subagent."""
    lines = [f"Answer these questions (agent {agent_index + 1}):"]
    for i, q in enumerate(questions, 1):
        lines.append(f"{i}. {q}")
    return "\n".join(lines)
