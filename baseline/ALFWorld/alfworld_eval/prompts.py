"""Zero-shot prompt for ALFWorld action selection. No demonstrations, no oracle info."""

SYSTEM_PROMPT = "You are an agent acting in a household environment."

USER_TEMPLATE = """Task:
{task}

Current observation:
{observation}

Recent history:
{history}

Available actions:
{actions}

Choose the single best next action to complete the task.
Return ONLY the exact action text from the available actions."""


def format_history(history):
    """history: list of (action, observation) tuples, oldest first."""
    if not history:
        return "(none)"
    return "\n".join(f"> {a}\n{o}" for a, o in history)


def build_messages(task, observation, history, admissible):
    actions = "\n".join(f"{i}. {c}" for i, c in enumerate(admissible))
    user = USER_TEMPLATE.format(
        task=task,
        observation=observation.strip(),
        history=format_history(history),
        actions=actions,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
