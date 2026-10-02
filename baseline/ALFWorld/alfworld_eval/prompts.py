"""Zero-shot prompts for ALFWorld action selection. No demonstrations, no oracle info.

strict: task + observation + history + actions, answer with the action only.
rich:   same, plus generic environment rules (how heat/cool/clean/look/pick work in
        ALFWorld) and progress hints. Still zero-shot: no example trajectories,
        no per-game information, no expert plans.
"""

SYSTEM_PROMPT = "You are an agent acting in a household environment."

RICH_SYSTEM_PROMPT = """You are an agent acting in a text-based household environment (ALFWorld).

How the environment works:
- You can only interact with things at your current location. Use "go to X" to move.
- Objects are inside or on receptacles (countertops, cabinets, drawers, shelves, fridge, ...). \
To find an object, visit likely receptacles; "open X" closed ones (cabinets, drawers, fridge, microwave, safe) to see inside.
- You can carry only one object at a time. "take OBJ from RECEP" picks it up; "move OBJ to RECEP" puts it down.
- To heat an object: take it, go to the microwave, then "heat OBJ with microwave".
- To cool an object: take it, go to the fridge, then "cool OBJ with fridge".
- To clean an object: take it, go to the sinkbasin, then "clean OBJ with sinkbasin".
- To look at/examine an object under a lamp: take the object, go to where the desklamp is, then "use desklamp".
- For "two" tasks, place the first object, then find and place the second one.
- "Nothing happens." means the action had no effect: do not repeat it, try something else.
- Do not keep revisiting the same places; explore receptacles you have not checked yet."""

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

RICH_USER_TEMPLATE = """Task:
{task}

Step {step} of {max_steps}.

Recent history:
{history}

Actions that had no effect so far (avoid repeating them):
{failed}

Current observation:
{observation}

Available actions:
{actions}

Think about which sub-goal you are on (find the object, take it, transform it, place it) and \
choose the single best next action to complete the task.
Return ONLY the exact action text from the available actions."""


def format_history(history):
    """history: list of (action, observation) tuples, oldest first."""
    if not history:
        return "(none)"
    return "\n".join(f"> {a}\n{o}" for a, o in history)


def build_messages(task, observation, history, admissible, style="strict",
                   step=0, max_steps=50, failed=()):
    actions = "\n".join(f"{i}. {c}" for i, c in enumerate(admissible))
    if style == "rich":
        system = RICH_SYSTEM_PROMPT
        user = RICH_USER_TEMPLATE.format(
            task=task, step=step + 1, max_steps=max_steps,
            history=format_history(history),
            failed=", ".join(failed) if failed else "(none)",
            observation=observation.strip(), actions=actions,
        )
    else:
        system = SYSTEM_PROMPT
        user = USER_TEMPLATE.format(
            task=task, observation=observation.strip(),
            history=format_history(history), actions=actions,
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
