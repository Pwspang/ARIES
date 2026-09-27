"""Intent extraction and model-built experiences.

The experience layout follows MemRL's proceduralization build strategy
(https://github.com/MemTensor/MemRL, MIT License): an LLM turns the
trajectory into a short high-level script, stored together with the
trajectory (memrl/service/builders.py ProceduralizationBuilder, prompt from
memrl/providers/llm.py generate_script). Failed sessions get an LLM
reflection (memrl/service/updater.py AdjustmentUpdater._generate_reflection,
append mode).

The trajectory is read from Hermes's own session store, because in one-shot
mode the provider only ever receives the final answer.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Called with (messages, temperature) and returns the reply text.
ChatFn = Callable[[List[Dict[str, str]], float], str]

LLM_TIMEOUT = 120.0
LLM_MAX_TOKENS = 4096
# What the script/reflection model sees, and what a stored memory keeps.
PROMPT_MESSAGE_LIMIT, PROMPT_TRAJECTORY_LIMIT = 2000, 60000
STORED_MESSAGE_LIMIT, STORED_TRAJECTORY_LIMIT = 400, 6000

SCRIPT_PROMPT = """
Analyze the following detailed task trajectory and create a concise,
high-level script that captures the essential steps and decision points.

The script should be:
1. Generic enough to apply to similar tasks
2. Specific enough to provide useful guidance
3. 3-5 high-level steps maximum
4. Focus on the strategy and key decisions, not detailed actions

Trajectory:
{trajectory}

High-level script:"""

REFLECTION_PROMPT = """
Task: {task}

Failed trajectory:
{trajectory}

This task failed. Analyze what went wrong and suggest improvements for future similar tasks.
Focus on:
1. Incorrect assumptions
2. Steps to improve
3. What to avoid next time

Provide a brief reflection:
"""


def extract_intent(prompt: str, tag: str) -> str:
    """Return the text inside <tag>…</tag>, or the whole prompt without one."""
    if tag:
        match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", prompt, re.DOTALL)
        if match and match.group(1).strip():
            return match.group(1).strip()
    return prompt


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " …(truncated)"


def load_session_messages(db_path: Path, session_id: str) -> List[Dict[str, str]]:
    """Read one session's messages from Hermes's state.db, oldest first."""
    if not db_path.is_file():
        return []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    try:
        rows = conn.execute(
            "SELECT role, content, tool_calls, tool_name FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    return [{"role": r[0] or "", "content": r[1] or "", "tool_calls": r[2] or "", "tool_name": r[3] or ""} for r in rows]


def format_trajectory(messages: List[Dict[str, str]], message_limit: int, total_limit: int) -> str:
    """Serialize messages as "<role>: <content>" lines, like MemRL's runners.

    Assistant tool calls are rendered as `call name(arguments)`. When the whole
    trajectory exceeds total_limit, its middle is dropped so both the approach
    and the conclusion survive.
    """
    lines = []
    for m in messages:
        role, content = m["role"], _clip(m["content"].strip(), message_limit)
        if role == "assistant":
            calls = []
            try:
                for call in json.loads(m["tool_calls"] or "[]"):
                    fn = call.get("function") or {}
                    calls.append(f"call {fn.get('name', '?')}({_clip(str(fn.get('arguments', '')), message_limit)})")
            except (ValueError, TypeError, AttributeError):
                pass
            body = "\n".join(x for x in [content, *calls] if x)
            if body:
                lines.append(f"assistant: {body}")
        elif role == "tool":
            lines.append(f"tool {m['tool_name']}: {content}" if m["tool_name"] else f"tool: {content}")
        elif content:
            lines.append(f"{role}: {content}")
    text = "\n".join(lines)
    if len(text) <= total_limit:
        return text
    half = total_limit // 2
    return text[:half] + "\n…(middle of trajectory omitted)…\n" + text[-half:]


def build_experience(task: str, script: str, trajectory: str) -> str:
    body = f"SCRIPT:\n{script}\n\nTRAJECTORY:\n{trajectory}" if script else f"TRAJECTORY:\n{trajectory}"
    return f"Task: {task}\n\n{body}"


def build_reflection(task: str, reflection: str, trajectory: str) -> str:
    return (
        "[PATTERN TO AVOID]\n"
        f"TASK REFLECTION:\nTask: {task}\n\n"
        f"What went wrong:\n{reflection}\n\n"
        f"Failed approach:\n{trajectory}\n"
    )


def hermes_chat(hermes_home: Path) -> Optional[ChatFn]:
    """Build a chat function for Hermes's own main model.

    ARIES renders an OpenAI-compatible model section (base_url, default,
    api_key as a ${NAME} reference that the wrapper exports), so the script
    and reflection calls reuse the task model and credential without storing
    the key anywhere else.
    """
    import yaml

    try:
        config = yaml.safe_load((hermes_home / "config.yaml").read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None
    model = config.get("model") or {}
    base_url, name = str(model.get("base_url") or ""), str(model.get("default") or "")
    api_key = os.path.expandvars(str(model.get("api_key") or ""))
    if not base_url or not name or not api_key or api_key.startswith("${"):
        return None
    url = base_url.rstrip("/") + "/chat/completions"

    def chat(messages: List[Dict[str, str]], temperature: float) -> str:
        body = json.dumps({"model": name, "messages": messages, "temperature": temperature,
                           "max_tokens": LLM_MAX_TOKENS}).encode()
        request = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(request, timeout=LLM_TIMEOUT) as response:
            reply = json.load(response)
        return str(reply["choices"][0]["message"].get("content") or "").strip()

    return chat
