"""Session reward heuristic and failure reflection text.

The reflection layout follows MemRL's ``AdjustmentUpdater._prepare_append_adjust``
(memrl/service/updater.py, https://github.com/MemTensor/MemRL, MIT License),
but is filled from a template instead of an extra LLM call.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

TRAJECTORY_LIMIT = 1500

_FAILURE_PHRASES = (
    "i couldn't", "i could not", "i was unable", "i am unable", "i'm unable",
    "unable to complete", "failed to", "i can't", "i cannot", "not able to",
)


def _tool_failed(content: Any) -> Optional[bool]:
    """Classify one tool result: True failed, False succeeded, None unknown."""
    if isinstance(content, list):  # multi-part content
        content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    if not isinstance(content, str):
        return None
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    if "exit_code" in data:
        try:
            return int(data["exit_code"]) != 0
        except (TypeError, ValueError):
            return None
    if data.get("error"):
        return True
    if data.get("success") is False:
        return True
    return False


def _final_text_reward(final_text: str) -> float:
    text = (final_text or "").strip().lower()
    if not text or any(p in text for p in _FAILURE_PHRASES):
        return -0.5
    return 0.25


def heuristic_reward(messages: Optional[List[Dict[str, Any]]], final_text: str) -> float:
    """Estimate a reward in [-1, 1] without calling a model.

    With a transcript, the reward is 1 - 2 * (failed tool results / classified
    tool results). Without one (Hermes one-shot exits without on_session_end),
    only the final answer text is available and the signal is weak.
    """
    if messages:
        outcomes = [
            _tool_failed(m.get("content"))
            for m in messages
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
        known = [o for o in outcomes if o is not None]
        if known:
            return max(-1.0, min(1.0, 1.0 - 2.0 * sum(known) / len(known)))
        last = next(
            (m.get("content") for m in reversed(messages)
             if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str)),
            final_text,
        )
        return _final_text_reward(last or "")
    return _final_text_reward(final_text)


def failure_reflection(intent: str, trajectory: str, reason: str) -> str:
    if len(trajectory) > TRAJECTORY_LIMIT:
        trajectory = trajectory[:TRAJECTORY_LIMIT] + "\n…(truncated)"
    return (
        "[PATTERN TO AVOID]\n"
        f"Task: {intent}\n"
        f"What went wrong: {reason}\n"
        f"Failed approach:\n{trajectory}\n"
    )
