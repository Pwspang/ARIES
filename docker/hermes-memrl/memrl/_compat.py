"""Helpers newer Hermes releases may ship in ``agent.memory_provider``.

The pinned Hermes image (v2026.5.29.2, Hermes 0.14.0) has none of them, and
neither does MemRL itself, so each one is imported when present and replaced
by a small local equivalent otherwise.
"""

from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass

try:  # pragma: no cover - depends on the installed Hermes version
    from agent.memory_provider import spawn_context_thread  # type: ignore[attr-defined]
except ImportError:

    def spawn_context_thread(fn, *args, name: str = "memrl") -> threading.Thread:
        """Run fn(*args) on a daemon thread inside a copy of the caller's context."""
        ctx = contextvars.copy_context()
        thread = threading.Thread(target=ctx.run, args=(fn, *args), name=name, daemon=True)
        thread.start()
        return thread


try:  # pragma: no cover - depends on the installed Hermes version
    from agent.memory_provider import RecallStatus  # type: ignore[attr-defined]
except ImportError:

    @dataclass(frozen=True)
    class RecallStatus:  # type: ignore[no-redef]
        glyph: str
        count: int
        label: str = ""


try:  # pragma: no cover - depends on the installed Hermes version
    from agent.memory_provider import is_trivial_prompt  # type: ignore[attr-defined]
except ImportError:
    _TRIVIAL = frozenset(
        {
            "hi", "hello", "hey", "thanks", "thank you", "thx", "ok", "okay",
            "yes", "no", "y", "n", "sure", "cool", "great", "bye", "continue",
        }
    )

    def is_trivial_prompt(query: str) -> bool:
        text = (query or "").strip().lower().rstrip(".!?")
        return len(text) < 4 or text in _TRIVIAL
