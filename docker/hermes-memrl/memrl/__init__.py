"""MemRL memory provider for Hermes.

Non-parametric runtime reinforcement learning over Intent-Experience-Utility
triplets (MemRL, https://github.com/MemTensor/MemRL, MIT License). The model
stays frozen; learning happens in the retrieved context:

- prefetch: Phase A keeps memories whose intent embedding has cosine >= delta
  (top k1); Phase B re-ranks them by (1 - lambda) * z(sim) + lambda * z(Q)
  and injects the top k2.
- commit: the session reward r (the memrl_feedback tool, else a heuristic)
  moves each injected memory's Q by Q <- Q + alpha * (r - Q). A successful
  session is stored as a new experience; a failed one is stored as a
  "[PATTERN TO AVOID]" reflection with Q = 0.5 so it is retrieved at once.

Hermes one-shot mode (`hermes -z`) exits without calling on_session_end or
shutdown, so the commit also runs from an atexit handler.
"""

from __future__ import annotations

import atexit
import importlib.util
import json
import logging
import os
import threading
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from agent.memory_provider import MemoryProvider

from ._compat import RecallStatus, is_trivial_prompt, spawn_context_thread
from .retrieval import ema, phase_a, phase_b
from .reward import failure_reflection, heuristic_reward
from .store import Store

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FEEDBACK_TOOL = "memrl_feedback"
EXPERIENCE_LIMIT = 2000

SYSTEM_PROMPT_BLOCK = (
    "## MemRL experience memory\n"
    "Before a turn you may receive a <memrl-memory> block of experiences from "
    "similar past tasks, ranked by how useful they proved. Entries marked "
    "[PATTERN TO AVOID] record approaches that failed before; do not repeat "
    "them. Use the rest as hints, not as ground truth.\n"
    f"Before your final answer, call `{FEEDBACK_TOOL}` once with an honest "
    "reward in [-1, 1]: 1 if the task is verifiably solved, -1 if it failed, "
    "and a short note on what worked or went wrong."
)

FEEDBACK_SCHEMA = {
    "name": FEEDBACK_TOOL,
    "description": (
        "Report how well the current task went so MemRL can learn which "
        "remembered experiences help. Call once, just before the final answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reward": {
                "type": "number",
                "minimum": -1,
                "maximum": 1,
                "description": "1 = verifiably solved, 0 = unclear, -1 = failed.",
            },
            "note": {"type": "string", "description": "What worked, or what went wrong."},
        },
        "required": ["reward"],
    },
}

_REQUIRED_MODULES = ("numpy", "sentence_transformers")


@dataclass
class MemRLConfig:
    delta: float = 0.38   # Phase A cosine threshold
    k1: int = 15          # Phase A candidate cap
    k2: int = 3           # memories injected per turn
    lam: float = 0.5      # Phase B weight on Q
    alpha: float = 0.3    # EMA step size
    epsilon: float = 0.0  # MemRL epsilon-greedy exploration (off by default)
    q_init: float = 0.0   # Q of a new successful experience (MemRL q_init_pos)
    q_reflection: float = 0.5  # Q of a new failure reflection

    @classmethod
    def from_env(cls) -> "MemRLConfig":
        """Read overrides such as MEMRL_DELTA or MEMRL_K2 from the environment."""
        cfg = cls()
        for f in fields(cls):
            raw = os.environ.get(f"MEMRL_{f.name.upper()}")
            if raw:
                setattr(cfg, f.name, type(getattr(cfg, f.name))(raw))
        return cfg


def _sentence_transformer_embedder() -> Callable[[List[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    def embed(texts: List[str]) -> np.ndarray:
        return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    return embed


class MemRLMemoryProvider(MemoryProvider):
    def __init__(self, config: Optional[MemRLConfig] = None,
                 embedder: Optional[Callable[[List[str]], np.ndarray]] = None) -> None:
        self._config = config or MemRLConfig.from_env()
        self._embedder = embedder
        self._embedder_lock = threading.Lock()
        self._store: Optional[Store] = None
        self._session_id = ""
        self._write_enabled = True
        self._last_final = ""
        self._feedback_note = ""
        self._last_recall = 0
        self._pending: List[threading.Thread] = []
        self._pending_lock = threading.Lock()
        self._atexit_registered = False

    @property
    def name(self) -> str:
        return "memrl"

    # -- availability --------------------------------------------------------

    def _missing(self) -> List[str]:
        required = ("numpy",) if self._embedder is not None else _REQUIRED_MODULES
        return [m for m in required if importlib.util.find_spec(m) is None]

    def is_available(self) -> bool:
        return not self._missing()

    def unavailable_reason(self) -> str:
        missing = self._missing()
        return f"missing Python packages: {', '.join(missing)}" if missing else ""

    # -- lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        home = kwargs.get("hermes_home") or os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        self._store = Store(Path(home) / "memrl" / "memrl.db")
        self._session_id = session_id
        self._write_enabled = kwargs.get("agent_context", "primary") == "primary"
        if not self._atexit_registered:
            atexit.register(self.shutdown)
            self._atexit_registered = True

    def system_prompt_block(self) -> str:
        return SYSTEM_PROMPT_BLOCK

    def _embed(self, text: str) -> np.ndarray:
        with self._embedder_lock:
            if self._embedder is None:
                self._embedder = _sentence_transformer_embedder()
        return np.asarray(self._embedder([text]), dtype=np.float32)[0]

    def _retrieve(self, query: str) -> List[str]:
        ids, matrix, q_values = self._store.embeddings()
        if not ids:
            return []
        cfg = self._config
        idx, sims = phase_a(self._embed(query), matrix, cfg.delta, cfg.k1)
        picked = phase_b(sims, q_values[idx], cfg.lam, cfg.k2, epsilon=cfg.epsilon)
        return [ids[int(idx[p])] for p in picked]

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._store is None or is_trivial_prompt(query):
            return ""
        sid = session_id or self._session_id
        try:
            session = self._store.session(sid)
            if not session.intent:
                self._store.update_session(sid, intent=query)
            chosen = self._retrieve(query)
        except Exception as e:
            logger.warning("MemRL prefetch failed: %s", e)
            return ""
        self._last_recall = len(chosen)
        if not chosen:
            return ""
        active = session.active_ids + [i for i in chosen if i not in session.active_ids]
        self._store.update_session(sid, active_ids=active)
        self._store.bump_usage(chosen)
        parts = []
        for n, mem in enumerate(self._store.get_memories(chosen), 1):
            parts.append(f"### Memory {n} (utility {mem.q_value:+.2f})\nPast task: {mem.intent}\n{mem.experience}")
        return "<memrl-memory>\n" + "\n\n".join(parts) + "\n</memrl-memory>"

    def recall_status(self) -> Optional[RecallStatus]:
        return RecallStatus("⚡", self._last_recall)

    def _spawn(self, fn: Callable[..., None], *args: Any) -> None:
        thread = spawn_context_thread(fn, *args, name="memrl")
        with self._pending_lock:
            self._pending = [t for t in self._pending if t.is_alive()] + [thread]

    def _join_pending(self, timeout: float = 30.0) -> None:
        with self._pending_lock:
            pending, self._pending = self._pending, []
        for t in pending:
            if t is not threading.current_thread():
                t.join(timeout)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if self._store is None or not self._write_enabled:
            return
        self._last_final = assistant_content or ""
        self._spawn(self._store.add_step, session_id or self._session_id, user_content or "", assistant_content or "")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FEEDBACK_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if tool_name != FEEDBACK_TOOL:
            raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")
        try:
            reward = float(args.get("reward"))
        except (TypeError, ValueError):
            return json.dumps({"error": "reward must be a number in [-1, 1]"})
        if not -1.0 <= reward <= 1.0:
            return json.dumps({"error": "reward must be a number in [-1, 1]"})
        if self._store is None:
            return json.dumps({"error": "MemRL is not initialized"})
        sid = kwargs.get("session_id") or self._session_id
        self._store.update_session(sid, feedback_reward=reward)
        self._feedback_note = str(args.get("note") or "")
        return json.dumps({"recorded": True, "reward": reward})

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        self._join_pending()
        return ""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        self._spawn(self._commit, self._session_id, list(messages or []))

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "",
                          reset: bool = False, **kwargs: Any) -> None:
        if reset and self._store is not None:
            self._join_pending()
            self._commit(self._session_id, None)
        self._session_id = new_session_id

    def shutdown(self) -> None:
        if self._store is None:
            return
        self._join_pending()
        self._commit(self._session_id, None)
        self._store.close()
        self._store = None

    # -- utility update --------------------------------------------------------

    def _commit(self, session_id: str, messages: Optional[List[Dict[str, Any]]]) -> None:
        """Apply the session reward once: EMA on injected memories, then store a new triplet."""
        store = self._store
        if store is None or not self._write_enabled:
            return
        try:
            session = store.session(session_id)
            if session.committed or not session.intent:
                return
            if not store.claim_commit(session_id):
                return
            reward = session.feedback_reward
            if reward is None:
                reward = heuristic_reward(messages, self._last_final)
            reward = max(-1.0, min(1.0, float(reward)))
            memories = store.get_memories(session.active_ids)
            store.set_q([(m.id, ema(m.q_value, reward, self._config.alpha)) for m in memories])

            trajectory = "\n".join(
                f"User: {u}\nAssistant: {a}" for u, a in store.steps(session_id)
            ) or self._last_final
            embedding = self._embed(session.intent)
            if reward >= 0:
                store.add_memory(session.intent, trajectory[:EXPERIENCE_LIMIT], embedding,
                                 self._config.q_init, "experience")
            else:
                reason = self._feedback_note or f"session reward {reward:+.2f}"
                store.add_memory(session.intent, failure_reflection(session.intent, trajectory, reason),
                                 embedding, self._config.q_reflection, "reflection")
            logger.info("MemRL committed session %s: reward=%+.2f updated=%d", session_id, reward, len(memories))
        except Exception as e:
            logger.warning("MemRL commit failed: %s", e)


def register(ctx) -> None:
    """Register the MemRL memory provider with the Hermes plugin system."""
    ctx.register_memory_provider(MemRLMemoryProvider())
