"""MemRL memory provider for Hermes.

Non-parametric runtime reinforcement learning over Intent-Experience-Utility
triplets (MemRL, https://github.com/MemTensor/MemRL, MIT License). The model
stays frozen; learning happens in the retrieved context:

- prefetch: Phase A keeps retrieval keys whose intent embedding has cosine
  >= delta (top k1); Phase B re-ranks the memories filed under them by
  (1 - lambda) * z(sim) + lambda * z(Q) and injects the top k2, with
  epsilon-greedy exploration.
- commit: the session reward r moves each injected memory's Q by
  Q <- Q + alpha * (r - Q). A successful session is stored as a new
  experience (an LLM-written script plus the trajectory, as in MemRL's
  proceduralization); a failed one as a "[PATTERN TO AVOID]" LLM reflection.
  A new memory whose task matched a retrieved key with similarity >=
  add_similarity joins that key, as in MemRL's dict_memory.

The reward comes from one of two sources (MemRLConfig.reward_source):
- "agent": the memrl_feedback tool, else a heuristic, applied at exit.
- "external": the benchmark's verdict. At exit the session is parked as
  pending; the host (ARIES) writes {task_id: reward} to memrl/rewards.json
  after evaluation, and the next session applies it before its first recall.
  This is how MemRL itself learns: from the environment's success signal.

The intent is the task prompt's <question> block when it has one (see
MemRLConfig.intent_tag), otherwise the whole prompt.

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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from agent.memory_provider import MemoryProvider

from ._compat import RecallStatus, is_trivial_prompt, spawn_context_thread
from .retrieval import ema, phase_a, phase_b
from .procedure import (
    PROMPT_MESSAGE_LIMIT, PROMPT_TRAJECTORY_LIMIT, REFLECTION_PROMPT, SCRIPT_PROMPT,
    STORED_MESSAGE_LIMIT, STORED_TRAJECTORY_LIMIT, ChatFn, build_experience, build_reflection,
    extract_intent, format_trajectory, hermes_chat, load_session_messages,
)
from .reward import heuristic_reward
from .store import Pending, Store

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FEEDBACK_TOOL = "memrl_feedback"

MEMORY_PROMPT = (
    "## MemRL experience memory\n"
    "Before a turn you may receive a <memrl-memory> block of experiences from "
    "similar past tasks, ranked by how useful they proved. Entries marked "
    "[PATTERN TO AVOID] record approaches that failed before; do not repeat "
    "them. Use the rest as hints, not as ground truth."
)
FEEDBACK_PROMPT = (
    f"\nBefore your final answer, call `{FEEDBACK_TOOL}` once with an honest "
    "reward in [-1, 1]: 1 if the task is verifiably solved, -1 if it failed, "
    "and a short note on what worked or went wrong."
)
REWARDS_FILE = "rewards.json"
FINALIZE_WAIT = 600.0

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
    """Defaults follow MemRL's BigCodeBench config (rl_bcb_config.yaml).

    delta and the similarity statistics are re-measured for this embedder:
    over the 124 SWE-Atlas QA questions, bge-small-en-v1.5 pairwise cosine has
    mean 0.679 and std 0.057, and delta = mean + 0.75 std, the same placement
    MemRL's thresholds have against its own statistics (BigCodeBench 0.38 at
    mean + 0.7 std, ALFWorld 0.62 at mean + 0.84 std).
    """

    delta: float = 0.72          # Phase A cosine threshold (MemRL sim_threshold)
    k1: int = 5                  # retrieval keys kept by Phase A (MemRL k_retrieve)
    k2: int = 5                  # memories injected (MemRL topk)
    lam: float = 0.5             # Phase B weight on Q (MemRL weight_q; weight_sim = 1 - lam)
    alpha: float = 0.3           # EMA step size
    epsilon: float = 0.1         # epsilon-greedy exploration
    q_init: float = 0.0          # Q of a new experience (MemRL q_init_pos)
    q_reflection: float = 0.0    # Q of a new reflection (MemRL q_init_neg)
    add_similarity: float = 0.9  # join a retrieved key at or above this similarity (MemRL add_similarity_threshold)
    sim_norm_mean: float = 0.679  # fixed similarity z-score statistics (MemRL sim_norm_mean/std)
    sim_norm_std: float = 0.0567
    script_temperature: float = 0.0  # MemRL uses the run's LLM temperature, 0 in its configs
    intent_tag: str = "question"  # keep only <intent_tag>…</intent_tag> as the intent; "" keeps the whole prompt
    reward_source: str = "agent"  # "agent" (memrl_feedback, else heuristic) or "external" (see module docstring)

    @classmethod
    def from_env(cls) -> "MemRLConfig":
        """Read overrides such as MEMRL_DELTA or MEMRL_K2 from the environment."""
        cfg = cls()
        for f in fields(cls):
            raw = os.environ.get(f"MEMRL_{f.name.upper()}")
            if raw is not None and (raw or f.name == "intent_tag"):
                setattr(cfg, f.name, type(getattr(cfg, f.name))(raw))
        return cfg


def _sentence_transformer_embedder() -> Callable[[List[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    def embed(texts: List[str]) -> np.ndarray:
        return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    return embed


class MemRLMemoryProvider(MemoryProvider):
    def __init__(self, config: Optional[MemRLConfig] = None,
                 embedder: Optional[Callable[[List[str]], np.ndarray]] = None,
                 chat: Optional[ChatFn] = None) -> None:
        self._config = config or MemRLConfig.from_env()
        self._embedder = embedder
        self._chat_override = chat
        self._chat: Optional[ChatFn] = chat
        self._home = Path()
        self._embedder_lock = threading.Lock()
        self._store: Optional[Store] = None
        self._session_id = ""
        self._write_enabled = True
        self._last_final = ""
        self._feedback_note = ""
        self._last_recall = 0
        self._task_id = ""
        self._finalizer: Optional[threading.Thread] = None
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
        self._home = Path(home)
        self._store = Store(self._home / "memrl" / "memrl.db")
        self._chat = self._chat_override or hermes_chat(self._home)
        if self._chat is None:
            logger.warning("MemRL found no chat model in config.yaml; experiences will lack a script")
        self._session_id = session_id
        self._write_enabled = kwargs.get("agent_context", "primary") == "primary"
        if not self._atexit_registered:
            atexit.register(self.shutdown)
            self._atexit_registered = True
        self._task_id = os.environ.get("MEMRL_TASK_ID", "")
        if self._external and not self._task_id:
            logger.warning("MemRL external rewards need MEMRL_TASK_ID; this session will stay pending")
        # Start loading the embedding model off the conversation thread;
        # prefetch waits for it (see there).
        self._spawn(self._embed, "")
        if self._external and self._write_enabled:
            # Apply the rewards the host recorded for earlier sessions before
            # this session's first recall (prefetch waits for it).
            self._finalizer = threading.Thread(target=self._apply_external_rewards, name="memrl-finalize", daemon=True)
            self._finalizer.start()

    @property
    def _external(self) -> bool:
        return self._config.reward_source == "external"

    def system_prompt_block(self) -> str:
        return MEMORY_PROMPT if self._external else MEMORY_PROMPT + FEEDBACK_PROMPT

    def _embed(self, text: str) -> np.ndarray:
        with self._embedder_lock:
            if self._embedder is None:
                self._embedder = _sentence_transformer_embedder()
        return np.asarray(self._embedder([text]), dtype=np.float32)[0]

    def _retrieve(self, query_vec: np.ndarray) -> Tuple[List[str], List[Tuple[str, float]]]:
        """Return (chosen memory ids, retrieved (key id, similarity) pairs)."""
        key_ids, matrix = self._store.keys()
        if not key_ids:
            return [], []
        cfg = self._config
        idx, sims = phase_a(query_vec, matrix, cfg.delta, cfg.k1)
        retrieved = [(key_ids[int(i)], float(s)) for i, s in zip(idx, sims)]
        key_sim = dict(retrieved)
        candidates = self._store.candidates([k for k, _ in retrieved])
        if not candidates:
            return [], retrieved
        picked = phase_b(
            np.array([key_sim[k] for _, k, _ in candidates]), np.array([q for _, _, q in candidates]),
            cfg.lam, cfg.k2, sim_mean=cfg.sim_norm_mean, sim_std=cfg.sim_norm_std, epsilon=cfg.epsilon,
        )
        return [candidates[p][0] for p in picked], retrieved

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._store is None or is_trivial_prompt(query):
            return ""
        sid = session_id or self._session_id
        try:
            session = self._store.session(sid)
            # The intent is both the memory key and the search query, as in
            # MemRL, so both use the same extracted text.
            intent = extract_intent(query, self._config.intent_tag)
            if not session.intent:
                self._store.update_session(sid, intent=intent)
            if self._finalizer is not None:
                self._finalizer.join(FINALIZE_WAIT)
            # Embedding here, even on a cold start, guarantees the model is
            # loaded before the one-shot exits: its commit runs from atexit,
            # where importing the model's dependencies fails ("can't register
            # atexit after shutdown").
            chosen, retrieved = self._retrieve(self._embed(intent))
            if not session.retrieved_keys and retrieved:
                self._store.update_session(sid, retrieved_keys=retrieved)
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
        return [] if self._external else [FEEDBACK_SCHEMA]

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
        """Close the session once: park it for an external reward, or apply its reward now."""
        store = self._store
        if store is None or not self._write_enabled:
            return
        try:
            session = store.session(session_id)
            if session.committed or not session.intent:
                return
            if not store.claim_commit(session_id):
                return
            trajectory = self._trajectory_messages(session_id)
            pending = Pending(
                session_id=session_id, task_id=self._task_id, intent=session.intent,
                embedding=self._embed(session.intent), active_ids=session.active_ids,
                retrieved_keys=session.retrieved_keys,
                prompt_trajectory=format_trajectory(trajectory, PROMPT_MESSAGE_LIMIT, PROMPT_TRAJECTORY_LIMIT),
                stored_trajectory=format_trajectory(trajectory, STORED_MESSAGE_LIMIT, STORED_TRAJECTORY_LIMIT),
            )
            if self._external:
                store.add_pending(pending)
                logger.info("MemRL parked session %s for task %s's reward", session_id, self._task_id)
                return
            reward = session.feedback_reward
            if reward is None:
                reward = heuristic_reward(messages, self._last_final)
            self._finalize(pending, reward)
        except Exception as e:
            logger.warning("MemRL commit failed: %s", e)

    def _apply_external_rewards(self) -> None:
        """Finalize every pending session whose task has a recorded reward.

        A null reward means the host could not judge the task (for example,
        evaluation was blocked), so that session is dropped, not learned from.
        """
        try:
            path = self._home / "memrl" / REWARDS_FILE
            rewards = json.loads(path.read_text()) if path.is_file() else {}
            for pending in self._store.pending():
                if pending.task_id not in rewards:
                    continue
                reward = rewards[pending.task_id]
                if reward is None:
                    self._store.delete_pending(pending.session_id)
                    logger.info("MemRL dropped session %s: task %s has no verdict", pending.session_id, pending.task_id)
                    continue
                self._finalize(pending, float(reward))
        except Exception as e:
            logger.warning("MemRL could not apply external rewards: %s", e)

    def _finalize(self, pending: Pending, reward: float) -> None:
        """Update the injected memories' Q toward the reward and store the new memory."""
        store = self._store
        reward = max(-1.0, min(1.0, float(reward)))
        memories = store.get_memories(pending.active_ids)
        store.set_q([(m.id, ema(m.q_value, reward, self._config.alpha)) for m in memories])
        # Join the best retrieved key if this task closely matched it (MemRL's
        # add_similarity_threshold), otherwise start a new key.
        known = set(store.memory_ids())
        matches = [(s, k) for k, s in pending.retrieved_keys if s >= self._config.add_similarity and k in known]
        key_id = max(matches)[1] if matches else None
        if reward >= 0:
            script = self._ask(SCRIPT_PROMPT.format(trajectory=pending.prompt_trajectory),
                               self._config.script_temperature)
            store.add_memory(pending.intent, build_experience(pending.intent, script, pending.stored_trajectory),
                             pending.embedding, self._config.q_init, "experience", key_id)
        else:
            reflection = self._ask(REFLECTION_PROMPT.format(task=pending.intent, trajectory=pending.prompt_trajectory), 0.3)
            reflection = reflection or self._feedback_note or f"session reward {reward:+.2f}"
            store.add_memory(pending.intent, build_reflection(pending.intent, reflection, pending.stored_trajectory),
                             pending.embedding, self._config.q_reflection, "reflection", key_id)
        store.delete_pending(pending.session_id)
        logger.info("MemRL finalized session %s: reward=%+.2f updated=%d", pending.session_id, reward, len(memories))

    def _trajectory_messages(self, session_id: str) -> List[Dict[str, str]]:
        """The full session from Hermes's state.db, else the turn buffer."""
        try:
            messages = load_session_messages(self._home / "state.db", session_id)
        except Exception as e:
            logger.warning("MemRL could not read the Hermes session store: %s", e)
            messages = []
        if messages:
            return messages
        steps = self._store.steps(session_id) if self._store else []
        messages = [m for u, a in steps for m in ({"role": "user", "content": u, "tool_calls": "", "tool_name": ""},
                                                  {"role": "assistant", "content": a, "tool_calls": "", "tool_name": ""})]
        return messages or [{"role": "assistant", "content": self._last_final, "tool_calls": "", "tool_name": ""}]

    def _ask(self, prompt: str, temperature: float) -> str:
        """One model call; a failure costs the script, never the memory."""
        if self._chat is None:
            return ""
        try:
            return self._chat([{"role": "user", "content": prompt}], temperature)
        except Exception as e:
            logger.warning("MemRL model call failed: %s", e)
            return ""


def register(ctx) -> None:
    """Register the MemRL memory provider with the Hermes plugin system."""
    ctx.register_memory_provider(MemRLMemoryProvider())
