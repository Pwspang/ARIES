"""SQLite storage for MemRL triplets, the turn buffer, and session state."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# A memory's key_id names the retrieval key it is filed under: the memory
# whose intent embedding stands for the group. As in MemRL's dict_memory, a
# new memory whose task closely matches a retrieved key joins that key
# instead of creating one, and retrieval scores keys, not memories.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    key_id      TEXT NOT NULL,
    intent      TEXT NOT NULL,
    experience  TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    q_value     REAL NOT NULL,
    usage_count INTEGER NOT NULL DEFAULT 0,
    kind        TEXT NOT NULL DEFAULT 'experience',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    session_id  TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    user        TEXT NOT NULL,
    assistant   TEXT NOT NULL,
    created_at  REAL NOT NULL,
    committed   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, seq)
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    intent          TEXT NOT NULL DEFAULT '',
    active_ids      TEXT NOT NULL DEFAULT '[]',
    retrieved_keys  TEXT NOT NULL DEFAULT '[]',
    feedback_reward REAL,
    committed       INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS pending (
    session_id        TEXT PRIMARY KEY,
    task_id           TEXT NOT NULL,
    intent            TEXT NOT NULL,
    embedding         BLOB NOT NULL,
    active_ids        TEXT NOT NULL,
    retrieved_keys    TEXT NOT NULL,
    prompt_trajectory TEXT NOT NULL,
    stored_trajectory TEXT NOT NULL,
    created_at        REAL NOT NULL
);
"""


@dataclass
class Memory:
    id: str
    key_id: str
    intent: str
    experience: str
    q_value: float
    usage_count: int
    kind: str


@dataclass
class Session:
    intent: str
    active_ids: List[str]
    retrieved_keys: List[Tuple[str, float]]
    feedback_reward: Optional[float]
    committed: bool


@dataclass
class Pending:
    """A finished session waiting for its reward (see MemRLConfig.reward_source)."""

    session_id: str
    task_id: str
    intent: str
    embedding: np.ndarray
    active_ids: List[str]
    retrieved_keys: List[Tuple[str, float]]
    prompt_trajectory: str
    stored_trajectory: str


def _blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


class Store:
    """One SQLite file in the default rollback-journal mode.

    WAL is deliberately not used: ARIES copies the single database file out of
    the container after Hermes exits, and a WAL sidecar would be left behind.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.commit()
                self._conn.close()
                self._conn = None

    # -- memories ----------------------------------------------------------

    def add_memory(self, intent: str, experience: str, embedding: np.ndarray, q_value: float, kind: str,
                   key_id: Optional[str] = None) -> str:
        mid = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (id, key_id, intent, experience, embedding, q_value, usage_count, kind,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)",
                (mid, key_id or mid, intent, experience, _blob(embedding), float(q_value), kind, now, now),
            )
            self._conn.commit()
        return mid

    def keys(self) -> Tuple[List[str], np.ndarray]:
        """Return (key ids, key embedding matrix) for every retrieval key."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, embedding FROM memories WHERE id = key_id ORDER BY created_at"
            ).fetchall()
        if not rows:
            return [], np.empty((0, 0), dtype=np.float32)
        return [r[0] for r in rows], np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])

    def candidates(self, key_ids: List[str]) -> List[Tuple[str, str, float]]:
        """Return (memory id, key id, Q) for every memory filed under key_ids."""
        if not key_ids:
            return []
        marks = ",".join("?" for _ in key_ids)
        with self._lock:
            return self._conn.execute(
                f"SELECT id, key_id, q_value FROM memories WHERE key_id IN ({marks}) ORDER BY created_at",
                key_ids,
            ).fetchall()

    def memory_ids(self) -> List[str]:
        with self._lock:
            return [r[0] for r in self._conn.execute("SELECT id FROM memories ORDER BY created_at")]

    def get_memories(self, ids: List[str]) -> List[Memory]:
        if not ids:
            return []
        marks = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, key_id, intent, experience, q_value, usage_count, kind FROM memories WHERE id IN ({marks})",
                ids,
            ).fetchall()
        by_id = {r[0]: Memory(*r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def bump_usage(self, ids: List[str]) -> None:
        now = time.time()
        with self._lock:
            self._conn.executemany(
                "UPDATE memories SET usage_count = usage_count + 1, updated_at = ? WHERE id = ?",
                [(now, i) for i in ids],
            )
            self._conn.commit()

    def set_q(self, updates: List[Tuple[str, float]]) -> None:
        now = time.time()
        with self._lock:
            self._conn.executemany(
                "UPDATE memories SET q_value = ?, updated_at = ? WHERE id = ?",
                [(q, now, i) for i, q in updates],
            )
            self._conn.commit()

    # -- turn buffer -------------------------------------------------------

    def add_step(self, session_id: str, user: str, assistant: str) -> None:
        with self._lock:
            (seq,) = self._conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM steps WHERE session_id = ?", (session_id,)
            ).fetchone()
            self._conn.execute(
                "INSERT INTO steps (session_id, seq, user, assistant, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, seq, user, assistant, time.time()),
            )
            self._conn.commit()

    def steps(self, session_id: str) -> List[Tuple[str, str]]:
        with self._lock:
            return self._conn.execute(
                "SELECT user, assistant FROM steps WHERE session_id = ? ORDER BY seq", (session_id,)
            ).fetchall()

    # -- sessions ----------------------------------------------------------

    def session(self, session_id: str) -> Session:
        with self._lock:
            self._conn.execute("INSERT OR IGNORE INTO sessions (session_id) VALUES (?)", (session_id,))
            row = self._conn.execute(
                "SELECT intent, active_ids, retrieved_keys, feedback_reward, committed FROM sessions"
                " WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            self._conn.commit()
        return Session(row[0], json.loads(row[1]), [tuple(k) for k in json.loads(row[2])], row[3], bool(row[4]))

    def update_session(self, session_id: str, *, intent: Optional[str] = None,
                       active_ids: Optional[List[str]] = None,
                       retrieved_keys: Optional[List[Tuple[str, float]]] = None,
                       feedback_reward: Optional[float] = None) -> None:
        self.session(session_id)
        updates: Dict[str, object] = {}
        if intent is not None:
            updates["intent"] = intent
        if active_ids is not None:
            updates["active_ids"] = json.dumps(active_ids)
        if retrieved_keys is not None:
            updates["retrieved_keys"] = json.dumps([[k, float(s)] for k, s in retrieved_keys])
        if feedback_reward is not None:
            updates["feedback_reward"] = float(feedback_reward)
        with self._lock:
            for column, value in updates.items():  # column names are the fixed literals above
                self._conn.execute(f"UPDATE sessions SET {column} = ? WHERE session_id = ?", (value, session_id))
            self._conn.commit()

    def claim_commit(self, session_id: str) -> bool:
        """Atomically mark the session committed; False if it already was."""
        self.session(session_id)
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET committed = 1 WHERE session_id = ? AND committed = 0", (session_id,)
            )
            self._conn.execute("UPDATE steps SET committed = 1 WHERE session_id = ?", (session_id,))
            self._conn.commit()
            return cur.rowcount == 1

    # -- sessions awaiting an external reward ------------------------------

    def add_pending(self, pending: Pending) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO pending (session_id, task_id, intent, embedding, active_ids, retrieved_keys,"
                " prompt_trajectory, stored_trajectory, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pending.session_id, pending.task_id, pending.intent, _blob(pending.embedding),
                 json.dumps(pending.active_ids), json.dumps([[k, float(s)] for k, s in pending.retrieved_keys]),
                 pending.prompt_trajectory, pending.stored_trajectory, time.time()),
            )
            self._conn.commit()

    def pending(self) -> List[Pending]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, task_id, intent, embedding, active_ids, retrieved_keys, prompt_trajectory,"
                " stored_trajectory FROM pending ORDER BY created_at"
            ).fetchall()
        return [Pending(r[0], r[1], r[2], np.frombuffer(r[3], dtype=np.float32), json.loads(r[4]),
                        [tuple(k) for k in json.loads(r[5])], r[6], r[7]) for r in rows]

    def delete_pending(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM pending WHERE session_id = ?", (session_id,))
            self._conn.commit()
