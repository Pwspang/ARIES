"""SQLite storage for MemRL triplets, the turn buffer, and session state."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
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
    feedback_reward REAL,
    committed       INTEGER NOT NULL DEFAULT 0
);
"""


@dataclass
class Memory:
    id: str
    intent: str
    experience: str
    q_value: float
    usage_count: int
    kind: str


@dataclass
class Session:
    intent: str
    active_ids: List[str]
    feedback_reward: Optional[float]
    committed: bool


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

    def add_memory(self, intent: str, experience: str, embedding: np.ndarray, q_value: float, kind: str) -> str:
        mid = uuid.uuid4().hex
        now = time.time()
        blob = np.asarray(embedding, dtype=np.float32).tobytes()
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (id, intent, experience, embedding, q_value, usage_count, kind, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)",
                (mid, intent, experience, blob, float(q_value), kind, now, now),
            )
            self._conn.commit()
        return mid

    def embeddings(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Return (ids, embedding matrix, q values) for every memory."""
        with self._lock:
            rows = self._conn.execute("SELECT id, embedding, q_value FROM memories ORDER BY created_at").fetchall()
        if not rows:
            return [], np.empty((0, 0), dtype=np.float32), np.empty(0)
        ids = [r[0] for r in rows]
        matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        return ids, matrix, np.array([r[2] for r in rows], dtype=np.float64)

    def get_memories(self, ids: List[str]) -> List[Memory]:
        if not ids:
            return []
        marks = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, intent, experience, q_value, usage_count, kind FROM memories WHERE id IN ({marks})",
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
                "SELECT intent, active_ids, feedback_reward, committed FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            self._conn.commit()
        return Session(row[0], json.loads(row[1]), row[2], bool(row[3]))

    def update_session(self, session_id: str, *, intent: Optional[str] = None,
                       active_ids: Optional[List[str]] = None,
                       feedback_reward: Optional[float] = None) -> None:
        self.session(session_id)
        with self._lock:
            if intent is not None:
                self._conn.execute("UPDATE sessions SET intent = ? WHERE session_id = ?", (intent, session_id))
            if active_ids is not None:
                self._conn.execute(
                    "UPDATE sessions SET active_ids = ? WHERE session_id = ?", (json.dumps(active_ids), session_id)
                )
            if feedback_reward is not None:
                self._conn.execute(
                    "UPDATE sessions SET feedback_reward = ? WHERE session_id = ?", (float(feedback_reward), session_id)
                )
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
