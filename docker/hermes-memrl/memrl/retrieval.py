"""MemRL two-phase retrieval and utility update math.

Ported from MemRL (https://github.com/MemTensor/MemRL, MIT License,
Copyright (c) 2026 jiaqian), commit c1b322c:

- phase_a/phase_b follow ``MemoryService.retrieve_query`` and ``_normalize_q``
  in memrl/service/memory_service.py: similarity threshold filter, top-k
  candidates, per-candidate z-scored Q clamped to +/-3, hybrid score, and
  optional epsilon-greedy selection. Unlike MemRL, similarity is z-scored
  over the candidate set instead of fixed corpus statistics.
- ema follows ``QValueUpdater.update`` in memrl/service/value_driven.py with
  gamma = 0 (single-step): Q <- Q + alpha * (r - Q).
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np

Q_Z_CLAMP = 3.0


def phase_a(query_vec: np.ndarray, matrix: np.ndarray, delta: float, k1: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, similarities) of rows with cosine >= delta, best k1 first.

    Rows of matrix and query_vec are unit-normalized, so the dot product is
    the cosine similarity.
    """
    if matrix.size == 0 or k1 <= 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=np.float32)
    sims = matrix @ query_vec
    keep = np.flatnonzero(sims >= delta)
    order = keep[np.argsort(-sims[keep], kind="stable")][:k1]
    return order, sims[order]


def _zscore(x: np.ndarray, eps: float) -> np.ndarray:
    return (x - x.mean()) / (x.std() + eps)


def phase_b(
    sims: np.ndarray,
    q_values: np.ndarray,
    lam: float,
    k2: int,
    *,
    eps: float = 1e-6,
    epsilon: float = 0.0,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Return positions into the candidate arrays, ranked by the hybrid score.

    Score = (1 - lam) * z(sim) + lam * clamp(z(Q), +/-3). With epsilon > 0 a
    random k2 subset is returned instead, as in MemRL's epsilon-greedy.
    """
    n = len(sims)
    if n == 0 or k2 <= 0:
        return []
    sim_z = _zscore(np.asarray(sims, dtype=np.float64), eps)
    q_z = np.clip(_zscore(np.asarray(q_values, dtype=np.float64), eps), -Q_Z_CLAMP, Q_Z_CLAMP)
    score = (1.0 - lam) * sim_z + lam * q_z
    ranked = [int(i) for i in np.argsort(-score, kind="stable")]
    k = min(k2, n)
    if epsilon > 0 and (rng or random).random() < epsilon:
        return (rng or random).sample(ranked, k)
    return ranked[:k]


def ema(q: float, reward: float, alpha: float) -> float:
    """Single-step TD update toward a reward clipped to [-1, 1]."""
    r = min(1.0, max(-1.0, float(reward)))
    return float(q) + float(alpha) * (r - float(q))
