"""MemRL two-phase retrieval and utility update math.

Ported from MemRL (https://github.com/MemTensor/MemRL, MIT License,
Copyright (c) 2026 jiaqian), commit c1b322c:

- phase_a/phase_b follow ``MemoryService.retrieve_query``,
  ``_normalize_similarity`` and ``_normalize_q`` in
  memrl/service/memory_service.py: similarity threshold filter over the
  retrieval keys, top-k keys, similarity z-scored with fixed corpus
  statistics, Q z-scored over the candidates and clamped to +/-3, hybrid
  score, and epsilon-greedy selection.
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


def phase_b(
    sims: np.ndarray,
    q_values: np.ndarray,
    lam: float,
    k2: int,
    *,
    sim_mean: float,
    sim_std: float,
    epsilon: float = 0.0,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Return positions into the candidate arrays, ranked by the hybrid score.

    As in MemRL's retrieve_query: similarity is z-scored with fixed corpus
    statistics (sim_mean, sim_std), Q with the candidates' own mean and
    population std (1.0 for a single candidate) and clamped to +/-3. Score =
    (1 - lam) * z(sim) + lam * z(Q). With probability epsilon a random k2
    subset of all candidates is returned instead (epsilon-greedy).
    """
    n = len(sims)
    if n == 0 or k2 <= 0:
        return []
    sim_z = (np.asarray(sims, dtype=np.float64) - sim_mean) / (sim_std if sim_std > 1e-9 else 1.0)
    q = np.asarray(q_values, dtype=np.float64)
    q_std = float(q.std()) if n > 1 else 1.0
    q_z = np.clip((q - q.mean()) / (q_std if q_std > 1e-9 else 1.0), -Q_Z_CLAMP, Q_Z_CLAMP)
    score = (1.0 - lam) * sim_z + lam * q_z
    ranked = [int(i) for i in np.argsort(-score, kind="stable")]
    k = min(k2, n)
    chooser = rng or random
    if epsilon > 0 and chooser.random() < epsilon:
        return chooser.sample(ranked, k)
    return ranked[:k]


def ema(q: float, reward: float, alpha: float) -> float:
    """Single-step TD update toward a reward clipped to [-1, 1]."""
    r = min(1.0, max(-1.0, float(reward)))
    return float(q) + float(alpha) * (r - float(q))
