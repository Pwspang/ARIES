"""Run the plugin tests with or without Hermes installed.

Inside the derived image the real ``agent.memory_provider`` is importable.
Elsewhere a minimal stand-in ABC is installed so the plugin can be imported.
"""

import abc
import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import agent.memory_provider  # noqa: F401
except ImportError:
    class MemoryProvider(abc.ABC):
        @property
        @abc.abstractmethod
        def name(self): ...

        @abc.abstractmethod
        def is_available(self): ...

        @abc.abstractmethod
        def initialize(self, session_id, **kwargs): ...

        @abc.abstractmethod
        def get_tool_schemas(self): ...

    agent_pkg = types.ModuleType("agent")
    agent_pkg.__path__ = []
    mp = types.ModuleType("agent.memory_provider")
    mp.MemoryProvider = MemoryProvider
    agent_pkg.memory_provider = mp
    sys.modules["agent"] = agent_pkg
    sys.modules["agent.memory_provider"] = mp


class FakeEmbedder:
    """Deterministic bag-of-words embedder with optional pinned vectors."""

    dim = 64

    def __init__(self):
        self.pinned = {}

    def _vec(self, text):
        if text in self.pinned:
            v = np.asarray(self.pinned[text], dtype=np.float32)
        else:
            v = np.zeros(self.dim, dtype=np.float32)
            for word in text.lower().split():
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def __call__(self, texts):
        return np.stack([self._vec(t) for t in texts])


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def make_provider(tmp_path, embedder):
    from memrl import MemRLConfig, MemRLMemoryProvider

    made = []

    def make(session_id="s1", chat=None, **cfg):
        # Deterministic retrieval, and the threshold these fixtures were drawn for.
        cfg.setdefault("epsilon", 0.0)
        cfg.setdefault("delta", 0.38)
        p = MemRLMemoryProvider(config=MemRLConfig(**cfg), embedder=embedder, chat=chat)
        p.initialize(session_id, hermes_home=str(tmp_path), platform="cli")
        made.append(p)
        return p

    yield make
    for p in made:
        p.shutdown()
