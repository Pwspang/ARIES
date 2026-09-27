import json
import math

import numpy as np
import pytest

from memrl import FEEDBACK_TOOL, MemRLMemoryProvider
from memrl.retrieval import ema, phase_a, phase_b


def unit(*xs):
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def run_session(provider, task, answer, reward=None):
    """Drive one session through prefetch, a turn, optional feedback, and commit."""
    block = provider.prefetch(task)
    provider.sync_turn(task, answer)
    if reward is not None:
        provider.handle_tool_call(FEEDBACK_TOOL, {"reward": reward, "note": "tests failed"})
    provider._join_pending()
    provider._commit(provider._session_id, None)
    return block


def seed(provider, intent, experience, q, kind="experience"):
    emb = provider._embed(intent)
    return provider._store.add_memory(intent, experience, emb, q, kind)


# -- cold start ---------------------------------------------------------------

def test_cold_start_is_a_graceful_noop(make_provider):
    p = make_provider()
    assert p.prefetch("fix the failing build in the repository") == ""
    assert p.recall_status().glyph == "⚡"
    assert p.recall_status().count == 0
    p.on_session_end([])
    p.on_pre_compress([])
    p.shutdown()
    p.shutdown()  # idempotent


def test_uninitialized_provider_is_inert():
    p = MemRLMemoryProvider(embedder=lambda texts: np.zeros((len(texts), 4), dtype=np.float32))
    assert p.prefetch("anything at all here") == ""
    p.sync_turn("u", "a")
    p.shutdown()


def test_availability_reports_missing_packages(monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "sentence_transformers" else real(name, *a))
    p = MemRLMemoryProvider()
    assert not p.is_available()
    assert "sentence_transformers" in p.unavailable_reason()


# -- Phase A ------------------------------------------------------------------

def test_phase_a_threshold_and_cap():
    q = unit(1, 0)
    matrix = np.stack([unit(1, 0), unit(1, 1), unit(0, 1), unit(1, 0.2), unit(1, 2)])
    # cosines: 1.0, 0.707, 0.0, 0.98, 0.447
    idx, sims = phase_a(q, matrix, delta=0.38, k1=15)
    assert list(idx) == [0, 3, 1, 4]
    assert np.all(sims >= 0.38)
    idx, _ = phase_a(q, matrix, delta=0.38, k1=2)
    assert list(idx) == [0, 3]
    idx, _ = phase_a(q, np.empty((0, 0), dtype=np.float32), delta=0.38, k1=15)
    assert len(idx) == 0


def test_provider_drops_memories_below_threshold(make_provider, embedder):
    embedder.pinned = {
        "query task": unit(1, 0, 0),
        "close task": unit(1, 0.1, 0),
        "far task": unit(0.2, 1, 0),  # cosine ~0.196 < 0.38
    }
    p = make_provider()
    seed(p, "close task", "close experience", 0.0)
    seed(p, "far task", "far experience", 1.0)
    block = p.prefetch("query task")
    assert "close experience" in block
    assert "far experience" not in block
    assert p.recall_status().count == 1


# -- Phase B ------------------------------------------------------------------

def test_phase_b_utility_outranks_pure_similarity():
    sims = np.array([0.90, 0.85])
    qs = np.array([-0.8, 0.9])
    assert phase_b(sims, qs, lam=0.5, k2=1) == [1]
    assert phase_b(sims, qs, lam=0.0, k2=1) == [0]


def test_phase_b_edge_cases():
    assert phase_b(np.array([]), np.array([]), lam=0.5, k2=3) == []
    assert phase_b(np.array([0.5]), np.array([0.2]), lam=0.5, k2=3) == [0]
    # equal Q: ranking falls back to similarity and stays finite
    assert phase_b(np.array([0.4, 0.9, 0.6]), np.zeros(3), lam=0.5, k2=3) == [1, 2, 0]


def test_provider_injects_high_utility_memory_first(make_provider, embedder):
    embedder.pinned = {
        "query task": unit(1, 0, 0),
        "twin task": unit(1, 0.05, 0),
        "cousin task": unit(1, 0.3, 0),
    }
    p = make_provider(k2=1)
    seed(p, "twin task", "misleading experience", -0.8)
    seed(p, "cousin task", "useful experience", 0.9)
    block = p.prefetch("query task")
    assert "useful experience" in block
    assert "misleading experience" not in block


# -- EMA ----------------------------------------------------------------------

def test_ema_single_step_is_exact():
    assert ema(0.0, 1.0, 0.3) == pytest.approx(0.3)
    assert ema(0.5, -1.0, 0.3) == pytest.approx(0.5 + 0.3 * (-1.5))


def test_ema_converges_monotonically_and_stays_bounded():
    q, prev = 0.0, -math.inf
    for _ in range(200):
        q = ema(q, 1.0, 0.3)
        assert prev < q <= 1.0 or q == pytest.approx(1.0)
        prev = q
    assert q == pytest.approx(1.0, abs=1e-9)

    q = 0.0
    for i in range(500):
        q = ema(q, 1.0 if i % 2 else -1.0, 0.3)
        assert -1.0 <= q <= 1.0 and not math.isnan(q)


def test_ema_clips_out_of_range_rewards():
    assert ema(0.0, 50.0, 0.3) == pytest.approx(0.3)
    assert ema(0.0, -50.0, 0.3) == pytest.approx(-0.3)


def test_commit_updates_injected_memory_once(make_provider, tmp_path, embedder):
    embedder.pinned = {"query task": unit(1, 0), "past task": unit(1, 0.1)}
    p = make_provider()
    mid = seed(p, "past task", "past experience", 0.0)
    run_session(p, "query task", "done", reward=1.0)
    p.on_session_end([])  # second commit trigger must not double-apply
    p._join_pending()
    p.shutdown()
    p.initialize("s2", hermes_home=str(tmp_path))
    (mem,) = p._store.get_memories([mid])
    assert mem.q_value == pytest.approx(0.3)
    assert mem.usage_count == 1


def test_successful_session_is_stored_as_experience(make_provider):
    p = make_provider()
    run_session(p, "compile the rust crate with features", "cargo build --features x worked", reward=1.0)
    ids, _, qs = p._store.embeddings()
    (mem,) = p._store.get_memories(ids)
    assert mem.kind == "experience"
    assert "cargo build" in mem.experience
    assert qs[0] == pytest.approx(0.0)


# -- failure reflections ---------------------------------------------------

def test_failure_seeds_reflection_that_is_retrieved_first(make_provider):
    p = make_provider(session_id="fail")
    run_session(p, "migrate the postgres schema safely", "ran DROP TABLE users", reward=-1.0)
    ids, _, qs = p._store.embeddings()
    (mem,) = p._store.get_memories(ids)
    assert mem.kind == "reflection"
    assert mem.experience.startswith("[PATTERN TO AVOID]")
    assert "tests failed" in mem.experience
    assert qs[0] == pytest.approx(0.5)

    p2 = make_provider(session_id="next")  # same hermes_home, same DB
    seed(p2, "migrate the postgres schema quickly", "neutral hint", 0.0)
    block = p2.prefetch("migrate the postgres schema safely")
    assert block.index("[PATTERN TO AVOID]") < block.index("neutral hint")


def test_heuristic_reward_without_feedback(make_provider):
    p = make_provider()
    p.prefetch("run the unit test suite")
    p.sync_turn("run the unit test suite", "")
    p._join_pending()
    messages = [
        {"role": "tool", "content": json.dumps({"output": "", "exit_code": 1})},
        {"role": "tool", "content": json.dumps({"output": "", "exit_code": 2})},
    ]
    p._commit(p._session_id, messages)
    (mem,) = p._store.get_memories(p._store.embeddings()[0])
    assert mem.kind == "reflection"


# -- provider contract ----------------------------------------------------

def test_system_prompt_block_is_static(make_provider):
    p = make_provider()
    first = p.system_prompt_block()
    p.prefetch("some substantive task description")
    p.sync_turn("a", "b")
    assert p.system_prompt_block() == first
    assert FEEDBACK_TOOL in first


def test_trivial_prompts_bypass_retrieval(make_provider):
    p = make_provider()
    seed(p, "thanks", "should never appear", 1.0)
    assert p.prefetch("thanks") == ""
    assert p.prefetch("ok") == ""


def test_sync_turn_buffers_steps_in_background(make_provider):
    p = make_provider()
    for i in range(5):
        p.sync_turn(f"u{i}", f"a{i}")
    p.on_pre_compress([])  # checkpoint: joins pending writes
    assert p._store.steps("s1") == [(f"u{i}", f"a{i}") for i in range(5)]


def test_feedback_tool_validates_reward(make_provider):
    p = make_provider()
    assert [s["name"] for s in p.get_tool_schemas()] == [FEEDBACK_TOOL]
    assert "error" in json.loads(p.handle_tool_call(FEEDBACK_TOOL, {"reward": 2}))
    assert "error" in json.loads(p.handle_tool_call(FEEDBACK_TOOL, {"reward": "bad"}))
    assert json.loads(p.handle_tool_call(FEEDBACK_TOOL, {"reward": -0.5}))["recorded"]
    with pytest.raises(NotImplementedError):
        p.handle_tool_call("other_tool", {})


def test_non_primary_context_does_not_write(tmp_path, embedder):
    from memrl import MemRLConfig
    p = MemRLMemoryProvider(config=MemRLConfig(), embedder=embedder)
    p.initialize("sub", hermes_home=str(tmp_path), agent_context="subagent")
    p.prefetch("subagent task description")
    p.sync_turn("u", "a")
    p.handle_tool_call(FEEDBACK_TOOL, {"reward": 1})
    p._join_pending()
    p._commit("sub", None)
    assert p._store.embeddings()[0] == []
    assert p._store.steps("sub") == []
    p.shutdown()


def test_shutdown_commits_like_oneshot_exit(make_provider, tmp_path, embedder):
    p = make_provider(session_id="oneshot")
    p.prefetch("summarize the repository layout")
    p.sync_turn("summarize the repository layout", "The repo has cmd/ and pkg/.")
    p.shutdown()  # what the atexit hook runs
    p.initialize("after", hermes_home=str(tmp_path))
    (mem,) = p._store.get_memories(p._store.embeddings()[0])
    assert mem.kind == "experience"


def test_model_is_loaded_before_exit(tmp_path, embedder, monkeypatch):
    """The one-shot commit runs from atexit, where the model can no longer load.

    initialize starts loading it, and even a cold-start prefetch embeds the
    query, so the model is ready before the first model call.
    """
    import memrl
    loads, calls = [], []

    def counting(texts):
        calls.append(list(texts))
        return embedder(texts)

    monkeypatch.setattr(memrl, "_sentence_transformer_embedder", lambda: loads.append(1) or counting)
    p = MemRLMemoryProvider(config=memrl.MemRLConfig())
    p.initialize("warm", hermes_home=str(tmp_path))
    assert p.prefetch("recover the lost commit and merge it") == ""
    assert loads == [1]
    assert ["recover the lost commit and merge it"] in calls
    p.shutdown()
