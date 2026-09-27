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
    assert phase_b(sims, qs, lam=0.5, k2=1, sim_mean=0.68, sim_std=0.057) == [1]
    assert phase_b(sims, qs, lam=0.0, k2=1, sim_mean=0.68, sim_std=0.057) == [0]


def test_phase_b_uses_fixed_similarity_statistics():
    """MemRL z-scores similarity with corpus statistics, not the candidates' own."""
    sims, qs = np.array([0.80, 0.74]), np.array([0.0, 0.6])
    # Wide corpus spread: the 0.06 similarity gap is small, Q decides.
    assert phase_b(sims, qs, lam=0.5, k2=1, sim_mean=0.5, sim_std=0.5) == [1]
    # Narrow corpus spread: the same gap is two standard deviations, similarity decides.
    assert phase_b(sims, qs, lam=0.5, k2=1, sim_mean=0.7, sim_std=0.03) == [0]


def test_phase_b_epsilon_greedy_samples_all_candidates():
    import random
    sims, qs = np.array([0.9, 0.8, 0.7, 0.6]), np.zeros(4)
    picks = {tuple(phase_b(sims, qs, lam=0.5, k2=1, sim_mean=0.7, sim_std=0.1, epsilon=1.0, rng=random.Random(i)))
             for i in range(50)}
    assert len(picks) > 1
    assert phase_b(sims, qs, lam=0.5, k2=1, sim_mean=0.7, sim_std=0.1, epsilon=0.0) == [0]


def test_phase_b_edge_cases():
    kw = dict(lam=0.5, k2=3, sim_mean=0.68, sim_std=0.057)
    assert phase_b(np.array([]), np.array([]), **kw) == []
    assert phase_b(np.array([0.5]), np.array([0.2]), **kw) == [0]
    # equal Q: ranking falls back to similarity and stays finite
    assert phase_b(np.array([0.4, 0.9, 0.6]), np.zeros(3), **kw) == [1, 2, 0]


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
    (mem,) = p._store.get_memories(p._store.memory_ids())
    assert mem.kind == "experience"
    assert "cargo build" in mem.experience
    assert mem.q_value == pytest.approx(0.0)


# -- failure reflections ---------------------------------------------------

def test_failure_seeds_reflection_that_is_retrieved_first(make_provider):
    p = make_provider(session_id="fail")
    run_session(p, "migrate the postgres schema safely", "ran DROP TABLE users", reward=-1.0)
    (mem,) = p._store.get_memories(p._store.memory_ids())
    assert mem.kind == "reflection"
    assert mem.experience.startswith("[PATTERN TO AVOID]")
    assert "tests failed" in mem.experience
    assert mem.q_value == pytest.approx(0.0)  # MemRL q_init_neg

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
    (mem,) = p._store.get_memories(p._store.memory_ids())
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
    assert p._store.memory_ids() == []
    assert p._store.steps("sub") == []
    p.shutdown()


def test_shutdown_commits_like_oneshot_exit(make_provider, tmp_path, embedder):
    p = make_provider(session_id="oneshot")
    p.prefetch("summarize the repository layout")
    p.sync_turn("summarize the repository layout", "The repo has cmd/ and pkg/.")
    p.shutdown()  # what the atexit hook runs
    p.initialize("after", hermes_home=str(tmp_path))
    (mem,) = p._store.get_memories(p._store.memory_ids())
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


# -- intent and model-built experiences ---------------------------------------

SWE_PROMPT = (
    "<uploaded_files>\n/app\n</uploaded_files>\nI've uploaded a code repository. Consider the following question:\n\n"
    "<question>\n{q}\n</question>\n\nCan you help me answer this question? Write the answer to /logs/agent/answer.txt."
)


def write_state_db(home, session_id, rows):
    import sqlite3
    conn = sqlite3.connect(str(home / "state.db"))
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, "
                 "content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, timestamp REAL)")
    for role, content, tool_calls, tool_name in rows:
        conn.execute("INSERT INTO messages (session_id, role, content, tool_calls, tool_name, timestamp) "
                     "VALUES (?, ?, ?, ?, ?, 0)", (session_id, role, content, tool_calls, tool_name))
    conn.commit()
    conn.close()


class FakeChat:
    def __init__(self, reply="", error=None, usage=None):
        self.reply, self.error, self.calls = reply, error, []
        self.usage, self.last_usage = usage or {}, {}

    def __call__(self, messages, temperature):
        self.calls.append((messages[0]["content"], temperature))
        if self.error:
            raise self.error
        self.last_usage = self.usage
        return self.reply


def test_intent_is_the_question_block(make_provider):
    p = make_provider()
    p.prefetch(SWE_PROMPT.format(q="How does the backend start in dev mode?"))
    p.sync_turn("prompt", "answer")
    p.handle_tool_call(FEEDBACK_TOOL, {"reward": 1})
    p._join_pending()
    p._commit(p._session_id, None)
    (mem,) = p._store.get_memories(p._store.memory_ids())
    assert mem.intent == "How does the backend start in dev mode?"


def test_intent_tag_can_be_disabled_or_absent():
    from memrl.procedure import extract_intent
    prompt = SWE_PROMPT.format(q="Q?")
    assert extract_intent(prompt, "") == prompt
    assert extract_intent("no tags here", "question") == "no tags here"
    assert extract_intent("<question>  </question>", "question") == "<question>  </question>"


def test_shared_boilerplate_no_longer_drives_retrieval(make_provider):
    p = make_provider()
    seed(p, "how are alias suffixes signed and validated", "alias hint", 0.0)
    block = p.prefetch(SWE_PROMPT.format(q="which celery tasks does paperless schedule nightly"))
    assert "alias hint" not in block


def test_success_stores_script_plus_trajectory_from_session_store(make_provider, tmp_path):
    chat = FakeChat("1. Start postgres on 15432\n2. Run init_app.add_sl_domains()")
    p = make_provider(session_id="swe", chat=chat)
    write_state_db(tmp_path, "swe", [
        ("user", SWE_PROMPT.format(q="How is the dev server started?"), "", ""),
        ("assistant", "Let me look.", json.dumps([{"function": {"name": "terminal", "arguments": "{\"command\": \"ls /app\"}"}}]), ""),
        ("tool", json.dumps({"output": "server.py", "exit_code": 0}), "", "terminal"),
        ("assistant", "It runs server.py.", "", ""),
    ])
    run_session(p, SWE_PROMPT.format(q="How is the dev server started?"), "It runs server.py.", reward=1.0)
    (mem,) = p._store.get_memories(p._store.memory_ids())
    assert mem.experience.startswith("Task: How is the dev server started?\n\nSCRIPT:\n1. Start postgres on 15432")
    assert "TRAJECTORY:\n" in mem.experience
    assert 'call terminal({"command": "ls /app"})' in mem.experience
    assert "tool terminal:" in mem.experience
    (prompt, temperature), = chat.calls
    assert "high-level script" in prompt and "call terminal(" in prompt and temperature == 0.0


def test_failure_stores_model_reflection(make_provider):
    chat = FakeChat("Assumed the default DB port; check tests/test.env first.")
    p = make_provider(chat=chat)
    run_session(p, "migrate the postgres schema safely", "ran DROP TABLE users", reward=-1.0)
    (mem,) = p._store.get_memories(p._store.memory_ids())
    assert mem.kind == "reflection" and mem.experience.startswith("[PATTERN TO AVOID]\nTASK REFLECTION:")
    assert "What went wrong:\nAssumed the default DB port" in mem.experience
    assert "Failed approach:\n" in mem.experience and "DROP TABLE" in mem.experience
    assert chat.calls[0][1] == 0.3


def test_model_failure_still_stores_the_trajectory(make_provider):
    p = make_provider(chat=FakeChat(error=TimeoutError("slow")))
    run_session(p, "compile the rust crate with features", "cargo build worked", reward=1.0)
    (mem,) = p._store.get_memories(p._store.memory_ids())
    assert "SCRIPT:" not in mem.experience and "cargo build worked" in mem.experience


def test_format_trajectory_drops_the_middle():
    from memrl.procedure import format_trajectory
    messages = [{"role": "assistant", "content": f"step {i:03d}", "tool_calls": "", "tool_name": ""} for i in range(200)]
    text = format_trajectory(messages, 100, 400)
    assert "step 000" in text and "step 199" in text and "middle of trajectory omitted" in text
    assert len(text) < 500


def test_hermes_chat_uses_the_configured_model(tmp_path, monkeypatch):
    import http.server
    import threading
    from memrl.procedure import hermes_chat

    seen = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"], seen["auth"] = self.path, self.headers["Authorization"]
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = json.dumps({"choices": [{"message": {"content": " the script "}}],
                                "usage": {"prompt_tokens": 11, "completion_tokens": 3}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        (tmp_path / "config.yaml").write_text(
            f'model:\n  default: "deepseek-flash"\n  base_url: "http://127.0.0.1:{server.server_port}/v1"\n'
            '  api_key: "${TEST_MEMRL_KEY}"\n')
        monkeypatch.delenv("TEST_MEMRL_KEY", raising=False)
        assert hermes_chat(tmp_path) is None
        monkeypatch.setenv("TEST_MEMRL_KEY", "sk-test")
        chat = hermes_chat(tmp_path)
        assert chat([{"role": "user", "content": "hi"}], 0.7) == "the script"
        assert chat.last_usage == {"prompt_tokens": 11, "completion_tokens": 3}
    finally:
        server.shutdown()
    assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "deepseek-flash" and seen["body"]["temperature"] == 0.7


# -- MemRL parity: defaults, key grouping, external rewards --------------------

def test_defaults_follow_memrl_bcb_config():
    from memrl import MemRLConfig
    cfg = MemRLConfig()
    assert (cfg.k1, cfg.k2, cfg.lam, cfg.alpha, cfg.epsilon) == (5, 5, 0.5, 0.3, 0.1)
    assert (cfg.q_init, cfg.q_reflection, cfg.add_similarity, cfg.script_temperature) == (0.0, 0.0, 0.9, 0.0)
    assert (cfg.delta, cfg.sim_norm_mean, cfg.sim_norm_std) == (0.72, 0.679, 0.0567)
    assert cfg.reward_source == "agent"


def test_near_duplicate_task_joins_the_retrieved_key(make_provider, embedder):
    embedder.pinned = {"task a": unit(1, 0, 0), "task a again": unit(1, 0.05, 0), "task b": unit(0.6, 0.8, 0)}
    p = make_provider(session_id="one")
    run_session(p, "task a", "did a", reward=1.0)
    p2 = make_provider(session_id="two")
    run_session(p2, "task a again", "did a again", reward=1.0)  # cosine ~0.999 >= 0.9
    p3 = make_provider(session_id="three")
    run_session(p3, "task b", "did b", reward=1.0)  # cosine 0.6 < 0.9
    first, second, third = p3._store.get_memories(p3._store.memory_ids())
    assert second.key_id == first.id == first.key_id
    assert third.key_id == third.id
    key_ids, _ = p3._store.keys()
    assert key_ids == [first.id, third.id]
    # Both memories under the key are candidates when the key is retrieved.
    p4 = make_provider(session_id="four")
    block = p4.prefetch("task a")
    assert "did a" in block and "did a again" in block


def external_provider(make_provider, monkeypatch, task_id, session_id, **cfg):
    monkeypatch.setenv("MEMRL_TASK_ID", task_id)
    return make_provider(session_id=session_id, reward_source="external", **cfg)


def test_external_rewards_are_applied_before_the_next_recall(make_provider, monkeypatch, tmp_path, embedder):
    embedder.pinned = {"query task": unit(1, 0), "past task": unit(1, 0.1)}
    p = external_provider(make_provider, monkeypatch, "task-001", "s1")
    assert p.get_tool_schemas() == [] and FEEDBACK_TOOL not in p.system_prompt_block()
    past = seed(p, "past task", "past experience", 0.0)
    p.prefetch("query task")
    p.sync_turn("query task", "an answer")
    p.shutdown()  # one-shot exit: parked, nothing learned yet
    p.initialize("peek", hermes_home=str(tmp_path))
    p._finalizer.join()
    assert [x.task_id for x in p._store.pending()] == ["task-001"]
    (mem,) = p._store.get_memories([past])
    assert mem.q_value == 0.0 and len(p._store.memory_ids()) == 1
    p.shutdown()

    (tmp_path / "memrl" / "rewards.json").write_text(json.dumps({"task-001": -1}))
    p2 = external_provider(make_provider, monkeypatch, "task-002", "s2")
    p2.prefetch("an unrelated follow-up question here")
    assert p2._store.pending() == []
    past_mem, new = p2._store.get_memories(p2._store.memory_ids())
    assert past_mem.q_value == pytest.approx(-0.3)
    assert new.kind == "reflection" and new.intent == "query task"


def test_external_null_reward_drops_and_unknown_task_waits(make_provider, monkeypatch, tmp_path):
    for task, session in (("task-001", "a"), ("task-002", "b")):
        p = external_provider(make_provider, monkeypatch, task, session)
        p.prefetch(f"question for {task} about the repository")
        p.sync_turn("q", "a")
        p.shutdown()
    (tmp_path / "memrl" / "rewards.json").write_text(json.dumps({"task-001": None}))
    p3 = external_provider(make_provider, monkeypatch, "task-003", "c")
    p3.prefetch("third question about the repository layout")
    assert [x.task_id for x in p3._store.pending()] == ["task-002"]
    assert p3._store.memory_ids() == []


# -- experiment bookkeeping -----------------------------------------------------

def test_memory_upkeep_calls_and_source_task_are_recorded(make_provider, monkeypatch, tmp_path):
    chat = FakeChat("1. run it", usage={"prompt_tokens": 900, "completion_tokens": 40})
    p = external_provider(make_provider, monkeypatch, "task-001", "a", chat=FakeChat(error=TimeoutError("unused")))
    p.prefetch("how is the worker queue configured in this repository")
    p.sync_turn("q", "a")
    p.shutdown()
    (tmp_path / "memrl" / "rewards.json").write_text(json.dumps({"task-001": 1}))
    # Task 001's memory is written by the next session, with that session's model.
    p2 = external_provider(make_provider, monkeypatch, "task-002", "b", chat=chat)
    p2.prefetch("an unrelated second question about the repository")
    (mem,) = p2._store.get_memories(p2._store.memory_ids())
    rows = p2._store._conn.execute(
        "SELECT session_id, task_id, purpose, ok, prompt_tokens, completion_tokens FROM llm_calls").fetchall()
    source = p2._store._conn.execute("SELECT task_id FROM memories WHERE id = ?", (mem.id,)).fetchone()
    assert rows == [("a", "task-001", "script", 1, 900, 40)]
    assert source == ("task-001",)


def test_store_written_before_task_id_is_migrated(tmp_path):
    import sqlite3
    from memrl.store import Store
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, key_id TEXT NOT NULL, intent TEXT NOT NULL,"
                 " experience TEXT NOT NULL, embedding BLOB NOT NULL, q_value REAL NOT NULL,"
                 " usage_count INTEGER NOT NULL DEFAULT 0, kind TEXT NOT NULL DEFAULT 'experience',"
                 " created_at REAL NOT NULL, updated_at REAL NOT NULL)")
    conn.close()
    store = Store(path)
    mid = store.add_memory("i", "e", np.ones(4, dtype=np.float32), 0.0, "experience", task_id="task-009")
    assert store._conn.execute("SELECT task_id FROM memories WHERE id = ?", (mid,)).fetchone() == ("task-009",)
    store.close()


def test_similarity_only_retrieval_ignores_utility(make_provider, embedder):
    """The ablation arm (MEMRL_LAM=0, MEMRL_EPSILON=0) ranks by similarity alone."""
    embedder.pinned = {"query task": unit(1, 0, 0), "close task": unit(1, 0.1, 0), "far task": unit(1, 0.6, 0)}
    p = make_provider(lam=0.0, k2=1)
    seed(p, "close task", "close but useless", -0.9)
    seed(p, "far task", "farther but proven", 0.9)
    assert "close but useless" in p.prefetch("query task")
