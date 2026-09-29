#!/usr/bin/env python3
"""Per-task critical-path profile of Hermes runs, with MemRL attribution.

Hermes exports its session only at the end, so per-call times are rebuilt from
the bridge: agent execs separated by more than --gap seconds belong to
different model calls, the gaps between clusters are model time and the
clusters are tool time. Hermes records token totals per session only, so each
call's prompt is modelled as a linear ramp from the first to the last prompt
with the session's mean prompt size; prefill/decode then come from the server
calibration (scripts/calibrate_llm_server.py) and queueing is the residual.

MemRL injects its recalled memories into the user message of every model call
of the turn (never persisted), so the injected block is rebuilt from the task's
memrl.db (sessions.active_ids joined to memories) and charged on every call.
"Used later" is a lexical check: a memory counts as used when one of its
distinctive identifiers (paths, dotted or snake_case names) that is absent from
the question appears in the agent's later tool calls or final answer.

Usage:
  scripts/profile_hermes_critical_path.py --calibration scripts/out_profile/calibration.json \
      --out scripts/out_profile/hermes/tasks.csv runs/memrl-study/<run> [...]
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(__file__))
from profile_critical_path import Model, Tokenizer  # noqa: E402

IDENT = re.compile(r"[A-Za-z_][\w./-]*[./_][\w./-]*\w")


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def clusters(bridge, gap):
    execs = sorted((ts(r["timestamp"]) - r["duration_ms"] / 1000, ts(r["timestamp"]))
                   for r in bridge if r.get("request_type") == "exec" and r.get("operation_class") == "agent")
    out = []
    for s, e in execs:
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def injected(db_path, session_ids):
    """The MemRL block injected into each call, and the recalled memories."""
    if not os.path.exists(db_path):
        return "", []
    db = sqlite3.connect(db_path)
    mems = []
    for sid in session_ids:
        row = db.execute("select active_ids from sessions where session_id = ?", (sid,)).fetchone()
        if row and row[0]:
            for mid in json.loads(row[0]):
                m = db.execute("select intent, experience, q_value from memories where id = ?", (mid,)).fetchone()
                if m and m not in mems:
                    mems.append(m)
    parts = [f"### Memory {n} (utility {q:+.2f})\nPast task: {i}\n{e}" for n, (i, e, q) in enumerate(mems, 1)]
    return ("<memrl-memory>\n" + "\n\n".join(parts) + "\n</memrl-memory>") if parts else "", mems


def profile(run, task_dir, model, tok, gap):
    h = task_dir + "/harness"
    sess_path, bridge_path = h + "/telemetry/sessions.jsonl", task_dir + "/bridge/tool-calls.jsonl"
    if not (os.path.exists(sess_path) and os.path.exists(bridge_path) and os.path.exists(h + "/session-outcome.json")):
        return None
    sessions = [json.loads(l) for l in open(sess_path)]
    outcome = json.load(open(h + "/session-outcome.json"))
    t0, t1 = ts(outcome["started_at"]), ts(outcome["ended_at"])
    calls = sum(s.get("api_call_count") or 0 for s in sessions)
    n_in = sum(s.get("input_tokens") or 0 for s in sessions)
    n_out = sum(s.get("output_tokens") or 0 for s in sessions)
    cached = sum(s.get("cache_read_tokens") or 0 for s in sessions)
    if not calls:
        return None
    cl = clusters([json.loads(l) for l in open(bridge_path)], gap)
    tool_s = sum(e - s for s, e in cl)
    edges = [t0] + [x for c in cl for x in c] + [t1]
    llm_s = sum(max(0.0, edges[i + 1] - edges[i]) for i in range(0, len(edges), 2))
    first_gap = (cl[0][0] - t0) if cl else t1 - t0

    msgs = [m for s in sessions for m in s.get("messages") or []]
    question = next((m["content"] for m in msgs if m["role"] == "user"), "") or ""
    system = sessions[0].get("system_prompt") or ""
    block, mems = injected(h + "/memrl/memrl.db", [s["id"] for s in sessions])
    k = tok.count(block)
    first = tok.count(system) + tok.count(question) + k
    mean = n_in / calls
    last = max(first, 2 * mean - first)
    prompts = [first + (last - first) * i / max(1, calls - 1) for i in range(calls)]
    prefill = sum(model.prefill(n) for n in prompts)
    mem_prefill = sum(model.prefill(n) - model.prefill(max(0, n - k)) for n in prompts) if k else 0.0
    decode = n_out * model.tpot(mean)

    later = " ".join(json.dumps(m.get("tool_calls") or []) + " " + (m["content"] or "")
                     for m in msgs if m["role"] == "assistant")
    answer_path = task_dir + "/evaluation/answer.txt"
    later += open(answer_path, errors="replace").read() if os.path.exists(answer_path) else ""
    q_ids = set(IDENT.findall(question))
    used = sum(1 for _, e, _ in mems if any(i in later for i in set(IDENT.findall(e)) - q_ids if len(i) >= 8))
    return {
        "run": run, "task": os.path.basename(task_dir).rsplit("-", 1)[0], "session_s": t1 - t0,
        "api_calls": calls, "input_tokens": n_in, "output_tokens": n_out, "cache_read_tokens": cached,
        "compressed": len(sessions) > 1, "llm_s": llm_s, "tool_s": tool_s, "first_call_s": first_gap,
        "prefill_s": prefill, "decode_s": decode, "queue_s": llm_s - prefill - decode,
        "memory_tokens_per_call": k, "memory_tokens_prefilled": k * calls,
        "memory_prefill_s": mem_prefill, "memories_recalled": len(mems), "memories_used_lexical": used,
        "prefix_saving_s": prefill - sum(model.prefill(b - a) for a, b in zip([0] + prompts, prompts)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap", type=float, default=5.0)
    ap.add_argument("--base-url", default="http://192.168.111.200:8100/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    a = ap.parse_args()
    model, tok = Model(a.calibration), Tokenizer(a.base_url, a.model)
    rows = []
    for run_path in a.runs:
        run = os.path.basename(run_path.rstrip("/"))
        for td in sorted(glob.glob(run_path.rstrip("/") + "/task-*")):
            r = profile(run, td, model, tok, a.gap)
            if r:
                rows.append(r)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} tasks -> {a.out}")


if __name__ == "__main__":
    main()
