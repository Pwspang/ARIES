#!/usr/bin/env python3
"""Per-turn critical-path profile of OpenClaw runs, with memory attribution.

For every model call ("turn") in every task of the given runs, this records the
wall time of the call, its input/output tokens, the tool phase that follows it
(split by tool), memory-plugin operation latencies, the memory-injected tokens
sitting in the context at that call, and whether the call was spent purely on a
memory tool. Each model call's wall time is decomposed with a server
calibration (scripts/calibrate_llm_server.py) into

  prefill   = TTFT(input_tokens)            (prefix caching is off on the server)
  decode    = output_tokens * TPOT(input_tokens)
  queueing  = wall - prefill - decode       (residual: contention + queueing)

and the prefill share caused by memory tokens is TTFT(n) - TTFT(n - k) for k
memory tokens in context. Cache-hit tokens and the theoretical prefix reuse
(the fraction of each prompt that is a verbatim prefix of the previous prompt,
i.e. what an ideal prefix cache could skip) are reported too.

Retrieval use comes from scripts/judge_retrieval_usefulness.py output when it
covers the run.

Usage:
  scripts/profile_critical_path.py --calibration scripts/out_profile/calibration.json \
      --out-dir scripts/out_profile runs/<run> [runs/<run> ...]
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import urllib.request

TS = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)\+00:00 (.*)$")
MAX_CONTEXT = 262144
MEMORY_TOOLS = {"memory_search", "memory_add", "memory_list", "memory_consolidate",
                "memory_quality_scan", "memory_add_episodic"}


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class Model:
    """Calibrated single-stream prefill/decode cost of the model server."""

    def __init__(self, path):
        import numpy as np
        c = json.load(open(path))
        n = np.array([p["prompt_tokens"] for p in c["points"]], float)
        t = np.array([p["ttft_s"] for p in c["points"]])
        tp = np.array([p["tpot_s"] for p in c["points"]])
        self.a = np.linalg.lstsq(np.vstack([np.ones_like(n), n, n * n]).T, t, rcond=None)[0]
        self.b = np.linalg.lstsq(np.vstack([np.ones_like(n), n]).T, tp, rcond=None)[0]

    def prefill(self, n):
        return 0.0 if n <= 0 else self.a[0] + self.a[1] * n + self.a[2] * n * n

    def tpot(self, n):
        return self.b[0] + self.b[1] * n


class Tokenizer:
    def __init__(self, base, model):
        self.url, self.model, self.cache = base.rsplit("/v1", 1)[0] + "/tokenize", model, {}

    def count(self, text):
        if not text:
            return 0
        if text not in self.cache:
            req = urllib.request.Request(self.url, json.dumps({"model": self.model, "prompt": text}).encode(),
                                         {"content-type": "application/json"})
            self.cache[text] = json.load(urllib.request.urlopen(req, timeout=60))["count"]
        return self.cache[text]


def text_of(content):
    if isinstance(content, str):
        return content
    return "".join(c.get("text", "") for c in content or [] if isinstance(c, dict))


def gateway_events(path):
    """Model-fetch starts and memory plugin operations from gateway.log."""
    fetch, ops = [], []
    for line in open(path, errors="replace"):
        m = TS.match(line.rstrip("\n"))
        if not m:
            continue
        t, body = ts(m.group(1)), m.group(2)
        if "[model-fetch] start" in body:
            fetch.append(t)
            continue
        o = re.search(r"openclaw-amem: (memory_\w+)\b.*\((\d+)ms\)", body)
        if o:
            ops.append((t, o.group(1), int(o.group(2)) / 1000))
    return sorted(fetch), sorted(ops)


def usefulness(paths):
    """(run_dir, task_id, call_seq) -> any retrieved note judged used."""
    used = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            if r.get("llm_used") not in ("True", "False"):
                continue
            key = (r["run_dir"], r["task_id"], int(r["call_seq"]))
            used[key] = used.get(key, False) or r["llm_used"] == "True"
    return used


def profile_task(run, task_dir, model, tok, used):
    sess = [f for f in glob.glob(task_dir + "/harness-turn-01/telemetry/*.jsonl")
            if not f.endswith("trajectory.jsonl")]
    gw = task_dir + "/harness-turn-01/gateway.log"
    if not sess or not os.path.exists(gw):
        return [], {}
    fetch, ops = gateway_events(gw)
    msgs = []
    for line in open(sess[0]):
        d = json.loads(line)
        if d.get("type") == "message":
            msgs.append((ts(d["timestamp"]), d["message"]))
    traj = glob.glob(task_dir + "/harness-turn-01/telemetry/*.trajectory.jsonl")
    system_tokens = 0
    for line in open(traj[0]) if traj else []:
        e = json.loads(line)
        if e["type"] == "context.compiled":
            system_tokens = tok.count(e["data"].get("systemPrompt", ""))
    task_id = os.path.basename(task_dir).rsplit("-", 1)[0]

    turns, search_seq = [], 0
    mem_ctx = 0            # memory-result tokens currently in context
    prev_input = 0
    for i, (t_end, m) in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prev_t = max((t for t, _ in msgs[:i]), default=t_end)
        starts = [f for f in fetch if prev_t - 0.5 <= f <= t_end]
        t_start = starts[-1] if starts else prev_t
        calls = [c for c in m["content"] if c.get("type") == "toolCall"]
        names = [c["name"] for c in calls]
        u = m.get("usage") or {}
        n_in, n_out, cached = u.get("input", 0), u.get("output", 0), u.get("cacheRead", 0)
        # a few final messages carry usage aggregated over several calls; no single
        # prompt can exceed the server's context window, so drop those
        usage_valid = n_in <= MAX_CONTEXT
        if not usage_valid:
            n_in = n_out = cached = 0
        wall = t_end - t_start
        prefill = model.prefill(n_in - cached)
        decode = n_out * model.tpot(n_in)
        mem_prefill = model.prefill(n_in) - model.prefill(n_in - mem_ctx) if mem_ctx else 0.0
        query_tokens = sum(tok.count(json.dumps(c.get("arguments", {})))
                           for c in calls if c["name"] in MEMORY_TOOLS)
        # tool phase: this turn's results
        results, j = [], i + 1
        while j < len(msgs) and msgs[j][1]["role"] == "toolResult":
            results.append(msgs[j]); j += 1
        t_tools_end = results[-1][0] if results else t_end
        mem_ops = [o for o in ops if t_end - 0.05 <= o[0] <= t_tools_end + 0.05]
        search_used = None
        added = 0
        for rt, r in results:
            if r["toolName"] == "memory_search":
                search_seq += 1
                k = tok.count(text_of(r["content"]))
                added += k
                key = (run, task_id, search_seq)
                if key in used:
                    search_used = (search_used or False) or used[key]
            elif r["toolName"] in MEMORY_TOOLS:
                added += tok.count(text_of(r["content"]))
        turns.append({
            "run": run, "task": task_id, "turn": len(turns) + 1,
            "t_start": t_start, "llm_wall_s": wall, "input_tokens": n_in, "output_tokens": n_out,
            "cache_read_tokens": cached, "usage_valid": usage_valid,
            "prefix_reuse_tokens": min(prev_input, n_in),  # append-only transcript: previous prompt is a prefix
            "prefill_s": prefill, "prefill_new_s": model.prefill(n_in - min(prev_input, n_in)), "decode_s": decode, "queue_s": wall - prefill - decode,
            "memory_ctx_tokens": mem_ctx, "system_tokens": system_tokens,
            "memory_prefill_s": mem_prefill,
            "memory_query_tokens": query_tokens, "memory_query_decode_s": query_tokens * model.tpot(n_in),
            "tools": "+".join(sorted(set(names))), "memory_only_turn": bool(names) and all(n in MEMORY_TOOLS for n in names),
            "tool_phase_s": t_tools_end - t_end,
            "memory_search_s": sum(o[2] for o in mem_ops if o[1] == "memory_search"),
            "memory_add_s": sum(o[2] for o in mem_ops if o[1] == "memory_add"),
            "memory_other_s": sum(o[2] for o in mem_ops if o[1] not in ("memory_search", "memory_add")),
            "memory_tool_phase_s": max([rt - t_end for rt, r in results if r["toolName"] in MEMORY_TOOLS], default=0.0),
            "retrieval_used": search_used,
            "memory_tokens_added": added,
        })
        mem_ctx += added
        prev_input = n_in if usage_valid else prev_input
    final = [e for e in (json.loads(l) for l in open(traj[0])) if e["type"] == "session.ended"] if traj else []
    t_first = msgs[0][0] if msgs else 0
    summary = {"run": run, "task": task_id,
               "session_s": (ts(final[-1]["ts"]) if final else msgs[-1][0]) - t_first,
               "turns": len(turns)}
    return turns, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-url", default="http://192.168.111.200:8100/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--usefulness", nargs="*", default=["scripts/out_retrieval_filter/retrieval_usefulness_judged.csv",
                                                        "scripts/out/memory_note_usage_judged.csv"])
    a = ap.parse_args()
    model, tok, used = Model(a.calibration), Tokenizer(a.base_url, a.model), usefulness(a.usefulness)
    turns, tasks = [], []
    for run_path in a.runs:
        run = os.path.basename(run_path.rstrip("/"))
        for td in sorted(glob.glob(run_path.rstrip("/") + "/task-*")):
            t, s = profile_task(run, td, model, tok, used)
            turns += t
            if s:
                tasks.append(s)
    # server load: model calls in flight (across all profiled runs) at each call's start
    spans = sorted((t["t_start"], t["t_start"] + t["llm_wall_s"]) for t in turns)
    for t in turns:
        t["inflight"] = sum(1 for s, e in spans if s <= t["t_start"] < e)
    os.makedirs(a.out_dir, exist_ok=True)
    for name, rows in (("turns.csv", turns), ("tasks.csv", tasks)):
        with open(os.path.join(a.out_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    print(f"{len(turns)} turns from {len(tasks)} tasks -> {a.out_dir}")


if __name__ == "__main__":
    main()
