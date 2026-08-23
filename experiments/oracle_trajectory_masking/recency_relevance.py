"""For every candidate search/fetch round: how far back it is (in rounds), whether it's
topically similar to the current turn's action, and whether it was kept -- to answer "is
necessary context recent, and is it topically relevant to the current search?"
"""
import glob
import json
import os
import statistics

from context_chunks import unit_label
from match import _similarity
from trajectory_extractor import extract_turns


def find_sessions_dir(base, task_id, _cache={}):
    if base not in _cache:
        mapping = {}
        for manifest_path in glob.glob(f"{base}/cells/*/runs/*/*/manifest.json"):
            m = json.load(open(manifest_path))
            mapping[m["instance_id"]] = os.path.join(os.path.dirname(manifest_path), "conversation", "sessions")
        _cache[base] = mapping
    return _cache[base].get(task_id)


def round_query_texts(chunk):
    calls = chunk.assistant_message.get("tool_calls", [])
    texts = []
    for tc in calls:
        try:
            args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(args, dict) and "query" in args:
            texts.append(args["query"])
        elif isinstance(args, dict) and "url" in args:
            texts.append(args["url"])
    return texts


def round_tool_type(chunk):
    names = set(c["function"]["name"] for c in chunk.assistant_message.get("tool_calls", []))
    if names <= {"web_search"}:
        return "web_search"
    if names <= {"web_fetch"}:
        return "web_fetch"
    return "mixed/other"


def current_query_texts(reference_command):
    texts = []
    for line in reference_command.split("\n"):
        texts.append(line.split(":", 1)[1].strip() if ":" in line else line)
    return texts


if __name__ == "__main__":
    base = "/home/ws/ARIES/runs/eviction_fanout_deepresearch_bench_openclaw_subset20_baseline_20260729_175953"
    rows = []
    seen = set()
    for f in sorted(glob.glob("results/eviction-fanout-run-*/*/turns.jsonl")):
        task_id = f.split("/")[-2]
        if task_id in seen:
            continue
        recs = [json.loads(line) for line in open(f)]
        if not recs:
            continue
        seen.add(task_id)
        sess_dir = find_sessions_dir(base, task_id)
        turns = extract_turns(sess_dir)
        turns_by_idx = {t.index: t for t in turns}
        for r in recs:
            if not r.get("reproducible"):
                continue
            t = turns_by_idx[r["turn_index"]]
            minimal = set(r["minimal_unit_labels"])
            current_texts = current_query_texts(r["reference_command"])
            round_chunks = [c for c in t.chunks if c.kind == "assistant_action"]
            n_rounds = len(round_chunks)
            for pos, c in enumerate(round_chunks):
                rounds_back = n_rounds - pos
                tool_type = round_tool_type(c)
                qtexts = round_query_texts(c)
                relevance = max((_similarity(q, ct) for q in qtexts for ct in current_texts), default=0.0)
                call_kept = unit_label(t, f"call:{c.id}") in minimal
                result_kept = bool(c.tool_results) and unit_label(t, f"result:{c.id}") in minimal
                rows.append(
                    {
                        "task_id": task_id,
                        "turn_index": r["turn_index"],
                        "rounds_back": rounds_back,
                        "tool_type": tool_type,
                        "relevance": relevance,
                        "call_kept": call_kept,
                        "result_kept": result_kept,
                        "kept_any": call_kept or result_kept,
                    }
                )

    with open("results/recency_relevance_rows.json", "w") as f:
        json.dump(rows, f)

    def summarize(rows, group_key, value_key):
        groups = {}
        for r in rows:
            groups.setdefault(r[group_key], []).append(r[value_key])
        return {k: (statistics.mean(v), len(v)) for k, v in groups.items()}

    print(f"{len(rows)} candidate rounds total\n")

    print("=== recency (rounds back) by kept status ===")
    for k, (mean, n) in summarize(rows, "kept_any", "rounds_back").items():
        print(f"  kept={k}: mean rounds_back={mean:.2f} (n={n})")

    print("\n=== relevance (similarity to current search) by kept status ===")
    for k, (mean, n) in summarize(rows, "kept_any", "relevance").items():
        print(f"  kept={k}: mean relevance={mean:.3f} (n={n})")

    print("\n=== recency + relevance, by tool type and kept status ===")
    combo = {}
    for r in rows:
        key = (r["tool_type"], r["kept_any"])
        combo.setdefault(key, []).append(r)
    for (tool_type, kept), rs in sorted(combo.items()):
        rb = statistics.mean(r["rounds_back"] for r in rs)
        rel = statistics.mean(r["relevance"] for r in rs)
        print(f"  {tool_type:<12} kept={kept!s:<5} n={len(rs):>3}  mean rounds_back={rb:.2f}  mean relevance={rel:.3f}")
