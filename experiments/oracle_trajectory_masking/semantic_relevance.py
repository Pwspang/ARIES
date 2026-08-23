"""Semantic (not lexical) relevance: the server has no /v1/embeddings endpoint for this model
(confirmed: 404) and this sandbox has no local embedding library, so this asks the model itself
to judge semantic relatedness -- a 0-10 score, 3 samples averaged per round for stability --
rather than falling back to character-overlap similarity (see match.py's _similarity, used by
recency_relevance.py's earlier lexical-only pass).
"""
import glob
import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor

from context_chunks import unit_label
from model_client import ModelClient
from recency_relevance import current_query_texts, find_sessions_dir, round_query_texts, round_tool_type
from trajectory_extractor import extract_turns

PROMPT = """On a scale of 0 (completely unrelated topics) to 10 (essentially the same topic or question), how semantically related are these two pieces of search-agent context? Judge meaning and topic overlap, not shared words.

PAST ROUND (something previously searched or fetched):
{past}

CURRENT SEARCH (what the agent is about to search for right now):
{current}

Respond with ONLY a single integer from 0 to 10, nothing else."""


def score_once(client, past_text, current_text):
    predicted = client.complete("", [{"role": "user", "content": PROMPT.format(past=past_text, current=current_text)}], None)
    if predicted is None or "error" in predicted:
        return None
    m = re.search(r"\d+", predicted.get("text") or "")
    if not m:
        return None
    return max(0, min(10, int(m.group())))


def semantic_score(client, pool, past_text, current_text, samples=3):
    scores = list(pool.map(lambda _: score_once(client, past_text, current_text), range(samples)))
    scores = [s for s in scores if s is not None]
    return statistics.mean(scores) / 10 if scores else None


if __name__ == "__main__":
    base = "/home/ws/ARIES/runs/eviction_fanout_deepresearch_bench_openclaw_subset20_baseline_20260729_175953"
    client = ModelClient(base_url="http://localhost:8100/v1", model="Qwen/Qwen3.6-35B-A3B-FP8", api_key_env="SGLANG_API_KEY", temperature=0)

    tasks = []
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
            current_joined = " | ".join(current_texts)
            round_chunks = [c for c in t.chunks if c.kind == "assistant_action"]
            n_rounds = len(round_chunks)
            for pos, c in enumerate(round_chunks):
                from context_chunks import unit_label

                qtexts = round_query_texts(c)
                if not qtexts:
                    continue
                past_joined = " | ".join(qtexts)
                call_kept = unit_label(t, f"call:{c.id}") in minimal
                result_kept = bool(c.tool_results) and unit_label(t, f"result:{c.id}") in minimal
                tasks.append(
                    {
                        "task_id": task_id,
                        "turn_index": r["turn_index"],
                        "rounds_back": n_rounds - pos,
                        "tool_type": round_tool_type(c),
                        "kept_any": call_kept or result_kept,
                        "past": past_joined,
                        "current": current_joined,
                    }
                )

    print(f"{len(tasks)} candidate rounds to score (3 samples each = {len(tasks)*3} model calls)")

    with ThreadPoolExecutor(max_workers=12) as pool:
        for i, item in enumerate(tasks):
            item["semantic_relevance"] = semantic_score(client, pool, item["past"], item["current"])
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(tasks)} scored")

    with open("results/semantic_relevance_rows.json", "w") as f:
        json.dump(tasks, f)
    print(f"wrote results/semantic_relevance_rows.json")

    scored = [t for t in tasks if t["semantic_relevance"] is not None]
    kept = [t["semantic_relevance"] for t in scored if t["kept_any"]]
    dropped = [t["semantic_relevance"] for t in scored if not t["kept_any"]]
    print(f"\nkept:    mean={statistics.mean(kept):.3f} (n={len(kept)})")
    print(f"dropped: mean={statistics.mean(dropped):.3f} (n={len(dropped)})")
