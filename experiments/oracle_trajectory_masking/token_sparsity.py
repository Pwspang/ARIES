"""Token-based sparsity: what fraction of a turn's context *tokens* (not just unit count)
survive minimization, plus best/worst/average-case summary statistics.

Unit-count sparsity (used everywhere else in this project) treats "the system prompt" and
"one search round" as equally-weighted units, but they can differ by orders of magnitude in
size -- dropping a 30k-token system prompt and dropping a 40-token tool call both count as
"1 unit" under that metric. This measures the same masking result by actual token weight
instead, using the model's own tokenizer via the serving endpoint's /tokenize route (NOT an
approximation -- verified against real mixed English/Chinese content).

Usage:
  python token_sparsity.py \
    --source "aries:/path/to/aries/run/dir:results/run-final-consolidated/*/turns.jsonl" \
    --source "fanout:/path/to/eviction_fanout/run/dir:results/eviction-fanout-run-*/*/turns.jsonl" \
    --base-url http://localhost:8100/v1
"""
import argparse
import glob
import json
import os
import statistics
import urllib.request

from context_chunks import unit_ids, unit_label
from trajectory_extractor import extract_turns


class TokenCounter:
    """Wraps the serving endpoint's /tokenize route with a cache keyed by exact text, since
    the same chunk's text is referenced by every later turn that keeps it in its history."""

    def __init__(self, base_url, model):
        # /tokenize lives at the server root, not under /v1 -- base_url is like ".../v1"
        self.tokenize_url = base_url.rsplit("/v1", 1)[0] + "/tokenize"
        self.model = model
        self._cache = {}

    def count(self, text):
        if not text:
            return 0
        if text in self._cache:
            return self._cache[text]
        payload = json.dumps({"model": self.model, "prompt": text}).encode()
        req = urllib.request.Request(self.tokenize_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            n = json.loads(resp.read())["count"]
        self._cache[text] = n
        return n


def unit_text(turn, unit_id):
    """The text whose token count represents this unit -- the same content build_request would
    actually send for it, so token counts reflect real request weight, not a synthetic proxy."""
    if unit_id == "sys":
        return turn.system_prompt
    kind, _, chunk_id = unit_id.partition(":")
    chunk = next(c for c in turn.chunks if c.id == chunk_id)
    if kind == "call":
        text = chunk.assistant_message.get("content", "")
        for tc in chunk.assistant_message.get("tool_calls") or []:
            text += " " + tc["function"]["name"] + " " + tc["function"]["arguments"]
        return text
    if kind == "result":
        return "\n".join(tr["content"] for tr in chunk.tool_results)
    # plain "chunk:" -- a user message or compaction summary, single openai message
    return chunk.openai_messages[0]["content"]


def token_ratios_for_source(kind, base, results_glob, resolver, counter):
    seen = set()
    rows = []
    for f in sorted(glob.glob(results_glob)):
        task_id = f.split("/")[-2]
        if task_id in seen:
            continue
        recs = [json.loads(line) for line in open(f)]
        if not recs:
            continue
        seen.add(task_id)
        sess_dir = resolver(base, task_id)
        if not sess_dir:
            continue
        turns = extract_turns(sess_dir)
        turns_by_idx = {t.index: t for t in turns}
        for r in recs:
            if not r.get("reproducible"):
                continue
            t = turns_by_idx[r["turn_index"]]
            ids = unit_ids(t)
            minimal = set(r["minimal_unit_labels"])
            total_tokens = kept_tokens = 0
            for u in ids:
                n = counter.count(unit_text(t, u))
                total_tokens += n
                if unit_label(t, u) in minimal:
                    kept_tokens += n
            token_ratio = kept_tokens / total_tokens if total_tokens else 0
            unit_ratio = r["minimal_units"] / r["total_units"] if r["total_units"] else 0
            rows.append(
                {
                    "dataset": kind,
                    "task_id": task_id,
                    "turn_index": r["turn_index"],
                    "total_tokens": total_tokens,
                    "kept_tokens": kept_tokens,
                    "token_ratio": token_ratio,
                    "unit_ratio": unit_ratio,
                }
            )
    return rows


def summarize(rows, key):
    values = [r[key] for r in rows]
    best = min(rows, key=lambda r: r[key])
    worst = max(rows, key=lambda r: r[key])
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "best_case": (best["dataset"], best["task_id"], best["turn_index"], best[key]),
        "worst_case": (worst["dataset"], worst["task_id"], worst["turn_index"], worst[key]),
    }


RESOLVERS = {
    "aries": lambda base, task_id: f"{base}/{task_id}/harness/telemetry",
    "fanout": lambda base, task_id, _cache={}: _fanout_resolver(base, task_id, _cache),
}


def _fanout_resolver(base, task_id, cache):
    if base not in cache:
        mapping = {}
        for manifest_path in glob.glob(f"{base}/cells/*/runs/*/*/manifest.json"):
            m = json.load(open(manifest_path))
            mapping[m["instance_id"]] = os.path.join(os.path.dirname(manifest_path), "conversation", "sessions")
        cache[base] = mapping
    return cache[base].get(task_id)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help='"<kind>:<base_dir>:<results_glob>", repeatable')
    ap.add_argument("--base-url", default="http://localhost:8100/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--out", default=None, help="optional path to dump all per-turn rows as JSON")
    args = ap.parse_args()

    counter = TokenCounter(args.base_url, args.model)
    all_rows = []
    for src in args.source:
        kind, base, results_glob = src.split(":", 2)
        rows = token_ratios_for_source(kind, base, results_glob, RESOLVERS[kind], counter)
        all_rows.extend(rows)
        print(f"{kind}: {len(rows)} stable turns processed")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_rows, f)
        print(f"wrote {len(all_rows)} rows to {args.out}")

    for metric in ("token_ratio", "unit_ratio"):
        s = summarize(all_rows, metric)
        print(f"\n=== {metric} across {s['n']} stable turns ===")
        print(f"mean={s['mean']:.3f}  median={s['median']:.3f}")
        bd, bt, bi, bv = s["best_case"]
        wd, wt, wi, wv = s["worst_case"]
        print(f"best  (sparsest): {bd} {bt} turn {bi} -> {bv:.3f}")
        print(f"worst (densest):  {wd} {wt} turn {wi} -> {wv:.3f}")
