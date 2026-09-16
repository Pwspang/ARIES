#!/usr/bin/env python3
"""For every memory_search call that failed to surface anything useful --
either a zero-result call, or a call whose returned notes were never judged
"relevant AND used" -- check whether the underlying memory store actually
held a note that would have helped, at the moment the call was made.

This distinguishes two root causes that look identical from the outside
(the agent got nothing useful) but require completely different fixes:

  retrieval_miss:  the store already contained a note that would have
                   helped, but amem's search (similarity threshold, ranking,
                   topK) didn't surface it -- fixable on the retrieval side.
  addition_gap:    nothing useful had been written to memory yet at that
                   point (store empty, or nothing in it is relevant) --
                   fixable only on the writing side (memory_add/consolidate
                   coverage/timing), not by tuning search.

Store locations by arm:
  task-scope:  <run_dir>/<task_id>-<occurrence:03d>/amem-memory.json
  repo-scope:  <run_dir>/amem-memory/<repo-slug>-*/amem-memory.json
  global-scope: <run_dir>/amem-memory/global/amem-memory.json

Usage:
  scripts/diagnose_retrieval_vs_addition.py \\
      --events scripts/out/memory_search_events_with_overlap.csv \\
      --relevance-judged scripts/out/memory_search_relevance_judged.csv \\
      --usage-judged scripts/out/memory_note_usage_judged.csv \\
      --runs-root runs \\
      --out scripts/out/retrieval_vs_addition.csv
"""
import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from extract_memory_retrieval_events import find_telemetry_session_files  # noqa: E402

DEFAULT_BASE_URL = "http://192.168.111.200:8100/v1"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
MAX_CANDIDATES = 40
MAX_CONTENT_CHARS = 220

JUDGE_SYSTEM_PROMPT = (
    "You are auditing a coding agent's memory system. It asked a search "
    "query and got nothing useful back. You will see the question it was "
    "really trying to answer, and a list of notes that were ALREADY SITTING "
    "in its memory store at that moment (but were not returned by the "
    "search). Judge whether ANY of these notes would actually have helped "
    "answer the question -- not just topically related, but genuinely "
    "useful for it. Respond with strict JSON only: "
    '{"better_candidate_existed": true|false, "note_id": "<id or null>", '
    '"reason": "<one sentence>"}'
)


def call_judge(base_url: str, model: str, question: str, candidates: list[dict], timeout: int = 90) -> dict:
    listing = "\n".join(f"- id={c['id']}: {c['content'][:MAX_CONTENT_CHARS]}" for c in candidates)
    user_prompt = f"QUESTION:\n{question}\n\nNOTES ALREADY IN MEMORY (not returned):\n{listing}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    text = body["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"better_candidate_existed": None, "note_id": None, "reason": f"unparseable: {text!r}"}
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"better_candidate_existed": None, "note_id": None, "reason": f"unparseable JSON: {text!r}"}
    return {
        "better_candidate_existed": parsed.get("better_candidate_existed"),
        "note_id": parsed.get("note_id"),
        "reason": parsed.get("reason"),
    }


def slugify_repo(repository: str) -> str:
    return repository.replace("/", "-")


def resolve_store_path(runs_root: Path, run_dir: str, arm: str, task_id: str, occurrence: int,
                        repository: str | None) -> Path | None:
    run_path = runs_root / run_dir
    if arm == "task-scope":
        p = run_path / f"{task_id}-{int(occurrence):03d}" / "amem-memory.json"
        return p if p.is_file() else None
    if arm == "global-scope":
        p = run_path / "amem-memory" / "global" / "amem-memory.json"
        return p if p.is_file() else None
    if arm == "repo-scope":
        base = run_path / "amem-memory"
        if not base.is_dir() or not repository:
            return None
        slug = slugify_repo(repository)
        matches = sorted(base.glob(f"{slug}-*")) + sorted(base.glob(slug)) \
            if False else sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith(slug))
        for m in matches:
            f = m / "amem-memory.json"
            if f.is_file():
                return f
        return None
    return None


def parse_ts(ts: str | None):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_store_notes(store_path: Path, before: datetime | None, exclude_ids: set[str]) -> list[dict]:
    try:
        raw = json.loads(store_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    out = []
    for entry in raw:
        note_id = entry.get("id")
        if note_id in exclude_ids:
            continue
        payload = entry.get("payload", {})
        ts = parse_ts(payload.get("timestamp"))
        if before is not None and ts is not None and ts > before:
            continue
        content = payload.get("content") or ""
        if not content:
            continue
        out.append({"id": note_id, "content": content, "timestamp": payload.get("timestamp")})
    return out


def find_call_timestamp(task_dir: Path, call_seq: int) -> datetime | None:
    for session_path in find_telemetry_session_files(task_dir):
        idx = 0
        with session_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = rec.get("message", {})
                if msg.get("role") == "toolResult" and msg.get("toolName") == "memory_search":
                    idx += 1
                    if idx == call_seq:
                        return parse_ts(rec.get("timestamp"))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=Path, default=Path("scripts/out/memory_search_events_with_overlap.csv"))
    parser.add_argument("--relevance-judged", type=Path, default=Path("scripts/out/memory_search_relevance_judged.csv"))
    parser.add_argument("--usage-judged", type=Path, default=Path("scripts/out/memory_note_usage_judged.csv"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("scripts/out/retrieval_vs_addition.csv"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N failing calls (debugging)")
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    rel = pd.read_csv(args.relevance_judged)[["run_dir", "task_id", "occurrence", "call_seq", "note_rank", "llm_relevant"]]
    use = pd.read_csv(args.usage_judged)[["run_dir", "task_id", "occurrence", "call_seq", "note_rank", "llm_used"]]
    notes_judged = rel.merge(use, on=["run_dir", "task_id", "occurrence", "call_seq", "note_rank"])

    call_key = ["run_dir", "task_id", "occurrence", "call_seq"]
    notes_judged["note_ok"] = (notes_judged["llm_relevant"] == True) & (notes_judged["llm_used"] == True)  # noqa: E712
    call_ok = notes_judged.groupby(call_key, as_index=False)["note_ok"].any().rename(columns={"note_ok": "call_succeeded"})

    events = events.merge(call_ok, on=call_key, how="left")
    events["call_succeeded"] = events["call_succeeded"].fillna(False).astype(bool)
    failing = events[~events["call_succeeded"]].copy()
    if args.limit:
        failing = failing.head(args.limit)

    print(f"{len(events)} total calls, {len(failing)} failing (zero-result or no relevant+used note)", file=sys.stderr)

    results = []
    for i, row in failing.iterrows():
        note_ids = set((row.get("note_ids") or "").split(";")) if isinstance(row.get("note_ids"), str) else set()
        task_dir = args.runs_root / row["run_dir"] / f"{row['task_id']}-{int(row['occurrence']):03d}"
        store_path = resolve_store_path(
            args.runs_root, row["run_dir"], row["arm"], row["task_id"], int(row["occurrence"]), row.get("repository")
        )
        call_ts = find_call_timestamp(task_dir, int(row["call_seq"])) if task_dir.is_dir() else None

        if store_path is None:
            outcome, verdict = "no_store_found", {"better_candidate_existed": None, "note_id": None, "reason": "store file not found"}
        else:
            candidates = load_store_notes(store_path, call_ts, note_ids)[:MAX_CANDIDATES]
            if not candidates:
                outcome, verdict = "addition_gap", {"better_candidate_existed": False, "note_id": None, "reason": "no candidate notes existed in store at call time"}
            else:
                question = row.get("question")
                if not isinstance(question, str):
                    outcome, verdict = "no_question", {"better_candidate_existed": None, "note_id": None, "reason": "missing question text"}
                else:
                    try:
                        verdict = call_judge(args.base_url, args.model, question, candidates)
                    except Exception as e:  # noqa: BLE001
                        verdict = {"better_candidate_existed": None, "note_id": None, "reason": f"judge call failed: {e}"}
                    outcome = (
                        "retrieval_miss" if verdict["better_candidate_existed"] is True
                        else "addition_gap" if verdict["better_candidate_existed"] is False
                        else "unknown"
                    )

        print(f"[{len(results) + 1}/{len(failing)}] {row['run_dir']} / {row['task_id']} call={row['call_seq']} "
              f"arm={row['arm']} n_results={row['n_results']} -> {outcome}", file=sys.stderr)
        results.append({
            "run_dir": row["run_dir"], "task_id": row["task_id"], "occurrence": row["occurrence"],
            "call_seq": row["call_seq"], "arm": row["arm"], "n_results": row["n_results"],
            "query": row["query"], "question": row.get("question"),
            "outcome": outcome, "note_id": verdict.get("note_id"), "reason": verdict.get("reason"),
        })

    out_df = pd.DataFrame(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"wrote {len(out_df)} rows to {args.out}", file=sys.stderr)

    print("\n=== Root cause of failing calls ===")
    print(out_df["outcome"].value_counts())
    print("\n=== By arm ===")
    print(pd.crosstab(out_df["arm"], out_df["outcome"]))


if __name__ == "__main__":
    main()
