#!/usr/bin/env python3
"""Extract every memory_search call made by the agent across one or more
ARIES swe-atlas-qa amem run directories into a structured dataset, joining
each call's query/results with the task's outcome (agg_score/reward) and,
when a --control run is supplied, its per-task delta against the control
arm.

This replaces hand-curating scripts/out/retrieval_relevance_cases.csv by
hand: per-result similarity scores and note ids are not estimates, they are
read verbatim from the toolResult message's structured `details.memories`
field (each entry: id, content, similarity [0..1], keywords, tags, links,
timestamp, via ["match"|"link"], note_type) recorded alongside the free-text
summary in harness-turn-01/telemetry/<sessionId>.jsonl. gateway.log's
one-line-per-call summary is used only as a latency/count cross-check.

This script does NOT judge relevance (i.e. whether a retrieved note actually
helped the answer) -- that requires reading answer.txt/rubrics against note
content and is left to a follow-up LLM-judge pass. It only makes the query
-> results -> outcome data available at scale, so patterns (e.g. does higher
sim_mean correlate with a better delta) can be examined across the whole
corpus instead of a handful of hand-picked cases.

Usage:
  scripts/extract_memory_retrieval_events.py \\
      --run runs/<pilot30-amem-repo-run> \\
      --run runs/<pilot30-amem-task-run> \\
      --control runs/<pilot30-control-run> \\
      --out-dir scripts/out
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import summarize_sweatlas_arms as ssa  # noqa: E402

import pandas as pd

ARM_SUFFIX_RE = re.compile(r"-amem-(repo|task|global)-")
REPLICATE_RE = re.compile(r"sweatlasqa-([a-z0-9]+(?:-shuffle\d+)?(?:-ctxlimit)?)")
GATEWAY_LOG_RE = re.compile(r'memory_search "(.*)" → (\d+) results \((\d+)ms\)')


def infer_arm(run_dir_name: str) -> str:
    m = ARM_SUFFIX_RE.search(run_dir_name)
    if not m:
        return "control"
    return {"repo": "repo-scope", "task": "task-scope", "global": "global-scope"}[m.group(1)]


def infer_replicate(run_dir_name: str) -> str:
    m = REPLICATE_RE.search(run_dir_name)
    return m.group(1) if m else run_dir_name


def find_telemetry_session_files(task_dir: Path):
    tel_dir = task_dir / "harness-turn-01" / "telemetry"
    if not tel_dir.is_dir():
        return []
    return sorted(
        p for p in tel_dir.glob("*.jsonl")
        if not p.name.endswith(".trajectory.jsonl")
    )


def iter_memory_search_calls(task_dir: Path):
    """Yields dicts with session_id, call_seq, query, n_results, n_via_match,
    n_via_link, notes (list of {note_id, similarity, content, timestamp,
    via}), in call order, for every memory_search call in this task dir."""
    events = []
    for session_path in find_telemetry_session_files(task_dir):
        session_id = session_path.stem
        tool_call_queries = {}  # toolCallId -> query
        call_order = []  # toolCallId in first-seen order
        tool_results = {}  # toolCallId -> (text, details)
        with session_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = record.get("message", {})
                role = message.get("role")
                if role == "assistant":
                    for item in message.get("content") or []:
                        if isinstance(item, dict) and item.get("type") == "toolCall" and item.get("name") == "memory_search":
                            tool_call_id = item["id"]
                            call_order.append(tool_call_id)
                            tool_call_queries[tool_call_id] = (item.get("arguments") or {}).get("query")
                elif role == "toolResult" and message.get("toolName") == "memory_search":
                    tool_call_id = message.get("toolCallId")
                    content = message.get("content") or []
                    text = content[0]["text"] if content else None
                    tool_results[tool_call_id] = (text, message.get("details") or {})

        for call_seq, tool_call_id in enumerate(call_order, start=1):
            query = tool_call_queries.get(tool_call_id)
            text, details = tool_results.get(tool_call_id, (None, {}))
            memories = details.get("memories") or []
            notes = [
                {
                    "note_id": m.get("id"),
                    "similarity": m.get("similarity"),
                    "content": m.get("content"),
                    "timestamp": m.get("timestamp"),
                    "via": m.get("via"),
                }
                for m in memories
            ]
            events.append({
                "session_id": session_id,
                "call_seq": call_seq,
                "query": query,
                "n_results": details.get("count", len(notes)),
                "n_via_match": sum(1 for n in notes if n["via"] == "match"),
                "n_via_link": sum(1 for n in notes if n["via"] == "link"),
                "notes": notes,
            })
    return events


def cross_check_gateway_log(task_dir: Path):
    """Returns a list of (query, n_results, latency_ms) parsed from
    gateway.log's one-line-per-call summaries, in file order."""
    log_path = task_dir / "harness-turn-01" / "gateway.log"
    if not log_path.is_file():
        return []
    out = []
    for line in log_path.read_text(errors="replace").splitlines():
        m = GATEWAY_LOG_RE.search(line)
        if m:
            out.append((m.group(1), int(m.group(2)), int(m.group(3))))
    return out


def load_note_origin_repos(run_dir: Path, task_meta: dict, scope: str, task_repository: str | None):
    """Best-effort note_id -> originating repository, for spotting cross-repo
    memory transfer under global scope. task-scope and repo-scope stores are
    single-repo by construction, so every note in them belongs to the task's
    own repository. global-scope pools notes from every repository in one
    file, so origin is inferred heuristically by matching repo name tokens
    against note content/keywords/tags; left null when no repo's tokens are
    found (ambiguous)."""
    if scope != "global-scope":
        return None  # caller falls back to task_repository for every note

    amem_dir = run_dir / "amem-memory"
    note_path = amem_dir / "amem-memory.json" if not amem_dir.is_dir() else None
    if note_path is None or not note_path.is_file():
        # global scope may still export under a single slug subdirectory
        candidates = list(amem_dir.glob("*/amem-memory.json")) if amem_dir.is_dir() else []
        note_path = candidates[0] if len(candidates) == 1 else None
    if note_path is None or not note_path.is_file():
        return {}

    repos = sorted({repo for repo, _ in task_meta.values()})
    repo_tokens = {repo: re.split(r"[/_-]+", repo.lower()) for repo in repos}

    notes = json.loads(note_path.read_text())
    origin = {}
    for note in notes:
        note_id = note.get("id")
        payload = note.get("payload", {})
        haystack = " ".join([
            payload.get("content", ""),
            " ".join(payload.get("keywords", []) or []),
            " ".join(payload.get("tags", []) or []),
        ]).lower()
        matched = [repo for repo, tokens in repo_tokens.items() if any(t and t in haystack for t in tokens)]
        origin[note_id] = matched[0] if len(matched) == 1 else None
    return origin


def mean_min_max(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, None
    return sum(values) / len(values), min(values), max(values)


def build_rows(run_dir: Path, task_meta: dict, control_evals: dict):
    arm = infer_arm(run_dir.name)
    replicate = infer_replicate(run_dir.name)
    events_rows = []
    notes_rows = []

    if arm == "control":
        return events_rows, notes_rows  # no memory_search calls possible

    note_origin_cache = {}

    for task_id, task_dir in ssa.find_task_dirs(run_dir):
        m = ssa.TASK_DIR_RE.match(task_dir.name)
        occurrence = int(m.group(2))
        evaln = ssa.load_evaluation(task_dir)
        repository, _base_commit = task_meta.get(task_id, (None, None))

        calls = iter_memory_search_calls(task_dir)
        if not calls:
            continue
        gw = cross_check_gateway_log(task_dir)

        if arm not in note_origin_cache:
            note_origin_cache[arm] = load_note_origin_repos(run_dir, task_meta, arm, repository)
        origin_map = note_origin_cache[arm] or {}

        for call in calls:
            sims = [n["similarity"] for n in call["notes"]]
            sim_mean, sim_min, sim_max = mean_min_max([s * 100 if s is not None else None for s in sims])
            gw_entry = gw[call["call_seq"] - 1] if call["call_seq"] - 1 < len(gw) else None

            control_evaln = control_evals.get((replicate, task_id)) if control_evals else None
            control_agg = control_evaln["agg_score"] if control_evaln else None
            agg_score = evaln["agg_score"] if evaln else None
            delta = (agg_score - control_agg) if (agg_score is not None and control_agg is not None) else None
            bucket = None
            if delta is not None:
                bucket = "improved" if delta > 0 else ("degraded" if delta < 0 else "tied")

            events_rows.append({
                "run_dir": run_dir.name,
                "replicate": replicate,
                "arm": arm,
                "task_id": task_id,
                "occurrence": occurrence,
                "repository": repository,
                "session_id": call["session_id"],
                "call_seq": call["call_seq"],
                "query": call["query"],
                "n_results": call["n_results"],
                "n_via_match": call["n_via_match"],
                "n_via_link": call["n_via_link"],
                "sim_mean": sim_mean,
                "sim_min": sim_min,
                "sim_max": sim_max,
                "note_ids": ";".join(n["note_id"] for n in call["notes"] if n["note_id"]),
                "gateway_log_query": gw_entry[0] if gw_entry else None,
                "gateway_log_n_results": gw_entry[1] if gw_entry else None,
                "gateway_log_latency_ms": gw_entry[2] if gw_entry else None,
                "agg_score": agg_score,
                "reward": evaln["reward"] if evaln else None,
                "control_agg_score": control_agg,
                "delta": delta,
                "bucket": bucket,
            })

            for rank, note in enumerate(call["notes"], start=1):
                source_repository = repository if arm != "global-scope" else origin_map.get(note["note_id"])
                content = note["content"] or ""
                notes_rows.append({
                    "run_dir": run_dir.name,
                    "task_id": task_id,
                    "occurrence": occurrence,
                    "call_seq": call["call_seq"],
                    "note_rank": rank,
                    "note_id": note["note_id"],
                    "similarity": note["similarity"] * 100 if note["similarity"] is not None else None,
                    "via": note["via"],
                    "note_text_snippet": content[:300],
                    "source_repository": source_repository,
                    "note_timestamp": note["timestamp"],
                })

    return events_rows, notes_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, type=Path, dest="runs",
                         help="amem run directory to extract from (repeatable)")
    parser.add_argument("--control", action="append", default=None, type=Path, dest="controls",
                         help="control (no amem) run directory, for per-task agg_score deltas "
                              "(repeatable; matched to amem runs by replicate, e.g. pilot30/shuffle1/shuffle2)")
    parser.add_argument("--qa-root", default=Path(".cache/swe-atlas-qa"), type=Path)
    parser.add_argument("--out-dir", default=Path("scripts/out"), type=Path)
    args = parser.parse_args()

    task_meta = ssa.load_task_metadata(args.qa_root)
    if not task_meta:
        raise SystemExit(f"no task metadata found under {args.qa_root} — wrong --qa-root?")

    control_evals = {}
    for control_dir in args.controls or []:
        control_replicate = infer_replicate(control_dir.name)
        for task_id, task_dir in ssa.find_task_dirs(control_dir):
            evaln = ssa.load_evaluation(task_dir)
            if evaln is not None:
                control_evals[(control_replicate, task_id)] = evaln

    all_events, all_notes = [], []
    for run_dir in args.runs:
        events_rows, notes_rows = build_rows(run_dir, task_meta, control_evals)
        all_events.extend(events_rows)
        all_notes.extend(notes_rows)
        print(f"{run_dir.name}: {len(events_rows)} memory_search calls", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events_df = pd.DataFrame(all_events).sort_values(["run_dir", "task_id", "occurrence", "call_seq"])
    notes_df = pd.DataFrame(all_notes).sort_values(["run_dir", "task_id", "occurrence", "call_seq", "note_rank"])

    events_path = args.out_dir / "memory_search_events.csv"
    notes_path = args.out_dir / "memory_search_notes.csv"
    events_df.to_csv(events_path, index=False)
    notes_df.to_csv(notes_path, index=False)
    print(f"wrote {len(events_df)} rows to {events_path}", file=sys.stderr)
    print(f"wrote {len(notes_df)} rows to {notes_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
