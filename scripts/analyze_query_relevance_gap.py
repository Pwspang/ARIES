#!/usr/bin/env python3
"""Test the hypothesis that amem's similarity score conflates *topical
similarity* with *task relevance* -- i.e. a memory_search call can return
notes that score well on cosine similarity (they're about the same repo /
same general area) while having little actual overlap with the specific
question the agent was asked, and that this gap correlates with worse
outcomes (degraded agg_score vs the control arm).

Reads scripts/out/memory_search_events.csv / memory_search_notes.csv
(produced by extract_memory_retrieval_events.py) and, for every call,
extracts the task's actual <question>...</question> text from telemetry and
computes a cheap lexical-overlap proxy for "is this note actually about what
was asked" independent of amem's own similarity score. It is NOT a
replacement for an LLM-judge relevance pass -- it's a fast, free first-pass
signal to see whether the hypothesis is worth pursuing before building one.

Usage:
  scripts/analyze_query_relevance_gap.py \\
      --run runs/<amem-run> [--run runs/<amem-run2> ...] \\
      --events scripts/out/memory_search_events.csv \\
      --notes scripts/out/memory_search_notes.csv
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import summarize_sweatlas_arms as ssa  # noqa: E402
from extract_memory_retrieval_events import find_telemetry_session_files  # noqa: E402

import pandas as pd

QUESTION_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL)
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z_']{3,}")
STOPWORDS = {
    "this", "that", "with", "from", "have", "what", "when", "where", "which",
    "does", "your", "into", "about", "would", "could", "should", "there",
    "their", "these", "those", "being", "been", "were", "them", "then",
    "want", "wondering", "trying", "understand", "main", "concerns", "here",
    "also", "some", "such", "only", "over", "each", "while", "will", "code",
    "codebase", "repository", "question", "please", "make", "sure", "actual",
    "actually", "happens", "happen", "need", "looking", "interested", "know",
}


def tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return {t.lower() for t in TOKEN_RE.findall(text)} - STOPWORDS


def extract_question(task_dir: Path) -> str | None:
    for session_path in find_telemetry_session_files(task_dir):
        with session_path.open() as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = record.get("message", {})
                if message.get("role") == "user" and isinstance(message.get("content"), str):
                    m = QUESTION_RE.search(message["content"])
                    if m:
                        return m.group(1)
        break  # first session file only; the question is asked once, up front
    return None


def jaccard(a: set[str], b: set[str]) -> float | None:
    if not a or not b:
        return None
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, type=Path, dest="runs")
    parser.add_argument("--events", type=Path, default=Path("scripts/out/memory_search_events.csv"))
    parser.add_argument("--notes", type=Path, default=Path("scripts/out/memory_search_notes.csv"))
    args = parser.parse_args()

    # (run_dir_name, task_id) -> question text
    questions: dict[tuple[str, str], str] = {}
    for run_dir in args.runs:
        for task_id, task_dir in ssa.find_task_dirs(run_dir):
            q = extract_question(task_dir)
            if q:
                questions[(run_dir.name, task_id)] = q

    events = pd.read_csv(args.events)
    notes = pd.read_csv(args.notes)

    events["question"] = events.apply(lambda r: questions.get((r["run_dir"], r["task_id"])), axis=1)
    events["question_tokens"] = events["question"].apply(tokenize)
    events["query_tokens"] = events["query"].apply(tokenize)
    events["query_question_overlap"] = events.apply(
        lambda r: jaccard(r["question_tokens"], r["query_tokens"]), axis=1
    )

    notes = notes.merge(
        events[["run_dir", "task_id", "occurrence", "call_seq", "question_tokens", "bucket", "delta", "sim_mean"]],
        on=["run_dir", "task_id", "occurrence", "call_seq"], how="left",
    )
    notes["note_tokens"] = notes["note_text_snippet"].apply(tokenize)
    notes["note_question_overlap"] = notes.apply(
        lambda r: jaccard(r["question_tokens"], r["note_tokens"]), axis=1
    )

    call_overlap = notes.groupby(["run_dir", "task_id", "occurrence", "call_seq"], as_index=False).agg(
        note_question_overlap_mean=("note_question_overlap", "mean"),
        note_question_overlap_max=("note_question_overlap", "max"),
    )
    events = events.merge(call_overlap, on=["run_dir", "task_id", "occurrence", "call_seq"], how="left")

    print("=== Per-bucket comparison: amem similarity vs question-overlap proxy ===")
    print(f"{'bucket':<10} {'n':>4} {'mean_sim':>10} {'mean_overlap':>14} {'mean_query_q_overlap':>22} {'zero_result_rate':>18}")
    for bucket, g in events.groupby("bucket", dropna=False):
        zero_rate = (g["n_results"] == 0).mean()
        print(
            f"{str(bucket):<10} {len(g):>4} "
            f"{g['sim_mean'].mean():>10.2f} "
            f"{g['note_question_overlap_mean'].mean():>14.3f} "
            f"{g['query_question_overlap'].mean():>22.3f} "
            f"{zero_rate:>18.2%}"
        )

    print()
    print("=== Correlation: does higher amem similarity predict higher question-overlap? ===")
    corr = events[["sim_mean", "note_question_overlap_mean"]].dropna().corr().iloc[0, 1]
    print(f"corr(sim_mean, note_question_overlap_mean) = {corr:.3f}  (n={events[['sim_mean','note_question_overlap_mean']].dropna().shape[0]})")

    print()
    print("=== Candidate 'similar but not relevant' cases: sim_mean > 35 and overlap < 0.04, in degraded-bucket calls ===")
    flagged = events[
        (events["bucket"] == "degraded")
        & (events["sim_mean"] > 35)
        & (events["note_question_overlap_mean"].fillna(0) < 0.04)
    ]
    for _, row in flagged.iterrows():
        print(f"- {row['run_dir']} / {row['task_id']} occ={row['occurrence']} call={row['call_seq']} "
              f"delta={row['delta']:.3f} sim_mean={row['sim_mean']:.1f} overlap={row['note_question_overlap_mean']:.3f}")
        print(f"  query: {row['query']!r}")
        print(f"  question: {str(row['question'])[:200]!r}")

    events.drop(columns=["question_tokens", "query_tokens"]).to_csv(
        args.events.parent / "memory_search_events_with_overlap.csv", index=False
    )
    print(f"\nwrote {args.events.parent / 'memory_search_events_with_overlap.csv'}")


if __name__ == "__main__":
    main()
