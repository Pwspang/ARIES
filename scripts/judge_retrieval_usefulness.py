#!/usr/bin/env python3
"""LLM-judge usefulness pass, control vs both -- scoped to notes already
judged RELEVANT by judge_retrieval_confusion.py (an irrelevant note is
essentially never going to be "used", so scoring the full 1566-note set for
usefulness would mostly re-confirm that; this answers the sharper question:
of what was relevant, how much actually drove the agent's subsequent actions
or final answer?).

Reuses judge_memory_note_usage.py's judge prompt/trajectory-reconstruction
machinery (extract_call_contexts/render_trajectory/call_judge) unchanged, and
judge_retrieval_confusion.py's truncation-proof note extractor (re-run here,
cheaply, purely to recover each relevant note's call_seq/run_dir -- fields
that didn't exist in the earlier relevance-judging pass's output CSV).

Usage:
  scripts/judge_retrieval_usefulness.py \\
      --relevance-csv scripts/out_retrieval_filter/retrieval_relevance_judged.csv \\
      --out-dir scripts/out_retrieval_filter
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import judge_retrieval_confusion as jrc  # noqa: E402
from judge_memory_note_usage import (  # noqa: E402
    call_judge, extract_call_contexts, render_trajectory, MAX_FINAL_ANSWER_CHARS,
    DEFAULT_BASE_URL, DEFAULT_MODEL,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relevance-csv", type=Path,
                         default=REPO_ROOT / "scripts" / "out_retrieval_filter" / "retrieval_relevance_judged.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "scripts" / "out_retrieval_filter")
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    relevant_keys = set()
    with open(args.relevance_csv) as f:
        for row in csv.DictReader(f):
            if row["llm_relevant"] == "True":
                relevant_keys.add((row["arm"], row["task_id"], row["occurrence"], row["note_id"], row["rank"]))
    print(f"loaded {len(relevant_keys)} relevant-note keys from {args.relevance_csv}", file=sys.stderr)

    qa_root = REPO_ROOT / ".cache" / "swe-atlas-qa"
    fresh_rows = []
    for arm, run_dirs in jrc.RUNS.items():
        for rd in run_dirs:
            fresh_rows.extend(jrc.extract_notes_from_run(REPO_ROOT / rd, arm, qa_root))

    notes = [
        r for r in fresh_rows
        if (r["arm"], r["task_id"], r["occurrence"], r["note_id"], str(r["rank"])) in relevant_keys
    ]
    # de-dupe (same key logic as judge_retrieval_confusion.py)
    seen = set()
    deduped = []
    for r in notes:
        key = (r["arm"], r["task_id"], r["occurrence"], r["note_id"], r["rank"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    notes = deduped
    print(f"matched {len(notes)} relevant notes with call_seq/run_dir recovered", file=sys.stderr)

    context_cache: dict[tuple, dict] = {}
    results = []
    for i, row in enumerate(notes):
        call_key = (row["run_dir"], row["task_id"], row["occurrence"], row["call_seq"])
        cache_key = (row["run_dir"], row["task_id"], row["occurrence"])
        if cache_key not in context_cache:
            task_dir = args.runs_root / row["run_dir"] / f"{row['task_id']}-{int(row['occurrence']):03d}"
            context_cache[cache_key] = extract_call_contexts(task_dir) if task_dir.is_dir() else {}
        ctx = context_cache[cache_key].get(row["call_seq"], {
            "exec_commands": [], "reasoning_snippets": [], "final_answer": None,
        })
        trajectory = render_trajectory(ctx)
        final_answer = (ctx["final_answer"] or "")[:MAX_FINAL_ANSWER_CHARS]

        try:
            verdict = call_judge(args.base_url, args.model, row["question"], row["note_content"], trajectory, final_answer)
        except Exception as e:  # noqa: BLE001
            verdict = {"used": None, "confidence": None, "evidence": f"judge call failed: {e}"}

        print(f"[{i + 1}/{len(notes)}] arm={row['arm']} task={row['task_id']} via={row['via']} "
              f"-> used={verdict['used']} ({verdict['confidence']})", file=sys.stderr)
        results.append({**row, "llm_used": verdict["used"], "llm_used_confidence": verdict["confidence"],
                         "llm_used_evidence": verdict["evidence"]})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "retrieval_usefulness_judged.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {len(results)} rows to {out_path}")

    print("\n=== Used rate by arm (among relevant notes) ===")
    from collections import defaultdict
    counts = defaultdict(lambda: defaultdict(int))
    for r in results:
        counts[r["arm"]][r["llm_used"]] += 1
    for arm, d in counts.items():
        total = sum(d.values())
        used = d.get(True, 0)
        print(f"{arm}: n={total} used={used} ({used/total:.1%}) not_used={d.get(False,0)} unparsed={d.get(None,0)}")

    print("\n=== Used rate by arm x via (among relevant notes) ===")
    counts2 = defaultdict(lambda: defaultdict(int))
    for r in results:
        counts2[(r["arm"], r["via"])][r["llm_used"]] += 1
    for key in sorted(counts2):
        d = counts2[key]
        total = sum(d.values())
        used = d.get(True, 0)
        print(f"{key}: n={total} used={used} ({used/total:.1%})")


if __name__ == "__main__":
    main()
