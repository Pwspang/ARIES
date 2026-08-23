"""Aggregate summary across a subset run's per-task turns.jsonl files.

Point --glob at a specific run, e.g. 'results/run-20260818T063000Z/*/turns.jsonl' --
results are timestamped per invocation (see run_subset.py) so multiple runs can coexist on disk.
"""
import argparse
import glob
import json

from analyze import label_kind
from collections import Counter


def analyze_all(pattern):
    files = sorted(glob.glob(pattern))
    all_records = []
    for f in files:
        all_records.extend(json.loads(line) for line in open(f))

    failed = [r for r in all_records if "error" in r]
    records = [r for r in all_records if "error" not in r]
    stable = [r for r in records if r["reproducible"]]
    agrees = [r for r in stable if r.get("oracle_matches_reference")]

    print(f"tasks: {len(files)}, turns tested: {len(all_records)}, failed: {len(failed)}")
    print(f"self-consistent (stable reference found): {len(stable)}/{len(records)}")
    print(f"of those, agrees with historical oracle action: {len(agrees)}/{len(stable)}")

    print("\nper-task breakdown:")
    print(f"{'task_id':>10}  {'turns':>6}  {'stable':>6}  {'avg_minimal_ratio':>18}")
    by_task = {}
    for r in records:
        by_task.setdefault(r["task_id"], []).append(r)
    for task_id, rs in by_task.items():
        rs_stable = [r for r in rs if r["reproducible"]]
        ratios = [r["minimal_units"] / r["total_units"] for r in rs_stable if r["total_units"]]
        avg_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
        print(f"{task_id:>10}  {len(rs):>6}  {len(rs_stable):>6}  {avg_ratio:>18.2f}")

    if not stable:
        print("\nno self-consistent turns -- nothing further to analyze")
        return

    print("\nturn-by-turn minimal sets (stable turns only):")
    for r in stable:
        print(f"  {r['task_id']} turn {r['turn_index']}: {r['minimal_units']}/{r['total_units']} -> {r['minimal_unit_labels']}")

    kind_counts = Counter()
    for r in stable:
        for label in r["minimal_unit_labels"]:
            kind_counts[label_kind(label)] += 1

    print("\ncontext-unit kinds appearing in minimal sets (across all stable turns, all tasks):")
    for kind, count in kind_counts.most_common():
        print(f"  {kind}: {count}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True, help="e.g. results/run-20260818T063000Z/*/turns.jsonl")
    args = p.parse_args()
    analyze_all(args.glob)
