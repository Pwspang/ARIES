"""Summarize results/<task_id>/turns.jsonl: minimal-context size vs turn index, and which
context-unit types survive minimization most often.
"""
import argparse
import json
from collections import Counter


def label_kind(label):
    if label == "system_prompt":
        return "system_prompt"
    if label.startswith("compaction@"):
        return "compaction"
    if label.startswith("call@"):
        return "tool_call"
    if label.startswith("result@"):
        return "tool_output"
    if label.startswith("user@"):
        return "user"
    return "other"


def analyze(path):
    all_records = [json.loads(line) for line in open(path)]
    failed = [r for r in all_records if "error" in r]
    records = [r for r in all_records if "error" not in r]
    reproducible = [r for r in records if r["reproducible"]]  # "reproducible" = the model's own full-context answer was self-consistent
    agrees_with_oracle = [r for r in reproducible if r.get("oracle_matches_reference")]

    print(f"turns: {len(all_records)}, failed: {len(failed)}, self-consistent (stable reference found): {len(reproducible)}/{len(records)}")
    print(f"of those, reference action agrees with the historical oracle action: {len(agrees_with_oracle)}/{len(reproducible)}")
    for r in failed:
        print(f"  turn {r['turn_index']}: FAILED -- {r['error']}")

    print("\noracle (historical) vs. reference (this run's self-consistent answer) per turn:")
    for r in records:
        marker = "STABLE" if r["reproducible"] else "no consensus"
        agree = " [agrees w/ oracle]" if r.get("oracle_matches_reference") else ""
        print(f"--- turn {r['turn_index']} [{marker}, agreement={r.get('agreement_fraction')}]{agree} ---")
        print(f"  oracle:    {r.get('oracle_command', '?')}")
        print(f"  reference: {r.get('reference_command', '?')}")

    if not reproducible:
        print("\nno self-consistent turns -- nothing further to analyze")
        return

    print("\nturn_index  total_units  minimal_units  ratio  pinned (always kept, not searched)")
    for r in reproducible:
        ratio = r["minimal_units"] / r["total_units"] if r["total_units"] else 0
        pinned = ", ".join(r.get("pinned_unit_labels") or []) or "(none -- turn 0)"
        print(f"{r['turn_index']:>10}  {r['total_units']:>11}  {r['minimal_units']:>14}  {ratio:.2f}  {pinned}")

    kind_counts = Counter()
    for r in reproducible:
        for label in r["minimal_unit_labels"]:
            kind_counts[label_kind(label)] += 1

    print("\ncontext-unit kinds appearing in minimal sets (across all reproducible turns):")
    for kind, count in kind_counts.most_common():
        print(f"  {kind}: {count}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--turns", required=True, help="path to results/<task_id>/turns.jsonl")
    args = p.parse_args()
    analyze(args.turns)
