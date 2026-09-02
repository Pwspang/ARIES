"""Build turn-dependency graphs (dependency_graph.build_task_graph) for every task across every
eviction-fanout results/ directory, matched back to its source telemetry via the same
discover_tasks() mapping run_eviction_fanout.py used to produce them, and dump one combined JSON.

Usage:
  python build_all_dependency_graphs.py --run-dir <eviction_fanout run dir> \
      --results-glob 'results/eviction-fanout-run-*' --out all_dependency_graphs.json
"""
import argparse
import glob
import json
import os

from dependency_graph import build_task_graph
from run_eviction_fanout import discover_tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--results-glob", default="results/eviction-fanout-run-*")
    p.add_argument("--out", default="all_dependency_graphs.json")
    args = p.parse_args()

    task_to_telemetry = {task_id: (sessions_dir, status) for task_id, sessions_dir, status in discover_tasks(args.run_dir)}

    combined = {}
    for turns_jsonl in sorted(glob.glob(f"{args.results_glob}/*/turns.jsonl")):
        task_id = os.path.basename(os.path.dirname(turns_jsonl))
        entry = task_to_telemetry.get(task_id)
        if entry is None:
            print(f"skip {task_id}: no matching telemetry dir under {args.run_dir}")
            continue
        telemetry_dir, status = entry
        print(f"{task_id}: {turns_jsonl} <- {telemetry_dir}")
        combined[task_id] = {"status": status, **build_task_graph(telemetry_dir, turns_jsonl)}

    with open(args.out, "w") as f:
        json.dump(combined, f)
    print(f"wrote {args.out}: {len(combined)} tasks")


if __name__ == "__main__":
    main()
