"""Run the minimal-working-memory search across every task in an Agent_Bench-style
"eviction_fanout" run directory (cells/*/runs/<suite>/<hash>/conversation/sessions/),
as opposed to run_subset.py's ARIES-style <task>/harness/telemetry/ layout.

Usage:
  python run_eviction_fanout.py --run-dir <path-to-eviction_fanout-run-dir> \
      --base-url http://localhost:8100/v1 --model Qwen/Qwen3.6-35B-A3B-FP8
"""
import argparse
import glob
import json
import os
from datetime import datetime, timezone

from run_experiment import run


def discover_tasks(run_dir):
    tasks = []
    for manifest_path in sorted(glob.glob(f"{run_dir}/cells/*/runs/*/*/manifest.json")):
        task_dir = os.path.dirname(manifest_path)
        m = json.load(open(manifest_path))
        sessions_dir = os.path.join(task_dir, "conversation", "sessions")
        if os.path.isdir(sessions_dir):
            tasks.append((m["instance_id"], sessions_dir, m.get("status")))
    return tasks


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--base-url", default="http://localhost:8100/v1")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    p.add_argument("--api-key-env", default="SGLANG_API_KEY")
    p.add_argument("--turn-workers", type=int, default=3)
    p.add_argument("--calls-per-turn-workers", type=int, default=6)
    p.add_argument("--exact-match", action="store_true")
    p.add_argument("--tools", default="web_search")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--skip", default="", help="comma-separated task_ids to skip (e.g. already completed in a prior invocation)")
    args = p.parse_args()

    tool_filter = set(t.strip() for t in args.tools.split(",") if t.strip()) or None
    skip = set(t.strip() for t in args.skip.split(",") if t.strip())
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for task_id, sessions_dir, status in discover_tasks(args.run_dir):
        if task_id in skip:
            print(f"\n=== task {task_id}: skipped (already done) ===")
            continue
        print(f"\n=== task {task_id} (status={status}) ===")
        run(
            sessions_dir,
            task_id,
            args.base_url,
            args.model,
            args.api_key_env,
            f"results/eviction-fanout-run-{run_stamp}/{task_id}/turns.jsonl",
            args.turn_workers,
            args.calls_per_turn_workers,
            use_fuzzy=not args.exact_match,
            tool_filter=tool_filter,
            samples=args.samples,
        )
