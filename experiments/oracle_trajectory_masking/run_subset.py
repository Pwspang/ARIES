"""Run the minimal-working-memory search across every task in a subset run directory.

Usage:
  python run_subset.py --run-dir <path-to-run-dir> \
      --base-url http://localhost:8100/v1 --model Qwen/Qwen3.6-35B-A3B-FP8

Tasks are processed one at a time (each task's own turn/call concurrency still applies) to
avoid piling 10 tasks' worth of concurrent requests onto the shared model server at once.
"""
import argparse
import os
from datetime import datetime, timezone

from run_experiment import run


def discover_task_dirs(run_dir):
    task_dirs = []
    for name in sorted(os.listdir(run_dir)):
        telemetry_dir = os.path.join(run_dir, name, "harness", "telemetry")
        if os.path.isdir(telemetry_dir):
            task_dirs.append((name, telemetry_dir))
    return task_dirs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--base-url", default="http://localhost:8100/v1")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    p.add_argument("--api-key-env", default="SGLANG_API_KEY")
    p.add_argument("--turn-workers", type=int, default=4)
    p.add_argument("--calls-per-turn-workers", type=int, default=6)
    p.add_argument("--exact-match", action="store_true")
    p.add_argument("--tools", default="web_search")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--timeout", type=int, default=300, help="per-request timeout in seconds")
    args = p.parse_args()

    tool_filter = set(t.strip() for t in args.tools.split(",") if t.strip()) or None

    # one timestamped folder shared by every task in this subset run, so re-running never
    # overwrites a previous run's results
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for task_id, telemetry_dir in discover_task_dirs(args.run_dir):
        print(f"\n=== task {task_id} ===")
        run(
            telemetry_dir,
            task_id,
            args.base_url,
            args.model,
            args.api_key_env,
            f"results/run-{run_stamp}/{task_id}/turns.jsonl",
            args.turn_workers,
            args.calls_per_turn_workers,
            use_fuzzy=not args.exact_match,
            tool_filter=tool_filter,
            samples=args.samples,
            timeout=args.timeout,
        )
