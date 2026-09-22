#!/usr/bin/env python3
"""Compare a repo-scoped amem run (fact memory only) against a repo-scoped
amem run with the episodic failure-log bootstrap additionally enabled ---
the H5/H6 comparison in the "episodic memory study" section of
docs/benchmarks/swe-atlas-qa.md.

Reads only what's already on disk, no instrumentation required:
  - <run>/task-*/evaluation/evaluation_results.json  (agg_score, reward, ...)
  - <run>/task-*/harness-turn-01/telemetry/sessions.json  (token/runtime cost)
  - <run>/task-*/harness-turn-01/telemetry/<session>.jsonl  (tool-call trace:
    memory_add_episodic calls, and exec calls that returned a non-zero exit)
  - .cache/swe-atlas-qa/data/qa/<task-id>/task.toml [metadata]
    (repository, base_commit -> position-within-repo)

For each task this derives:
  - n_episodic_writes: memory_add_episodic calls made during the task.
  - n_exec_errors: exec calls whose result had a non-zero exitCode.
  - n_repeated_exec_errors: of those, how many match a (command, exitCode)
    signature already seen in an *earlier* same-repo task in this run --
    the direct, mechanistic counterpart to H5/H6's aggregate-score claim
    (a repeated failure is a failure the agent already knew about, from its
    own or an earlier task's episodic log).

"Position within repo" and the reference task order are derived from the
--repo-scope run's own task-directory suffixes, mirroring
summarize_sweatlas_arms.py's assign_positions -- the two runs are expected to
share an identical task list/order by design (only harness.amem.episodic_bootstrap
differs between the arms).

Usage:
  scripts/analyze_episodic_transfer.py \\
      --repo-scope runs/<pilot30-amem-repo-run> \\
      --episodic runs/<pilot30-amem-repo-episodic-run> \\
      --qa-root .cache/swe-atlas-qa
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import summarize_sweatlas_arms as ssa  # noqa: E402

ARM_NAMES = ("repo-scope", "episodic")


def find_telemetry_session_files(task_dir: Path):
    tel_dir = task_dir / "harness-turn-01" / "telemetry"
    if not tel_dir.is_dir():
        return []
    return sorted(p for p in tel_dir.glob("*.jsonl") if not p.name.endswith(".trajectory.jsonl"))


def exec_error_signature(command: str, exit_code) -> tuple[str, object]:
    """Normalizes an errored exec call to a (command, exit_code) signature.
    Whitespace-collapsed rather than fuzzy-matched: this is meant to catch
    the same command literally repeated (the case episodic memory should
    directly prevent), not a semantically-similar-but-different command."""
    return (" ".join((command or "").split()), exit_code)


def collect_task_events(task_dir: Path) -> dict:
    """Returns {'n_episodic_writes': int, 'error_signatures': list[(command, exit_code)]}
    for one task occurrence, across every session transcript it wrote."""
    n_episodic_writes = 0
    error_signatures: list[tuple[str, object]] = []
    for session_path in find_telemetry_session_files(task_dir):
        exec_commands: dict[str, str] = {}  # toolCallId -> command
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
                        if not isinstance(item, dict) or item.get("type") != "toolCall":
                            continue
                        if item.get("name") == "memory_add_episodic":
                            n_episodic_writes += 1
                        elif item.get("name") == "exec":
                            exec_commands[item["id"]] = (item.get("arguments") or {}).get("command")
                elif role == "toolResult" and message.get("toolName") == "exec":
                    tool_call_id = message.get("toolCallId")
                    details = message.get("details") or {}
                    exit_code = details.get("exitCode")
                    if exit_code not in (0, None):
                        command = exec_commands.get(tool_call_id)
                        error_signatures.append(exec_error_signature(command, exit_code))
    return {"n_episodic_writes": n_episodic_writes, "error_signatures": error_signatures}


def collect_arm(run_dir: Path, task_meta: dict) -> list[dict]:
    rows = []
    for task_id, task_dir in ssa.find_task_dirs(run_dir):
        evaln = ssa.load_evaluation(task_dir)
        if evaln is None:
            print(f"warning: no evaluation_results.json under {task_dir}", file=sys.stderr)
            continue
        tokens = ssa.load_session_tokens(task_dir) or {}
        events = collect_task_events(task_dir)
        repository, base_commit = task_meta.get(task_id, (None, None))
        rows.append({
            "task_id": task_id,
            "repository": repository,
            "base_commit": base_commit,
            **evaln,
            **tokens,
            **events,
        })
    return rows


def annotate_repeated_failures(rows: list[dict], repo_order: dict[str, list[str]]) -> None:
    """Mutates rows in place: adds 'position' (per assign_positions) and
    'n_repeated_exec_errors' -- of a task's own error_signatures, how many
    were already seen in an earlier-position same-repo task *within this
    same arm's run*. Rows are visited in position order per repository so
    "earlier" only ever means "ran earlier in this run", not future tasks."""
    ssa.assign_positions(rows, repo_order)
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["repository"]:
            by_repo[row["repository"]].append(row)
    for repo_rows in by_repo.values():
        repo_rows.sort(key=lambda r: r["position"] or 0)
        seen: set[tuple[str, object]] = set()
        for row in repo_rows:
            signatures = row["error_signatures"]
            row["n_repeated_exec_errors"] = sum(1 for s in signatures if s in seen)
            row["n_exec_errors"] = len(signatures)
            seen.update(signatures)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(value, spec):
    return format(value, spec) if value is not None else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-scope", required=True, type=Path, help="repo-scoped amem run (fact memory only)")
    parser.add_argument("--episodic", required=True, type=Path, help="repo-scoped amem run with episodic_bootstrap enabled")
    parser.add_argument("--qa-root", default=Path(".cache/swe-atlas-qa"), type=Path)
    args = parser.parse_args()

    task_meta = ssa.load_task_metadata(args.qa_root)
    if not task_meta:
        raise SystemExit(f"no task metadata found under {args.qa_root} -- wrong --qa-root?")

    run_dirs = {"repo-scope": args.repo_scope, "episodic": args.episodic}

    repo_order: dict[str, list[str]] = defaultdict(list)
    for task_id, _ in ssa.find_task_dirs(args.repo_scope):
        repository, _ = task_meta.get(task_id, (None, None))
        if repository:
            repo_order[repository].append(task_id)

    arms = {name: collect_arm(path, task_meta) for name, path in run_dirs.items()}
    for rows in arms.values():
        annotate_repeated_failures(rows, repo_order)

    print(f"{'repo':<30} {'pos':>3} {'arm':<10} {'n':>3} {'agg_score':>10} {'episodic_wr':>11} {'exec_err':>9} {'repeat_err':>10}")
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for arm_name, rows in arms.items():
        for row in rows:
            buckets[(row["repository"], row["position"], arm_name)].append(row)
    for key in sorted(buckets.keys(), key=lambda k: (str(k[0]), k[1] or 0, k[2])):
        repo, pos, arm_name = key
        rows = buckets[key]
        print(
            f"{str(repo):<30} {str(pos):>3} {arm_name:<10} {len(rows):>3} "
            f"{fmt(mean([r['agg_score'] for r in rows]), '10.3f')} "
            f"{fmt(mean([r['n_episodic_writes'] for r in rows]), '11.1f')} "
            f"{fmt(mean([r['n_exec_errors'] for r in rows]), '9.1f')} "
            f"{fmt(mean([r['n_repeated_exec_errors'] for r in rows]), '10.1f')}"
        )

    print()
    print("=== Pooled position>=2 (the only tasks that could benefit from repo-scope sharing) ===")
    pooled: dict[str, list[dict]] = defaultdict(list)
    for arm_name, rows in arms.items():
        for row in rows:
            if row["position"] and row["position"] >= 2:
                pooled[arm_name].append(row)
    for arm_name in ARM_NAMES:
        rows = pooled.get(arm_name, [])
        print(
            f"{arm_name:<10} n={len(rows):<4} "
            f"mean_agg_score={fmt(mean([r['agg_score'] for r in rows]), '.3f')} "
            f"mean_repeated_exec_errors={fmt(mean([r['n_repeated_exec_errors'] for r in rows]), '.2f')} "
            f"mean_exec_errors={fmt(mean([r['n_exec_errors'] for r in rows]), '.2f')} "
            f"mean_episodic_writes={fmt(mean([r['n_episodic_writes'] for r in rows]), '.2f')}"
        )

    print()
    print("=== Paired per-task deltas (position>=2, episodic minus repo-scope) ===")
    by_task = {"repo-scope": {r["task_id"]: r for r in arms["repo-scope"]},
               "episodic": {r["task_id"]: r for r in arms["episodic"]}}
    common_tasks = sorted(set(by_task["repo-scope"]) & set(by_task["episodic"]))
    score_deltas, repeat_deltas = [], []
    for task_id in common_tasks:
        control_row = by_task["repo-scope"][task_id]
        episodic_row = by_task["episodic"][task_id]
        if not control_row["position"] or control_row["position"] < 2:
            continue
        score_deltas.append(episodic_row["agg_score"] - control_row["agg_score"])
        repeat_deltas.append(episodic_row["n_repeated_exec_errors"] - control_row["n_repeated_exec_errors"])
    print(f"n={len(score_deltas)} mean_agg_score_delta={fmt(mean(score_deltas), '+.3f')} "
          f"mean_repeated_exec_error_delta={fmt(mean(repeat_deltas), '+.2f')}")


if __name__ == "__main__":
    main()
