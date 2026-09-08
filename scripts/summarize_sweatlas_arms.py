#!/usr/bin/env python3
"""Compare three ARIES swe-atlas-qa run directories (control / task-scope
amem / repo-scope amem) stratified by each task's position within its
repository — the study design in the "amem repo scope" plan.

Reads only what's already on disk, no instrumentation required:
  - <run>/task-*/evaluation/evaluation_results.json  (agg_score, reward, ...)
  - <run>/task-*/harness-turn-01/telemetry/sessions.json
    (inputTokens, outputTokens, totalTokens, cacheRead, cacheWrite, runtimeMs)
  - .cache/swe-atlas-qa/data/qa/<task-id>/task.toml [metadata]
    (repository, base_commit -> position-within-repo)

Usage:
  scripts/summarize_sweatlas_arms.py \\
      --control runs/<control-run> \\
      --task-scope runs/<task-scope-run> \\
      --repo-scope runs/<repo-scope-run>
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

TASK_DIR_RE = re.compile(r"^(task-[0-9a-f]+)-(\d+)$")
ARM_NAMES = ("control", "task-scope", "repo-scope")


def load_task_metadata(qa_root: Path) -> dict[str, tuple[str, str]]:
    """task_id -> (repository, base_commit), parsed straight out of each
    task.toml's [metadata] table without a TOML dependency (only two flat
    string fields are needed)."""
    meta: dict[str, tuple[str, str]] = {}
    qa_dir = qa_root / "data" / "qa"
    if not qa_dir.is_dir():
        return meta
    for task_dir in sorted(qa_dir.iterdir()):
        toml_path = task_dir / "task.toml"
        if not toml_path.is_file():
            continue
        repository = base_commit = None
        for line in toml_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("repository "):
                repository = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("base_commit "):
                base_commit = line.split("=", 1)[1].strip().strip('"')
        if repository and base_commit:
            meta[task_dir.name] = (repository, base_commit)
    return meta


def find_task_dirs(run_dir: Path):
    """Yields (task_id, task_dir_path) for every task-*-NNN directory,
    stripping the trailing occurrence suffix to recover the bare task ID
    that joins against task.toml's directory name."""
    if not run_dir.is_dir():
        raise SystemExit(f"not a run directory: {run_dir}")
    for entry in sorted(run_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = TASK_DIR_RE.match(entry.name)
        if m:
            yield m.group(1), entry


def load_session_tokens(task_dir: Path) -> dict | None:
    sessions_path = task_dir / "harness-turn-01" / "telemetry" / "sessions.json"
    if not sessions_path.is_file():
        return None
    data = json.loads(sessions_path.read_text())
    if not data:
        return None
    session = next(iter(data.values()))
    return {
        "inputTokens": session.get("inputTokens"),
        "outputTokens": session.get("outputTokens"),
        "totalTokens": session.get("totalTokens"),
        "cacheRead": session.get("cacheRead"),
        "cacheWrite": session.get("cacheWrite"),
        "runtimeMs": session.get("runtimeMs"),
    }


def load_evaluation(task_dir: Path) -> dict | None:
    eval_path = task_dir / "evaluation" / "evaluation_results.json"
    if not eval_path.is_file():
        return None
    data = json.loads(eval_path.read_text())
    return {
        "agg_score": data.get("agg_score"),
        "reward": data.get("reward"),
        "pass": data.get("pass"),
        "num_passed": data.get("num_passed"),
        "num_scored": data.get("num_scored"),
    }


def collect_arm(run_dir: Path, task_meta: dict) -> list[dict]:
    rows = []
    for task_id, task_dir in find_task_dirs(run_dir):
        evaln = load_evaluation(task_dir)
        if evaln is None:
            print(f"warning: no evaluation_results.json under {task_dir}", file=sys.stderr)
            continue
        tokens = load_session_tokens(task_dir) or {}
        repository, base_commit = task_meta.get(task_id, (None, None))
        if repository is None:
            print(f"warning: no task.toml metadata for {task_id}", file=sys.stderr)
        rows.append({
            "task_id": task_id,
            "repository": repository,
            "base_commit": base_commit,
            **evaln,
            **tokens,
        })
    return rows


def assign_positions(rows: list[dict], repo_order: dict[str, list[str]]) -> None:
    """Mutates rows in place, adding 1-based 'position' within its
    repository, per the fixed order supplied (see main: derived from the
    control arm's task-directory order, since all three arms are expected
    to share an identical task list/order by design)."""
    for row in rows:
        order = repo_order.get(row["repository"])
        if not order:
            row["position"] = None
            continue
        try:
            row["position"] = order.index(row["task_id"]) + 1
        except ValueError:
            row["position"] = None


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def fmt(value, spec):
    return format(value, spec) if value is not None else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control", required=True, type=Path, help="control (no amem) run directory")
    parser.add_argument("--task-scope", required=True, type=Path, help="task-scoped amem run directory")
    parser.add_argument("--repo-scope", required=True, type=Path, help="repo-scoped amem run directory")
    parser.add_argument("--qa-root", default=Path(".cache/swe-atlas-qa"), type=Path)
    args = parser.parse_args()

    task_meta = load_task_metadata(args.qa_root)
    if not task_meta:
        raise SystemExit(f"no task metadata found under {args.qa_root} — wrong --qa-root?")

    run_dirs = {"control": args.control, "task-scope": args.task_scope, "repo-scope": args.repo_scope}

    # "Position within repo" is defined by the control arm's task order —
    # all three arms are expected to share an identical task list/order by
    # design (the study invariant), so any one arm works as the reference.
    repo_order: dict[str, list[str]] = defaultdict(list)
    for task_id, _ in find_task_dirs(args.control):
        repository, _ = task_meta.get(task_id, (None, None))
        if repository:
            repo_order[repository].append(task_id)

    arms = {name: collect_arm(path, task_meta) for name, path in run_dirs.items()}
    for rows in arms.values():
        assign_positions(rows, repo_order)

    print(f"{'repo':<30} {'pos':>3} {'arm':<12} {'n':>3} {'agg_score':>10} {'inputTok':>12} {'outputTok':>10} {'runtimeMs':>10}")
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for arm_name, rows in arms.items():
        for row in rows:
            buckets[(row["repository"], row["position"], arm_name)].append(row)
    for key in sorted(buckets.keys(), key=lambda k: (str(k[0]), k[1] or 0, k[2])):
        repo, pos, arm_name = key
        rows = buckets[key]
        print(
            f"{str(repo):<30} {str(pos):>3} {arm_name:<12} {len(rows):>3} "
            f"{fmt(mean([r['agg_score'] for r in rows]), '10.3f')} "
            f"{fmt(mean([r.get('inputTokens') for r in rows]), '12.0f')} "
            f"{fmt(mean([r.get('outputTokens') for r in rows]), '10.0f')} "
            f"{fmt(mean([r.get('runtimeMs') for r in rows]), '10.0f')}"
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
            f"{arm_name:<12} n={len(rows):<4} "
            f"mean_agg_score={fmt(mean([r['agg_score'] for r in rows]), '.3f')} "
            f"mean_inputTokens={fmt(mean([r.get('inputTokens') for r in rows]), '.0f')} "
            f"mean_outputTokens={fmt(mean([r.get('outputTokens') for r in rows]), '.0f')} "
            f"mean_runtimeMs={fmt(mean([r.get('runtimeMs') for r in rows]), '.0f')}"
        )

    print()
    print("=== Sanity check A: position==1 (cold start in every arm — should NOT differ) ===")
    pos1: dict[str, list[dict]] = defaultdict(list)
    for arm_name, rows in arms.items():
        for row in rows:
            if row["position"] == 1:
                pos1[arm_name].append(row)
    for arm_name in ARM_NAMES:
        rows = pos1.get(arm_name, [])
        print(f"{arm_name:<12} n={len(rows):<4} mean_agg_score={fmt(mean([r['agg_score'] for r in rows]), '.3f')}")

    print()
    print("=== Paired per-task deltas (position>=2, task present in all 3 arms) ===")
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for arm_name, rows in arms.items():
        for row in rows:
            if row["position"] and row["position"] >= 2:
                by_task[row["task_id"]][arm_name] = row
    deltas: dict[str, list[float]] = defaultdict(list)
    for task_id, by_arm in by_task.items():
        if not all(a in by_arm for a in ARM_NAMES):
            continue
        c, t, r = by_arm["control"], by_arm["task-scope"], by_arm["repo-scope"]
        if None not in (r["agg_score"], c["agg_score"]):
            deltas["repo_minus_control.agg_score"].append(r["agg_score"] - c["agg_score"])
        if None not in (r["agg_score"], t["agg_score"]):
            deltas["repo_minus_task.agg_score"].append(r["agg_score"] - t["agg_score"])
        if None not in (t["agg_score"], c["agg_score"]):
            deltas["task_minus_control.agg_score"].append(t["agg_score"] - c["agg_score"])
        if None not in (r.get("inputTokens"), c.get("inputTokens")):
            deltas["repo_minus_control.inputTokens"].append(r["inputTokens"] - c["inputTokens"])
        if None not in (r.get("outputTokens"), c.get("outputTokens")):
            deltas["repo_minus_control.outputTokens"].append(r["outputTokens"] - c["outputTokens"])
    for key in sorted(deltas.keys()):
        values = deltas[key]
        print(f"{key:<32} n={len(values):<4} mean={mean(values):.4f}")


if __name__ == "__main__":
    main()
