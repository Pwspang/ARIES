#!/usr/bin/env python3
"""Summarize the Hermes episodic-memory (MemRL) study on swe-atlas-qa.

Design and hypotheses: docs/benchmarks/swe-atlas-qa.md, "The Hermes
episodic-memory study". Reads only files ARIES already writes:

  <run>/run-result.json                              profile name -> replicate, arm
  <run>/task-*-NNN/evaluation/evaluation_results.json agg_score, reward
  <run>/task-*-NNN/evaluation/reward.txt             fallback: judge never finished -> 0
  <run>/task-*-NNN/harness/telemetry/sessions.jsonl  tokens, API/tool calls
  <run>/task-*-NNN/harness/session-outcome.json      wall-clock duration
  <run>/task-*-NNN/harness/memrl/memrl.db            what this task recalled
  <run>/memrl/memrl.db                               memory upkeep calls (llm_calls)
  .cache/swe-atlas-qa/data/qa/<task>/task.toml       repository

Usage:
  scripts/summarize_memrl_study.py runs/memrl-study/*hermes-sweatlasqa-pilot30*
  scripts/summarize_memrl_study.py --csv tasks.csv runs/memrl-study/*

A replicate's latest run is used when a profile was run more than once.
"""
import argparse
import csv
import json
import random
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TASK_DIR_RE = re.compile(r"^(task-[0-9a-f]+)-(\d+)$")
NAME_RE = re.compile(r"^hermes-sweatlasqa-(?P<replicate>.+?)(?P<arm>-epoch2-memrl|-memrl-sim|-memrl)?-deepseek$")
ARMS = {None: "control", "-memrl-sim": "similarity", "-memrl": "memrl", "-epoch2-memrl": "memrl-epoch2"}
ITERATION_CAP = 90  # Hermes max_iterations, recorded in sessions.jsonl model_config
POSITION_BUCKETS = ((0, 0, "1st in repo"), (1, 4, "2nd-5th"), (5, 9, "6th-10th"), (10, 10**6, "11th+"))


def repositories(qa_root):
    repos = {}
    for toml in (qa_root / "data" / "qa").glob("*/task.toml"):
        for line in toml.read_text().splitlines():
            if line.strip().startswith("repository "):
                repos[toml.parent.name] = line.split("=", 1)[1].strip().strip('"')
    return repos


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def query(db, sql, args=()):
    if not db.is_file():
        return []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return conn.execute(sql, args).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def score(task_dir):
    """(agg_score, passed); a missing verdict counts as 0, as in the pilot30 study."""
    result = load_json(task_dir / "evaluation" / "evaluation_results.json")
    if result is not None:
        return float(result.get("agg_score") or 0), float(result.get("reward") or 0) >= 1
    if (task_dir / "evaluation" / "reward.txt").is_file():
        return 0.0, False
    return None, None


def task_rows(run_dir, repos):
    result = load_json(run_dir / "run-result.json") or {}
    match = NAME_RE.match(result.get("name", ""))
    if not match:
        return []
    replicate, arm = match["replicate"], ARMS[match["arm"]]
    upkeep = defaultdict(lambda: [0, 0, 0])
    for task_id, prompt, completion, ms in query(
            run_dir / "memrl" / "memrl.db",
            "SELECT task_id, prompt_tokens, completion_tokens, duration_ms FROM llm_calls"):
        upkeep[task_id][0] += prompt
        upkeep[task_id][1] += completion
        upkeep[task_id][2] += ms
    dirs = sorted((int(m[2]), m[1], d) for d in run_dir.iterdir() if (m := TASK_DIR_RE.match(d.name)))
    task_count = len({task for _, task, _ in dirs})
    seen_in_repo = defaultdict(int)
    rows = []
    for order, task, task_dir in dirs:
        repo = repos.get(task, "?")
        agg, passed = score(task_dir)
        session = {}
        telemetry = task_dir / "harness" / "telemetry" / "sessions.jsonl"
        if telemetry.is_file():
            session = json.loads(telemetry.read_text().splitlines()[0])
        outcome = load_json(task_dir / "harness" / "session-outcome.json") or {}
        recalled = query(task_dir / "harness" / "memrl" / "memrl.db",
                         "SELECT active_ids FROM sessions WHERE session_id = ?", (session.get("id", ""),))
        recalled_ids = json.loads(recalled[0][0]) if recalled else []
        sources = [r[0] for r in query(
            task_dir / "harness" / "memrl" / "memrl.db",
            f"SELECT task_id FROM memories WHERE id IN ({','.join('?' * len(recalled_ids))})", recalled_ids)] \
            if recalled_ids else []
        source_tasks = [TASK_DIR_RE.match(s)[1] if TASK_DIR_RE.match(s) else s for s in sources]
        occurrence = f"{task}-{order:03d}"
        prompt_tokens = (session.get("input_tokens") or 0) + (session.get("cache_read_tokens") or 0)
        up_prompt, up_completion, up_ms = upkeep.get(occurrence, (0, 0, 0))
        rows.append({
            "replicate": replicate, "arm": arm, "task": task, "repo": repo, "order": order,
            "epoch": 1 + (order - 1) * 2 // task_count if arm == "memrl-epoch2" else 1,
            "position_in_repo": seen_in_repo[repo],
            "agg_score": agg, "passed": passed,
            "prompt_tokens": prompt_tokens, "uncached_input_tokens": session.get("input_tokens"),
            "output_tokens": session.get("output_tokens"), "api_calls": session.get("api_call_count"),
            "tool_calls": session.get("tool_call_count"),
            "hit_iteration_cap": (session.get("api_call_count") or 0) >= ITERATION_CAP,
            "duration_s": (outcome.get("duration_ms") or 0) / 1000,
            "upkeep_prompt_tokens": up_prompt, "upkeep_output_tokens": up_completion, "upkeep_s": up_ms / 1000,
            "recalled": len(recalled_ids),
            "recalled_same_task": sum(s == task for s in source_tasks),
            "recalled_same_repo": sum(repos.get(s) == repo for s in source_tasks),
        })
        seen_in_repo[repo] += 1
    return rows


def latest_runs(paths):
    """The latest run per profile name (run IDs start with a sortable timestamp)."""
    by_name = {}
    for path in sorted(paths):
        name = (load_json(path / "run-result.json") or {}).get("name")
        if name:
            by_name[name] = path
    return by_name.values()


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else float("nan")


def paired(deltas, rng, draws=10000):
    """Mean delta, 95% bootstrap CI, and a two-sided sign-flip permutation p."""
    if not deltas:
        return float("nan"), (float("nan"), float("nan")), float("nan")
    observed = statistics.fmean(deltas)
    boots = sorted(statistics.fmean(rng.choices(deltas, k=len(deltas))) for _ in range(draws))
    flips = sum(abs(statistics.fmean(d if rng.random() < 0.5 else -d for d in deltas)) >= abs(observed) - 1e-12
                for _ in range(draws))
    return observed, (boots[int(0.025 * draws)], boots[int(0.975 * draws)]), (flips + 1) / (draws + 1)


METRICS = ("agg_score", "passed", "prompt_tokens", "output_tokens", "api_calls", "tool_calls", "duration_s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--qa-root", type=Path, default=Path(".cache/swe-atlas-qa"))
    parser.add_argument("--csv", type=Path, help="also write one row per task occurrence")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    repos = repositories(args.qa_root)
    rows = [row for run in latest_runs(args.runs) for row in task_rows(run, repos)]
    rows = [r for r in rows if r["agg_score"] is not None]
    if not rows:
        sys.exit("no scored hermes-sweatlasqa tasks found")
    if args.csv:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["arm"] if r["epoch"] == 1 else "memrl-epoch2 (2nd pass)")].append(r)
    print("## Per arm (all replicates pooled)\n")
    print("| arm | n | agg_score | pass | prompt tok | output tok | API calls | tool calls | at cap | "
          "wall s | upkeep tok | recalled | same-repo recall |")
    print("|---|" + "---|" * 12)
    for arm, rs in sorted(groups.items()):
        print(f"| {arm} | {len(rs)} | {mean(r['agg_score'] for r in rs):.3f} | {mean(r['passed'] for r in rs):.2f} | "
              f"{mean(r['prompt_tokens'] for r in rs):,.0f} | {mean(r['output_tokens'] for r in rs):,.0f} | "
              f"{mean(r['api_calls'] for r in rs):.1f} | {mean(r['tool_calls'] for r in rs):.1f} | "
              f"{mean(r['hit_iteration_cap'] for r in rs):.2f} | {mean(r['duration_s'] for r in rs):.0f} | "
              f"{mean(r['upkeep_prompt_tokens'] + r['upkeep_output_tokens'] for r in rs):,.0f} | "
              f"{mean(r['recalled'] for r in rs):.2f} | {mean(r['recalled_same_repo'] for r in rs):.2f} |")

    # Paired contrasts: per task, average each arm over replicates, then
    # difference, so task difficulty cancels and replicates are not treated as
    # independent tasks.
    rng = random.Random(args.seed)
    per_task = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["epoch"] == 1 and r["arm"] != "memrl-epoch2":
            per_task[r["arm"]][r["task"]].append(r)
    contrasts = [("memrl", "control"), ("similarity", "control"), ("memrl", "similarity")]
    print("\n## Paired per-task contrasts (task means over replicates)\n")
    print("| contrast | metric | n tasks | mean delta | 95% CI | p |")
    print("|---|---|---|---|---|---|")
    for a, b in contrasts:
        common = sorted(set(per_task[a]) & set(per_task[b]))
        for metric in METRICS:
            deltas = [mean(r[metric] for r in per_task[a][t]) - mean(r[metric] for r in per_task[b][t])
                      for t in common]
            d, (lo, hi), p = paired(deltas, rng)
            if common:
                print(f"| {a} - {b} | {metric} | {len(common)} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3f} |")

    # H3: does the benefit grow with same-repo history? Deltas are paired on
    # (replicate, task) here, because position differs between replicates.
    print("\n## memrl - control by position within repository (paired on replicate and task)\n")
    print("| position | n | agg_score delta | tool calls delta | prompt tok delta |")
    print("|---|---|---|---|---|")
    control = {(r["replicate"], r["task"]): r for r in rows if r["arm"] == "control"}
    for lo, hi, label in POSITION_BUCKETS:
        pairs = [(r, control[(r["replicate"], r["task"])]) for r in rows
                 if r["arm"] == "memrl" and lo <= r["position_in_repo"] <= hi and (r["replicate"], r["task"]) in control]
        if pairs:
            print(f"| {label} | {len(pairs)} | {mean(m['agg_score'] - c['agg_score'] for m, c in pairs):+.3f} | "
                  f"{mean(m['tool_calls'] - c['tool_calls'] for m, c in pairs):+.1f} | "
                  f"{mean(m['prompt_tokens'] - c['prompt_tokens'] for m, c in pairs):+,.0f} |")

    epochs = defaultdict(dict)
    for r in rows:
        if r["arm"] == "memrl-epoch2":
            epochs[r["task"]][r["epoch"]] = r
    both = [e for e in epochs.values() if 1 in e and 2 in e]
    if both:
        print("\n## Repeat exposure (manipulation check): 2nd pass - 1st pass, same run\n")
        for metric in ("agg_score", "passed", "tool_calls", "prompt_tokens"):
            d, (lo, hi), p = paired([e[2][metric] - e[1][metric] for e in both], rng)
            print(f"- {metric}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}], p={p:.3f}, n={len(both)}; "
                  f"2nd pass recalled its own task in {mean(e[2]['recalled_same_task'] > 0 for e in both):.0%}")


if __name__ == "__main__":
    main()
