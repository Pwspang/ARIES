#!/usr/bin/env python3
"""Summarize the Hermes episodic-memory (MemRL) study on swe-atlas-qa.

Design and hypotheses: docs/benchmarks/swe-atlas-qa.md, "The Hermes
episodic-memory study". Reads only files ARIES already writes:

  <run>/run-result.json                              profile name -> study, split, arm
  <run>/task-*-NNN/evaluation/evaluation_results.json agg_score, reward
  <run>/task-*-NNN/evaluation/reward.txt             fallback: judge never finished -> 0
  <run>/task-*-NNN/harness/telemetry/sessions.jsonl  tokens, API/tool calls, session id
  <run>/task-*-NNN/harness/session-outcome.json      wall-clock duration
  <run>/task-*-NNN/harness/memory/memrl.db           what this task recalled
  <run>/memory/memrl.db                              memory upkeep calls (llm_calls)

Runs from before the generic harness.memory contract kept the store at
harness/memrl/memrl.db and <run>/memrl/memrl.db; both layouts are read.
  .cache/swe-atlas-qa/data/qa/<task>/task.toml       repository

Usage:
  scripts/summarize_memrl_study.py runs/memrl-study/*
  scripts/summarize_memrl_study.py --study memrl-smoke --csv tasks.csv runs/memrl-study/*

The latest run is used when a profile was run more than once.
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
NAME_RE = re.compile(r"^hermes-sweatlasqa-(?P<study>memrl(?:-smoke)?)-(?P<split>train|test|replay)-"
                     r"(?P<arm>control|memrl)-sglang$")
ITERATION_CAP = 90  # Hermes max_iterations, recorded in sessions.jsonl model_config
# Tasks per reporting window of the training curve: the mini-batch size of
# the runs from before sequential training.
BATCH_SIZES = {"memrl": 15, "memrl-smoke": 2}
METRICS = ("agg_score", "passed", "prompt_tokens", "output_tokens", "api_calls", "tool_calls", "duration_s")


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


def memrl_store(directory):
    """The MemRL store in a state directory's current or pre-contract layout."""
    current = directory / "memory" / "memrl.db"
    return current if current.is_file() else directory / "memrl" / "memrl.db"


def task_rows(run_dir, name, repos):
    """One row per occurrence. Train memory arms repeat the split once per
    epoch; epoch and batch number come from the execution order."""
    match = NAME_RE.match(name)
    # Upkeep is keyed by the session that produced the memory, never by task
    # ID: a frozen run finishes training's last memory under training's ID.
    upkeep = defaultdict(lambda: [0, 0, 0])
    for session, prompt, completion, ms in query(
            memrl_store(run_dir),
            "SELECT session_id, prompt_tokens, completion_tokens, duration_ms FROM llm_calls"):
        upkeep[session][0] += prompt
        upkeep[session][1] += completion
        upkeep[session][2] += ms
    dirs = sorted((int(m[2]), m[1], d) for d in run_dir.iterdir() if (m := TASK_DIR_RE.match(d.name)))
    per_epoch = len({task for _, task, _ in dirs}) or 1
    batch_size = BATCH_SIZES[match["study"]]
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
        store = memrl_store(task_dir / "harness")
        recalled = query(store, "SELECT active_ids FROM sessions WHERE session_id = ?", (session.get("id", ""),))
        recalled_ids = json.loads(recalled[0][0]) if recalled else []
        sources = [r[0] for r in query(
            store, f"SELECT task_id FROM memories WHERE id IN ({','.join('?' * len(recalled_ids))})", recalled_ids)] \
            if recalled_ids else []
        source_tasks = [m[1] if (m := TASK_DIR_RE.match(s)) else s for s in sources]
        up_prompt, up_completion, up_ms = upkeep.get(session.get("id", ""), (0, 0, 0))
        rows.append({
            "study": match["study"], "split": match["split"], "arm": match["arm"], "task": task, "repo": repo,
            "order": order, "epoch": (order - 1) // per_epoch + 1, "batch": (order - 1) // batch_size + 1,
            "position_in_repo": seen_in_repo[repo], "agg_score": agg, "passed": passed,
            "prompt_tokens": (session.get("input_tokens") or 0) + (session.get("cache_read_tokens") or 0),
            "uncached_input_tokens": session.get("input_tokens"), "output_tokens": session.get("output_tokens"),
            "api_calls": session.get("api_call_count"), "tool_calls": session.get("tool_call_count"),
            "hit_iteration_cap": (session.get("api_call_count") or 0) >= ITERATION_CAP,
            "duration_s": (outcome.get("duration_ms") or 0) / 1000,
            "upkeep_prompt_tokens": up_prompt, "upkeep_output_tokens": up_completion, "upkeep_s": up_ms / 1000,
            "recalled": len(recalled_ids),
            "recalled_same_task": sum(s == task for s in source_tasks),
            "recalled_same_repo": sum(repos.get(s) == repo for s in source_tasks),
        })
        seen_in_repo[repo] += 1
    return rows


def latest_runs(paths, study):
    """(name, path) of the latest run per study profile (run IDs start with a sortable timestamp)."""
    by_name = {}
    for path in sorted(paths):
        name = (load_json(path / "run-result.json") or {}).get("name", "")
        match = NAME_RE.match(name)
        if match and match["study"] == study:
            by_name[name] = path
    return by_name.items()


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


def contrast_table(title, pairs, cells, rng):
    """pairs: [(label, arm_a, arm_b)] over cells {(split, arm): {task: row}}; paired on shared tasks."""
    print(f"\n## {title}\n")
    print("| contrast | metric | n tasks | mean delta | 95% CI | p |")
    print("|---|---|---|---|---|---|")
    for label, a, b in pairs:
        common = sorted(set(cells.get(a, {})) & set(cells.get(b, {})))
        if not common:
            continue
        for metric in METRICS:
            d, (lo, hi), p = paired([cells[a][t][metric] - cells[b][t][metric] for t in common], rng)
            print(f"| {label} | {metric} | {len(common)} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3f} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--study", default="memrl", choices=("memrl", "memrl-smoke"))
    parser.add_argument("--qa-root", type=Path, default=Path(".cache/swe-atlas-qa"))
    parser.add_argument("--csv", type=Path, help="also write one row per task occurrence")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    repos = repositories(args.qa_root)
    rows = [row for name, run in latest_runs(args.runs, args.study) for row in task_rows(run, name, repos)]
    rows = [r for r in rows if r["agg_score"] is not None]
    if not rows:
        sys.exit(f"no scored runs of study {args.study!r} found")
    if args.csv:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["split"], r["arm"], r["epoch"])].append(r)
    print("## Per split, arm and epoch\n")
    print("| split | arm | epoch | n | agg_score | pass | prompt tok | output tok | API calls | tool calls | at cap | "
          "wall s | upkeep tok | recalled | same-repo recall |")
    print("|---|" + "---|" * 14)
    for (split, arm, epoch), rs in sorted(groups.items()):
        print(f"| {split} | {arm} | {epoch} | {len(rs)} | {mean(r['agg_score'] for r in rs):.3f} | "
              f"{mean(r['passed'] for r in rs):.2f} | {mean(r['prompt_tokens'] for r in rs):,.0f} | "
              f"{mean(r['output_tokens'] for r in rs):,.0f} | {mean(r['api_calls'] for r in rs):.1f} | "
              f"{mean(r['tool_calls'] for r in rs):.1f} | {mean(r['hit_iteration_cap'] for r in rs):.2f} | "
              f"{mean(r['duration_s'] for r in rs):.0f} | "
              f"{mean(r['upkeep_prompt_tokens'] + r['upkeep_output_tokens'] for r in rs):,.0f} | "
              f"{mean(r['recalled'] for r in rs):.2f} | {mean(r['recalled_same_repo'] for r in rs):.2f} |")

    print("\n## One-off training cost of MemRL (all epochs of the train split)\n")
    tests = len({r["task"] for r in rows if r["split"] == "test"})
    for arm in ("memrl",):
        rs = [r for r in rows if r["split"] == "train" and r["arm"] == arm]
        if rs:
            upkeep = sum(r["upkeep_prompt_tokens"] + r["upkeep_output_tokens"] for r in rs)
            agent = sum(r["prompt_tokens"] + (r["output_tokens"] or 0) for r in rs)
            print(f"- {arm}: {upkeep:,} memory-writing tokens, {agent:,} agent tokens over {len(rs)} task runs"
                  + (f"; {upkeep / tests:,.0f} memory-writing tokens per test task amortized" if tests else ""))

    rng = random.Random(args.seed)
    # Contrasts pair on task; train-memrl contributes its first epoch,
    # the only pass that is comparable to a single control pass.
    cells = {(split, arm): {r["task"]: r for r in rs} for (split, arm, epoch), rs in groups.items() if epoch == 1}
    contrast_table("Held-out test, frozen memory (H1 accuracy, H2 efficiency)", [
        ("memrl - control", ("test", "memrl"), ("test", "control")),
    ], cells, rng)
    contrast_table("Manipulation check: frozen MemRL replayed on its own train tasks (H0)", [
        ("replay - train control", ("replay", "memrl"), ("train", "control")),
        ("replay - memrl's first training epoch", ("replay", "memrl"), ("train", "memrl")),
    ], cells, rng)

    # H3: the learning curve during training. Each train task in epoch e is
    # paired with the same task's control run, so epochs compare like with
    # like; batches show the curve inside the first epoch.
    print("\n## Training: MemRL - control by epoch, and score by batch (H3)\n")
    print("| arm | epoch | n | agg_score delta | tool calls delta | prompt tok delta | recalled |")
    print("|---|---|---|---|---|---|---|")
    control = cells.get(("train", "control"), {})
    for arm in ("memrl",):
        for (split, group_arm, epoch), rs in sorted(groups.items()):
            if split != "train" or group_arm != arm:
                continue
            pairs = [(r, control[r["task"]]) for r in rs if r["task"] in control]
            if pairs:
                print(f"| {arm} | {epoch} | {len(pairs)} | {mean(m['agg_score'] - c['agg_score'] for m, c in pairs):+.3f} | "
                      f"{mean(m['tool_calls'] - c['tool_calls'] for m, c in pairs):+.1f} | "
                      f"{mean(m['prompt_tokens'] - c['prompt_tokens'] for m, c in pairs):+,.0f} | "
                      f"{mean(m['recalled'] for m, _ in pairs):.2f} |")
    print("\n| arm | batch | epoch | n | agg_score | recalled | same-task recall |")
    print("|---|---|---|---|---|---|---|")
    by_batch = defaultdict(list)
    for r in rows:
        if r["split"] == "train" and r["arm"] != "control":
            by_batch[(r["arm"], r["batch"])].append(r)
    for (arm, batch), rs in sorted(by_batch.items()):
        print(f"| {arm} | {batch} | {rs[0]['epoch']} | {len(rs)} | {mean(r['agg_score'] for r in rs):.3f} | "
              f"{mean(r['recalled'] for r in rs):.2f} | {mean(r['recalled_same_task'] > 0 for r in rs):.2f} |")

if __name__ == "__main__":
    main()
