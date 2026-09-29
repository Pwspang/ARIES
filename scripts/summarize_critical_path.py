#!/usr/bin/env python3
"""Summarize scripts/profile_critical_path.py output into per-arm time budgets
and upper bounds on what hiding, shrinking, skipping or cheapening memory work
(and prefix caching / queueing) could save.

Prefill/decode are rescaled per server era: a trimmed least-squares fit of
wall = overhead + alpha*prefill_cal + beta*decode_cal on turns with a single
call in flight, so the calibration taken on today's server can be applied to
runs served by an older configuration. Prefill and decode are strongly
correlated across turns, so the alpha/beta split is only well identified when
the fit lands near 1/1 (the calibration's own server).

Usage:
  scripts/summarize_critical_path.py scripts/out_profile/openclaw
"""
import sys

import numpy as np
import pandas as pd

ARMS = [("amem-repo-both", "A-Mem repo (goal+rerank)"), ("amem-repo-control", "A-Mem repo (plain)"),
        ("amem-repo", "A-Mem repo"), ("amem-task", "A-Mem task"), ("", "no memory")]


def arm_of(run):
    tail = run.split("Z-", 1)[1]
    return next(label for key, label in ARMS if key in tail)


def main():
    out = sys.argv[1]
    t = pd.read_csv(f"{out}/turns.csv")
    tasks = pd.read_csv(f"{out}/tasks.csv")
    t["era"] = np.where(t.run.str[:8] >= "20260921", "Sep21-22", "Sep06-07")
    t["arm"] = t.run.map(arm_of)
    tasks["era"] = np.where(tasks.run.str[:8] >= "20260921", "Sep21-22", "Sep06-07")
    tasks["arm"] = tasks.run.map(arm_of)

    fits = {}
    for era, g in t[(t.inflight == 1) & (t.output_tokens > 0) & t.usage_valid].groupby("era"):
        X = np.vstack([np.ones(len(g)), g.prefill_s, g.decode_s]).T
        y, keep = g.llm_wall_s.values, np.ones(len(g), bool)
        for _ in range(8):  # drop turns that still queued behind unprofiled traffic
            fits[era] = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
            resid = np.abs(y - X @ fits[era])
            keep = resid < 2.5 * np.median(resid[keep])
        print(f"{era}: wall = {fits[era][0]:.2f}s + {fits[era][1]:.2f}*prefill_cal + {fits[era][2]:.2f}*decode_cal  (n={len(g)})")
    k = t.era.map(lambda e: fits[e])
    t["overhead"] = k.map(lambda c: c[0])
    for col, i in (("prefill_s", 1), ("prefill_new_s", 1), ("memory_prefill_s", 1),
                   ("decode_s", 2), ("memory_query_decode_s", 2)):
        t[col] = t[col] * k.map(lambda c: c[i])
    # ideal prefix cache: only the new suffix of each prompt is prefilled
    t["prefix_saving_s"] = t.prefill_s - t.prefill_new_s
    t["queue_s"] = np.where(t.usage_valid, t.llm_wall_s - t.overhead - t.prefill_s - t.decode_s, 0.0)
    t["overhead"] = np.where(t.usage_valid, t.overhead, t.llm_wall_s)
    t["mem_turn_llm_s"] = np.where(t.memory_only_turn, t.llm_wall_s, 0.0)
    t["other_tool_s"] = (t.tool_phase_s - t.memory_tool_phase_s).clip(lower=0)
    t["unused_search"] = t.retrieval_used == False  # noqa: E712
    t["used_search"] = t.retrieval_used == True  # noqa: E712
    t["searched"] = t.memory_search_s > 0

    per_task = t.groupby(["era", "arm", "run", "task"]).agg(
        turns=("turn", "size"), llm=("llm_wall_s", "sum"), overhead=("overhead", "sum"),
        prefill=("prefill_s", "sum"), prefix_saving=("prefix_saving_s", "sum"), mem_prefill=("memory_prefill_s", "sum"),
        decode=("decode_s", "sum"), mem_decode=("memory_query_decode_s", "sum"), queue=("queue_s", "sum"),
        tools=("tool_phase_s", "sum"), mem_tools=("memory_tool_phase_s", "sum"), other_tools=("other_tool_s", "sum"),
        search_s=("memory_search_s", "sum"), add_s=("memory_add_s", "sum"), mem_other=("memory_other_s", "sum"),
        mem_turns=("memory_only_turn", "sum"), mem_turn_llm=("mem_turn_llm_s", "sum"),
        searches=("searched", "sum"), used=("used_search", "sum"), unused=("unused_search", "sum"),
        inj_tokens=("memory_tokens_added", "sum"), ctx_mem_tokens=("memory_ctx_tokens", "sum"),
        input_tokens=("input_tokens", "sum"), cache_read=("cache_read_tokens", "sum"),
        prefix_reuse=("prefix_reuse_tokens", "sum"), system_tokens=("system_tokens", "max"),
    ).reset_index().merge(tasks[["run", "task", "session_s"]], on=["run", "task"])

    # an unused search's cost: its tool time, the memory-only turn that issued it, and its share of memory prefill
    t["unused_cost"] = np.where(t.unused_search, t.memory_tool_phase_s + t.mem_turn_llm_s, 0.0)
    per_task = per_task.merge(t.groupby(["run", "task"]).unused_cost.sum().reset_index(), on=["run", "task"])
    per_task["unused_prefill"] = per_task.mem_prefill * np.where(per_task.searches > 0,
                                                                 per_task.unused / per_task.searches.clip(lower=1), 0)

    arm = per_task.groupby(["era", "arm"]).mean(numeric_only=True)
    n = per_task.groupby(["era", "arm"]).size().rename("tasks")
    S = arm.session_s
    pct = lambda col: (100 * arm[col] / S).round(1)
    budget = pd.DataFrame({
        "tasks": n, "session_s": S.round(0), "turns": arm.turns.round(1),
        "prefill%": pct("prefill"), "  of which memory tokens%": pct("mem_prefill"),
        "decode%": pct("decode"), "  of which memory-call args%": pct("mem_decode"),
        "queue/contention%": pct("queue"), "http/stream overhead%": pct("overhead"),
        "exec/other tools%": pct("other_tools"), "memory tools%": pct("mem_tools"),
        "  search%": pct("search_s"), "  add%": pct("add_s"),
        "harness gaps%": (100 * (S - arm.llm - arm.tools) / S).round(1),
    })
    print("\n== Time budget per task (mean seconds; % of session wall) ==")
    print(budget.T.to_string())

    tok = pd.DataFrame({
        "memory tokens injected/task": arm.inj_tokens.round(0),
        "memory tokens re-prefilled/task": arm.ctx_mem_tokens.round(0),
        "memory share of all prefilled tokens %": (100 * arm.ctx_mem_tokens / arm.input_tokens).round(2),
        "system prompt tokens": arm.system_tokens.round(0),
        "reported cache hit rate %": (100 * arm.cache_read / arm.input_tokens).round(2),
        "ideal prefix-reuse rate %": (100 * arm.prefix_reuse / arm.input_tokens).round(1),
        "memory_search calls/task": arm.searches.round(2),
        "  judged used/task": arm.used.round(2), "  judged unused/task": arm.unused.round(2),
        "memory-only turns/task": arm.mem_turns.round(2),
    })
    print("\n== Tokens, cache, retrieval use (per task means) ==")
    print(tok.T.to_string())

    bound = pd.DataFrame({
        "hide: overlap memory tool calls with model work": pct("mem_tools"),
        "shrink: drop all injected memory tokens (prefill)": pct("mem_prefill"),
        "skip: unused retrievals (tool+turn+prefill)": (100 * (arm.unused_cost + arm.unused_prefill) / S).round(1),
        "skip: every memory-only turn": pct("mem_turn_llm"),
        "cheapen: memory_add/search compute to zero": (100 * (arm.search_s + arm.add_s + arm.mem_other) / S).round(1),
        "prefix cache: prefill only new suffix": pct("prefix_saving"),
        "no queueing/contention": pct("queue"),
    })
    print("\n== Upper bounds on saving (% of session wall) ==")
    print(bound.T.to_string())
    per_task.to_csv(f"{out}/per_task.csv", index=False)


if __name__ == "__main__":
    main()
