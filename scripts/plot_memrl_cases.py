#!/usr/bin/env python3
"""Case-study charts for the Hermes episodic-memory (MemRL) study: where
memory helped, where it did not, and what it recalled.

Companion to plot_memrl_study.py (aggregates); rows come from
summarize_memrl_study.py. Run with:
  uv run --with matplotlib --with numpy \\
      scripts/plot_memrl_cases.py runs/memrl-study/*

Writes to scripts/out_memrl/:
  cases_test_tasks.png    every held-out test task: control vs frozen MemRL
                          score, with what MemRL recalled and why a run ended
  cases_memory_kinds.png  the frozen store and what tasks recalled from it:
                          success episodes vs failure reflections
  cases_effort_score.png  per-task effort change vs score change, test and
                          replay, without timed-out runs
"""
import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_memrl_study as pms  # noqa: E402
import summarize_memrl_study as sms  # noqa: E402
from plot_memrl_study import AXIS, CONTROL, GRID, INK, INK_2, MEMRL, MUTED, SURFACE  # noqa: E402

EXPERIENCE = "#1baf7a"  # categorical slot 3: a stored success episode
REFLECTION = "#eda100"  # categorical slot 4: a stored "[PATTERN TO AVOID]" reflection


def task_dir(run, row):
    return run / f"{row['task']}-{row['order']:03d}"


def annotate(rows, run):
    """Adds why the harness ended and the kinds of memory the task recalled."""
    for r in rows:
        directory = task_dir(run, r)
        outcome = sms.load_json(directory / "harness" / "session-outcome.json") or {}
        r["timed_out"] = outcome.get("end_reason") == "deadline_exceeded"
        r["experiences"] = r["reflections"] = 0
        telemetry = directory / "harness" / "telemetry" / "sessions.jsonl"
        if not telemetry.is_file():
            continue
        session = json.loads(telemetry.read_text().splitlines()[0]).get("id", "")
        store = sms.memrl_store(directory / "harness")
        active = sms.query(store, "SELECT active_ids FROM sessions WHERE session_id = ?", (session,))
        ids = json.loads(active[0][0]) if active else []
        if ids:
            for (kind,) in sms.query(store, f"SELECT kind FROM memories WHERE id IN ({','.join('?' * len(ids))})", ids):
                r["experiences" if kind == "experience" else "reflections"] += 1


def case_tag(m, c):
    """The one-phrase reason a test task's outcome differs, or ''."""
    if c["timed_out"] or m["timed_out"]:
        return "timeout: " + " + ".join(arm for arm, r in (("control", c), ("MemRL", m)) if r["timed_out"])
    if c["hit_iteration_cap"] and not m["hit_iteration_cap"] and m["agg_score"] > c["agg_score"]:
        return "MemRL avoided the 90-call cap"
    if m["hit_iteration_cap"] and m["agg_score"] < c["agg_score"]:
        return "MemRL hit the 90-call cap"
    return ""


def plot_test_tasks(cells, path, note):
    control, memrl = cells[("test", "control", 1)], cells[("test", "memrl", 1)]
    tasks = sorted(set(control) & set(memrl), key=lambda t: (memrl[t]["agg_score"] - control[t]["agg_score"],
                                                            control[t]["agg_score"]))
    fig, ax = plt.subplots(figsize=(12, 0.3 * len(tasks) + 1.6))
    for y, t in enumerate(tasks):
        c, m = control[t], memrl[t]
        ax.plot([c["agg_score"], m["agg_score"]], [y, y], color=AXIS, linewidth=2, solid_capstyle="round", zorder=1)
        for r, color in ((c, CONTROL), (m, MEMRL)):
            # A timed-out run is hollow: its score is the harness deadline, not the agent's answer.
            ax.scatter([r["agg_score"]], [y], s=56, zorder=3, linewidth=2,
                       color=SURFACE if r["timed_out"] else color, edgecolor=color)
        delta = m["agg_score"] - c["agg_score"]
        ax.annotate(f"{delta:+.2f}" if abs(delta) >= 0.005 else "0", (1.03, y), xycoords=("axes fraction", "data"),
                    va="center", fontsize=7.5, color=INK)
        recall = f"{m['experiences']} ep / {m['reflections']} refl" if m["recalled"] else "-"
        ax.annotate(recall, (1.12, y), xycoords=("axes fraction", "data"), va="center", fontsize=7.5, color=INK_2)
        ax.annotate(case_tag(m, c), (1.36, y), xycoords=("axes fraction", "data"), va="center", fontsize=7.5,
                    color=INK_2)
    for x, label in ((1.03, "delta"), (1.12, "MemRL recalled"), (1.36, "case")):
        ax.annotate(label, (x, len(tasks) - 0.2), xycoords=("axes fraction", "data"), fontsize=7.5, color=MUTED,
                    fontweight="bold")
    ax.set_yticks(range(len(tasks)), [f"{control[t]['repo'].split('/')[-1]}  {t[-5:]}" for t in tasks], fontsize=7.5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.7, len(tasks) - 0.3)
    ax.set_xlabel("agg_score", fontsize=8)
    ax.grid(axis="y", visible=False)
    handles = [ax.scatter([], [], s=56, color=CONTROL), ax.scatter([], [], s=56, color=MEMRL),
               ax.scatter([], [], s=56, color=SURFACE, edgecolor=INK_2, linewidth=2)]
    fig.legend(handles, ["control (no memory)", "frozen MemRL", "run hit the 3 h harness deadline"], ncol=3,
               frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.01, 0.955))
    fig.suptitle("Held-out test tasks, one row each: where frozen MemRL helped and where it hurt", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    pms.footnote(fig, note + " ep = success episode, refl = reflection. Equal scores: MemRL's dot covers control's.")
    fig.subplots_adjust(left=0.12, right=0.56, top=0.9, bottom=0.07)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_memory_kinds(store, cells, path, note, rng):
    kinds = dict(sms.query(store, "SELECT kind, COUNT(*) FROM memories GROUP BY kind"))
    bars = [("frozen store\n(all memories)", kinds.get("experience", 0), kinds.get("reflection", 0))]
    for label, key in (("recalled on\ntest tasks", ("test", "memrl", 1)), ("recalled on\nreplay tasks", ("replay", "memrl", 1))):
        rs = cells[key].values()
        bars.append((label, sum(r["experiences"] for r in rs), sum(r["reflections"] for r in rs)))
    fig, (mix_ax, delta_ax) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    for y, (label, ep, refl) in enumerate(reversed(bars)):
        total = ep + refl
        # A 2px surface gap between the two segments.
        mix_ax.barh(y, ep / total, height=0.5, color=EXPERIENCE, edgecolor=SURFACE, linewidth=2)
        mix_ax.barh(y, refl / total, left=ep / total, height=0.5, color=REFLECTION, edgecolor=SURFACE, linewidth=2)
        mix_ax.annotate(f"{ep}", (ep / total / 2, y), ha="center", va="center", fontsize=8, color=INK)
        mix_ax.annotate(f"{refl} ({refl / total:.0%})", (ep / total + refl / total / 2, y), ha="center",
                        va="center", fontsize=8, color=INK)
    mix_ax.set_yticks(range(len(bars)), [b[0] for b in reversed(bars)], fontsize=8)
    mix_ax.set_xlim(0, 1)
    mix_ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    mix_ax.grid(axis="y", visible=False)
    mix_ax.set_title("Memory kinds: success episodes vs failure reflections")
    mix_ax.legend([plt.Rectangle((0, 0), 1, 1, color=EXPERIENCE), plt.Rectangle((0, 0), 1, 1, color=REFLECTION)],
                  ["success episode (script + trajectory)", "[PATTERN TO AVOID] reflection"], frameon=False,
                  fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    # Score change on test, split by whether any success episode was recalled. Timeouts are left out:
    # a timed-out run records no recall.
    control, memrl = cells[("test", "control", 1)], cells[("test", "memrl", 1)]
    groups = [("recalled >= 1\nsuccess episode", lambda m: m["experiences"] > 0),
              ("recalled only\nreflections", lambda m: m["experiences"] == 0 and m["reflections"] > 0)]
    for y, (label, keep) in enumerate(reversed(groups)):
        deltas = [memrl[t]["agg_score"] - control[t]["agg_score"] for t in memrl
                  if keep(memrl[t]) and not memrl[t]["timed_out"] and not control[t]["timed_out"]]
        d, lo, hi = pms.bootstrap(deltas, rng)
        delta_ax.plot([lo, hi], [y, y], color=MEMRL, linewidth=2, solid_capstyle="round")
        delta_ax.scatter([d], [y], s=56, color=MEMRL, edgecolor=SURFACE, linewidth=2, zorder=3)
        delta_ax.annotate(f"{d:+.2f}  (n={len(deltas)})", (d, y), xytext=(0, 8), textcoords="offset points",
                          ha="center", fontsize=7.5, color=INK)
    delta_ax.axvline(0, color=INK_2, linewidth=1)
    delta_ax.set_yticks(range(len(groups)), [g[0] for g in reversed(groups)], fontsize=8)
    delta_ax.set_ylim(-0.6, len(groups) - 0.4)
    delta_ax.grid(axis="y", visible=False)
    delta_ax.margins(x=0.2)
    delta_ax.set_title("Test agg_score, MemRL - control (95% CI)")
    fig.suptitle("What MemRL recalls: mostly reflections on its own failures", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    pms.footnote(fig, note + " Right panel leaves out pairs where either run timed out.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_effort_score(cells, path, note):
    panels = [("held-out test: frozen MemRL - control", ("test", "memrl", 1), ("test", "control", 1)),
              ("replay of train tasks: frozen MemRL - train control", ("replay", "memrl", 1), ("train", "control", 1))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (title, a, b) in zip(axes, panels):
        pairs = [(cells[a][t], cells[b][t]) for t in sorted(set(cells[a]) & set(cells[b]))]
        kept = [(m, c) for m, c in pairs if not m["timed_out"] and not c["timed_out"]]
        xs = [m["api_calls"] - c["api_calls"] for m, c in kept]
        ys = [m["agg_score"] - c["agg_score"] for m, c in kept]
        ax.axhline(0, color=INK_2, linewidth=1, zorder=1)
        ax.axvline(0, color=INK_2, linewidth=1, zorder=1)
        ax.scatter(xs, ys, s=56, color=MEMRL, edgecolor=SURFACE, linewidth=2, zorder=3)
        fewer = sum(x < 0 for x in xs)
        better, worse = sum(y > 0.005 for y in ys), sum(y < -0.005 for y in ys)
        ax.set_title(title, fontsize=9.5)
        ax.annotate(f"{fewer} of {len(kept)} used fewer API calls\nscore up on {better}, down on {worse}, "
                    f"unchanged on {len(kept) - better - worse}\n{len(pairs) - len(kept)} timed-out pairs left out",
                    (0.02, 0.97), xycoords="axes fraction", va="top", fontsize=7.5, color=INK_2)
        for x, y, (m, _) in zip(xs, ys, kept):
            if abs(y) >= 0.25:
                ax.annotate(f"{m['repo'].split('/')[-1]} {m['task'][-5:]}", (x, y), xytext=(6, -3),
                            textcoords="offset points", fontsize=7, color=MUTED)
        ax.set_xlabel("API calls per task, MemRL - control  (<- less effort)", fontsize=8)
        ax.margins(x=0.1, y=0.12)
    axes[0].set_ylabel("agg_score, MemRL - control", fontsize=8)
    fig.suptitle("Per task: memory cuts effort far more often than it changes the answer", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    pms.footnote(fig, note)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--study", default="memrl", choices=("memrl", "memrl-smoke"))
    parser.add_argument("--qa-root", type=Path, default=Path(".cache/swe-atlas-qa"))
    parser.add_argument("--store", type=Path, help="frozen MemRL state (default: runs/memrl-study/stores/<study>-memrl)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out_memrl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    repos = sms.repositories(args.qa_root)
    rows = []
    for name, run in sms.latest_runs(args.runs, args.study):
        run_rows = [r for r in sms.task_rows(run, name, repos) if r["agg_score"] is not None]
        annotate(run_rows, run)
        rows += run_rows
    if not rows:
        sys.exit(f"no scored runs of study {args.study!r} found")
    cells = {}
    for r in rows:
        cells.setdefault((r["split"], r["arm"], r["epoch"]), {})[r["task"]] = r
    store = (args.store or Path("runs/memrl-study/stores") / f"{args.study}-memrl") / "memrl.db"
    note = (f"Hermes + Qwen3.6-35B-A3B on swe-atlas-qa, {args.study} study, "
            "2026-09-27/28 runs (MemRL reward = benchmark verdict).")
    args.out.mkdir(parents=True, exist_ok=True)
    pms.style()
    plot_test_tasks(cells, args.out / "cases_test_tasks.png", note)
    plot_memory_kinds(store, cells, args.out / "cases_memory_kinds.png", note, random.Random(args.seed))
    plot_effort_score(cells, args.out / "cases_effort_score.png", note)
    print(f"wrote cases_*.png to {args.out}")


if __name__ == "__main__":
    main()
