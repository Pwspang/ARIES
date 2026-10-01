#!/usr/bin/env python3
"""Plot accuracy and efficiency for the Hermes episodic-memory (MemRL) study.

Design and hypotheses: docs/benchmarks/swe-atlas-qa.md, "The Hermes
episodic-memory study". Rows come from summarize_memrl_study.py, so the
charts and the summary tables read the same runs the same way.

Run with:
  uv run --with matplotlib --with numpy \\
      scripts/plot_memrl_study.py runs/memrl-study/*

Writes to scripts/out_memrl/:
  accuracy.png       agg_score and pass rate per condition, mean and 95% CI
  efficiency.png     tool calls, API calls, prompt/output tokens, wall time,
                     iteration-cap share per condition, mean and 95% CI
  accuracy_test.png, efficiency_test.png
                     the same, for the held-out test split only (control vs
                     frozen MemRL), each panel with its paired delta and p
  paired_deltas.png  per-task paired MemRL - control deltas with 95% CI
  training.png       agg_score and recall by window of 15 training tasks
"""
import argparse
import random
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_memrl_study as sms  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTROL = "#2a78d6"
MEMRL = "#eb6834"

# (label, split, arm, epoch); the x order of every per-condition chart.
CONDITIONS = [
    ("train\ncontrol", "train", "control", 1),
    ("train MemRL\nepoch 1", "train", "memrl", 1),
    ("train MemRL\nepoch 2", "train", "memrl", 2),
    ("replay\nfrozen MemRL", "replay", "memrl", 1),
    ("test\ncontrol", "test", "control", 1),
    ("test\nfrozen MemRL", "test", "memrl", 1),
]
TEST_CONDITIONS = [("control\n(no memory)", "test", "control", 1), ("frozen\nMemRL", "test", "memrl", 1)]
# (label, (split, arm, epoch) of a, (split, arm, epoch) of b): a - b paired on task.
CONTRASTS = [
    ("test: frozen MemRL - control (H1, H2)", ("test", "memrl", 1), ("test", "control", 1)),
    ("replay: frozen MemRL - train control (H0)", ("replay", "memrl", 1), ("train", "control", 1)),
    ("train epoch 2: MemRL - control (H3)", ("train", "memrl", 2), ("train", "control", 1)),
    ("train epoch 1: MemRL - control (H3)", ("train", "memrl", 1), ("train", "control", 1)),
]
# (key, title, scale): the metrics drawn, in order.
ACCURACY = [("agg_score", "agg_score (higher is better)", 1), ("passed", "pass rate (higher is better)", 1)]
EFFICIENCY = [
    ("tool_calls", "tool calls per task", 1),
    ("api_calls", "API calls per task", 1),
    ("prompt_tokens", "prompt tokens per task (M)", 1e-6),
    ("output_tokens", "output tokens per task (K)", 1e-3),
    ("duration_s", "wall time per task (min)", 1 / 60),
    ("hit_iteration_cap", "share at 90-iteration cap", 1),
]
WINDOW = 15


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlesize": 10, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    })


def bootstrap(values, rng, draws=10000):
    """Mean and 95% bootstrap CI."""
    values = [float(v) for v in values if v is not None]
    if not values:
        return float("nan"), float("nan"), float("nan")
    boots = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(draws))
    return statistics.fmean(values), boots[int(0.025 * draws)], boots[int(0.975 * draws)]


def footnote(fig, text):
    fig.text(0.01, 0.005, text, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def condition_bars(ax, groups, conditions, key, scale, rng, decimals):
    """One bar per condition with a 95% CI whisker; the mean is labeled at the tip."""
    xs = range(len(conditions))
    stats = [bootstrap([r[key] for r in groups.get((s, a, e), [])], rng)
             for _, s, a, e in conditions]
    means = [m * scale for m, _, _ in stats]
    colors = [CONTROL if a == "control" else MEMRL for _, _, a, _ in conditions]
    ax.bar(xs, means, width=0.55 if len(conditions) > 2 else 0.45, color=colors, zorder=2)
    ax.errorbar(xs, means, yerr=[[(m - lo) * scale for m, lo, _ in stats], [(hi - m) * scale for m, _, hi in stats]],
                fmt="none", ecolor=INK_2, elinewidth=1, capsize=3, zorder=3)
    for x, (m, _, hi) in zip(xs, stats):
        ax.annotate(f"{m * scale:.{decimals}f}", (x, hi * scale), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(list(xs), [label for label, *_ in conditions], fontsize=7.5)
    ax.grid(axis="x", visible=False)
    if conditions is CONDITIONS:
        # Hairline separators between the train, replay and test splits.
        for x in (2.5, 3.5):
            ax.axvline(x, color=GRID, linewidth=1, zorder=1)
    ax.margins(y=0.15)


def legend(fig):
    handles = [plt.Rectangle((0, 0), 1, 1, color=CONTROL), plt.Rectangle((0, 0), 1, 1, color=MEMRL)]
    fig.legend(handles, ["control (no memory)", "MemRL"], loc="upper right", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.99, 0.995))


def plot_conditions(groups, conditions, metrics, title, ncols, path, note, rng, paired_rng=None):
    """Bars per condition; with paired_rng (two conditions only), each panel
    also names the paired per-task delta of the second minus the first."""
    nrows = -(-len(metrics) // ncols)
    width = 5.2 if len(conditions) > 2 else 3.4
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, 3.4 * nrows + 0.6), squeeze=False)
    for ax, (key, label, scale) in zip(axes.flat, metrics):
        decimals = 2 if max(abs(scale * r[key]) for rs in groups.values() for r in rs) < 10 else 0
        condition_bars(ax, groups, conditions, key, scale, rng, decimals)
        ax.set_title(label, pad=16 if paired_rng is not None else 6)
        if paired_rng is not None:
            (_, *a), (_, *b) = conditions[1], conditions[0]
            first, second = ({r["task"]: r for r in groups[tuple(k)]} for k in (b, a))
            common = sorted(set(first) & set(second))
            d, (lo, hi), p = sms.paired([second[t][key] - first[t][key] for t in common], paired_rng)
            places = max(decimals, 1)
            ax.annotate(f"paired delta {d * scale:+.{places}f}  [{lo * scale:+.{places}f}, "
                        f"{hi * scale:+.{places}f}]  p={p:.3f}", (0, 1), xycoords="axes fraction",
                        xytext=(0, 4), textcoords="offset points", va="bottom", fontsize=7.5, color=INK_2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    if len(conditions) > 2:
        legend(fig)  # With two bars, the x labels already name each arm.
    footnote(fig, note)
    fig.tight_layout(rect=(0, 0.03 * note.count("\n") + 0.03, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_deltas(cells, path, note, rng):
    metrics = ACCURACY[:1] + [m for m in EFFICIENCY if m[0] in ("tool_calls", "api_calls", "prompt_tokens",
                                                                 "duration_s")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 3.6), sharey=True)
    ys = list(range(len(CONTRASTS)))[::-1]
    for ax, (key, label, scale) in zip(axes, metrics):
        for y, (_, a, b) in zip(ys, CONTRASTS):
            common = sorted(set(cells.get(a, {})) & set(cells.get(b, {})))
            deltas = [cells[a][t][key] - cells[b][t][key] for t in common]
            d, (lo, hi), p = sms.paired(deltas, rng)
            ax.plot([lo * scale, hi * scale], [y, y], color=MEMRL, linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter([d * scale], [y], s=48, color=MEMRL, edgecolor=SURFACE, linewidth=2, zorder=3)
            shown = round(d * scale, 2) or 0.0  # no "-0.00"
            ax.annotate(f"{shown:+.2f}" if abs(shown) < 10 else f"{shown:+.0f}", (d * scale, y),
                        xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5, color=INK)
            ax.annotate(f"p={p:.3f}", (hi * scale, y), xytext=(4, -3), textcoords="offset points",
                        fontsize=7, color=MUTED)
        ax.axvline(0, color=INK_2, linewidth=1, zorder=1)
        ax.set_title(label.replace(" per task", "").replace(" (higher is better)", ""), fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.25, y=0.2)
        ax.set_xlabel("better ->" if key == "agg_score" else "<- better", color=MUTED, fontsize=7.5)
    axes[0].set_yticks(ys, [label for label, *_ in CONTRASTS], fontsize=8)
    fig.suptitle("Paired per-task deltas, MemRL minus control (mean, 95% bootstrap CI, sign-flip p)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    footnote(fig, note)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_training(rows, cells, path, note, rng):
    train = sorted((r for r in rows if r["split"] == "train" and r["arm"] == "memrl"), key=lambda r: r["order"])
    control = cells.get(("train", "control", 1), {})
    windows = [train[i:i + WINDOW] for i in range(0, len(train), WINDOW)]
    xs = list(range(1, len(windows) + 1))
    labels = [f"tasks {w[0]['order']}-{w[-1]['order']}\nepoch {w[0]['epoch']}" for w in windows]
    fig, (score_ax, recall_ax) = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for arm, color, pick in (("control (same tasks)", CONTROL, lambda r: control.get(r["task"])),
                             ("MemRL", MEMRL, lambda r: r)):
        stats = [bootstrap([p["agg_score"] for r in w if (p := pick(r))], rng) for w in windows]
        means = [m for m, _, _ in stats]
        score_ax.fill_between(xs, [lo for _, lo, _ in stats], [hi for _, _, hi in stats], color=color, alpha=0.1,
                              linewidth=0)
        score_ax.plot(xs, means, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                      markeredgewidth=2, label=arm)
        score_ax.annotate(f"{means[-1]:.2f}", (xs[-1], means[-1]), xytext=(8, 0), textcoords="offset points",
                          va="center", fontsize=8, color=INK)
    score_ax.set_title(f"agg_score per window of {WINDOW} training tasks")
    score_ax.legend(frameon=False, fontsize=8, loc="lower left")
    for ax in (score_ax, recall_ax):
        ax.set_xticks(xs, labels, fontsize=7.5)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(0.6, len(xs) + 0.5)
    recalled = [statistics.fmean(r["recalled"] for r in w) for w in windows]
    same_task = [statistics.fmean(r["recalled_same_task"] for r in w) for w in windows]
    same_repo = [statistics.fmean(r["recalled_same_repo"] for r in w) for w in windows]
    recall_ax.plot(xs, recalled, color=MEMRL, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                   markeredgewidth=2, label="all recalled")
    recall_ax.plot(xs, same_repo, color=MEMRL, linewidth=2, linestyle=(0, (4, 2)), marker="s", markersize=6,
                   markeredgecolor=SURFACE, markeredgewidth=2, label="from the same repository")
    recall_ax.plot(xs, same_task, color=MEMRL, linewidth=2, linestyle=(0, (1, 2)), marker="^", markersize=7,
                   markeredgecolor=SURFACE, markeredgewidth=2, label="from the same task")
    recall_ax.set_title("MemRL memories recalled per task")
    recall_ax.legend(frameon=False, fontsize=8, loc="upper left")
    recall_ax.set_ylim(bottom=0)
    fig.suptitle("Training: does MemRL improve as its store fills? (H3)", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    footnote(fig, note)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--study", default="memrl", choices=("memrl", "memrl-smoke"))
    parser.add_argument("--qa-root", type=Path, default=Path(".cache/swe-atlas-qa"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out_memrl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    repos = sms.repositories(args.qa_root)
    runs = list(sms.latest_runs(args.runs, args.study))
    rows = [row for name, run in runs for row in sms.task_rows(run, name, repos)]
    rows = [r for r in rows if r["agg_score"] is not None]
    if not rows:
        sys.exit(f"no scored runs of study {args.study!r} found")
    groups = {}
    for r in rows:
        groups.setdefault((r["split"], r["arm"], r["epoch"]), []).append(r)
    cells = {k: {r["task"]: r for r in rs} for k, rs in groups.items()}
    tasks = len({r["task"] for r in rows if r["split"] == "test"})
    note = (f"Hermes + Qwen3.6-35B-A3B on swe-atlas-qa, {args.study} study, {tasks} test tasks; CIs over tasks. "
            "MemRL training ran at concurrency 1, every other run at 8 on the same server, "
            "so its wall time is not comparable.")
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    # Each chart draws from its own seeded stream so one can change without moving the others' CIs.
    plot_conditions(groups, CONDITIONS, ACCURACY, "Accuracy by condition (mean, 95% bootstrap CI)", 2,
                    args.out / "accuracy.png", note, random.Random(args.seed))
    plot_conditions(groups, CONDITIONS, EFFICIENCY,
                    "Efficiency by condition (mean, 95% bootstrap CI; lower is better)", 3,
                    args.out / "efficiency.png", note, random.Random(args.seed))
    test_note = (f"Hermes + Qwen3.6-35B-A3B on swe-atlas-qa, {args.study} study, the same {tasks} held-out test tasks\n"
                 "in both arms, both at concurrency 8. Bars: mean, 95% bootstrap CI over tasks.\n"
                 "Paired delta: MemRL - control per task, 95% CI, sign-flip p.")
    plot_conditions(groups, TEST_CONDITIONS, ACCURACY, "Held-out test accuracy: control vs frozen MemRL", 2,
                    args.out / "accuracy_test.png", test_note, random.Random(args.seed), random.Random(args.seed))
    plot_conditions(groups, TEST_CONDITIONS, EFFICIENCY,
                    "Held-out test efficiency: control vs frozen MemRL (lower is better)", 3,
                    args.out / "efficiency_test.png", test_note, random.Random(args.seed), random.Random(args.seed))
    plot_deltas(cells, args.out / "paired_deltas.png", note, random.Random(args.seed))
    plot_training(rows, cells, args.out / "training.png", note, random.Random(args.seed))
    print(f"wrote {', '.join(p.name for p in sorted(args.out.glob('*.png')))} to {args.out}")


if __name__ == "__main__":
    main()
