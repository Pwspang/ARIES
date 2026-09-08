#!/usr/bin/env python3
"""Plot accuracy and efficiency metrics for the pilot30 amem study, pooling
all three replicates (pilot30, shuffle1, shuffle2) into one per-task dataset
and plotting mean +/- std per arm with seaborn.

Run with:
  uv run --with seaborn --with pandas --with matplotlib \\
      scripts/plot_sweatlas_arms.py

Metrics:
  - Accuracy: agg_score, with tasks that never produced
    evaluation_results.json (but did write a reward.txt) counted as 0 rather
    than dropped -- see docs/benchmarks/swe-atlas-qa.md, "pilot30 amem study"
    section, failure-accounting note.
  - Efficiency: input tokens, output tokens, runtime (s), and assistant-turn
    count, all from harness-turn-01/telemetry.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import summarize_sweatlas_arms as ssa  # noqa: E402

TASK_DIR_RE = re.compile(r"^(task-[0-9a-f]+)-(\d+)$")
ARM_LABELS = {"control": "control", "task-scope": "task-scope", "repo-scope": "repo-scope"}
ARM_ORDER = ["control", "task-scope", "repo-scope"]
ARM_PALETTE = {"control": "#2a78d6", "task-scope": "#eb6834", "repo-scope": "#1baf7a"}

REPLICATES = [
    dict(
        name="pilot30",
        control="runs/20260906T141944.963675899Z-openclaw-sweatlasqa-pilot30-sglang",
        task="runs/20260906T141944.965072498Z-openclaw-sweatlasqa-pilot30-amem-task-sglang",
        repo="runs/20260906T141944.964671697Z-openclaw-sweatlasqa-pilot30-amem-repo-sglang",
    ),
    dict(
        name="shuffle1",
        control="runs/20260907T013422.077077636Z-openclaw-sweatlasqa-pilot30-shuffle1-sglang",
        task="runs/20260907T013422.077076453Z-openclaw-sweatlasqa-pilot30-shuffle1-amem-task-sglang",
        repo="runs/20260907T013422.077067776Z-openclaw-sweatlasqa-pilot30-shuffle1-amem-repo-sglang",
    ),
    dict(
        name="shuffle2",
        control="runs/20260907T111756.126844838Z-openclaw-sweatlasqa-pilot30-shuffle2-sglang",
        task="runs/20260907T111756.126345861Z-openclaw-sweatlasqa-pilot30-shuffle2-amem-task-sglang",
        repo="runs/20260907T111756.126379309Z-openclaw-sweatlasqa-pilot30-shuffle2-amem-repo-sglang",
    ),
]


def find_task_dirs_ordered(run_dir: Path):
    items = []
    for entry in run_dir.iterdir():
        if not entry.is_dir():
            continue
        mo = TASK_DIR_RE.match(entry.name)
        if mo:
            items.append((mo.group(1), int(mo.group(2)), entry))
    items.sort(key=lambda x: x[1])
    return items


def count_assistant_turns(task_dir: Path):
    tel_dir = task_dir / "harness-turn-01" / "telemetry"
    if not tel_dir.is_dir():
        return None
    candidates = [p for p in tel_dir.glob("*.jsonl") if ".trajectory." not in p.name]
    if not candidates:
        return None
    n = 0
    for line in candidates[0].read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("type") == "message" and r.get("message", {}).get("role") == "assistant":
            n += 1
    return n


def agg_score_with_failures(task_dir: Path):
    """(agg_score, is_failure). agg_score is the real score if scored, else
    0.0 if the task failed outright (reward.txt present, no
    evaluation_results.json -- see pkg/benchmark/sweatlas/evaluate.go's
    scoreNoAnswer path), else None if truly missing (neither file).
    is_failure is explicit rather than inferred from agg_score==0, since a
    legitimately-scored task could in principle also score exactly 0."""
    eval_path = task_dir / "evaluation" / "evaluation_results.json"
    if eval_path.is_file():
        return json.loads(eval_path.read_text()).get("agg_score"), False
    reward_path = task_dir / "evaluation" / "reward.txt"
    return (0.0, True) if reward_path.is_file() else (None, False)


def build_dataset(qa_root: Path) -> pd.DataFrame:
    task_meta = ssa.load_task_metadata(qa_root)
    rows = []
    for rep in REPLICATES:
        dirs = {"control": Path(rep["control"]), "task-scope": Path(rep["task"]), "repo-scope": Path(rep["repo"])}

        # Position is defined by the CONTROL arm's actual run order (the
        # task dir's own numeric suffix, which is the harness's real run
        # sequence -- confirmed to disagree with alphabetical task-ID order
        # for a few tasks per replicate, so that is not a safe substitute).
        # All three arms in a replicate share the same task list/order by
        # design (enforced by TestSWEAtlasQAStudyArmsDifferOnlyInAMEM), so
        # control's order is the correct reference for every arm.
        #
        # "position" is reset per repository (1st task run against THIS
        # repo, 2nd, ...), preserving execution order within that repo --
        # NOT reset by re-sorting alphabetically or by dataset order. This
        # matters because the two repos in this study run as two
        # back-to-back blocks (confirmed: positions 1-15 are always
        # simple-login/app and 16-30 always paperless-ngx/paperless-ngx, in
        # ALL THREE replicates, only the order *within* each block is
        # shuffled) -- a raw global 1..30 position is therefore perfectly
        # confounded with repo identity, while within-repo position pools
        # "the Nth task seen from a given repo" across both repos into one
        # point, cancelling that confound out.
        # "global_position" (1..30, not reset) is kept alongside for
        # reference/diagnostics but is not what the longitudinal plots use.
        global_order = [task_id for task_id, suffix, _ in find_task_dirs_ordered(dirs["control"])]
        repo_order: dict[str, list[str]] = {}
        for task_id in global_order:
            repository, _ = task_meta.get(task_id, (None, None))
            if repository:
                repo_order.setdefault(repository, []).append(task_id)

        for arm, run_dir in dirs.items():
            for task_id, suffix, task_dir in find_task_dirs_ordered(run_dir):
                agg, is_failure = agg_score_with_failures(task_dir)
                tokens = ssa.load_session_tokens(task_dir) or {}
                turns = count_assistant_turns(task_dir)
                repository, _ = task_meta.get(task_id, (None, None))
                global_position = (global_order.index(task_id) + 1) if task_id in global_order else None
                order = repo_order.get(repository)
                position = (order.index(task_id) + 1) if order and task_id in order else None
                rows.append({
                    "replicate": rep["name"],
                    "arm": arm,
                    "task_id": task_id,
                    "repository": repository,
                    "position": position,
                    "global_position": global_position,
                    "agg_score": agg,
                    "is_failure": is_failure,
                    "inputTokens": tokens.get("inputTokens"),
                    "outputTokens": tokens.get("outputTokens"),
                    "runtimeSec": (tokens.get("runtimeMs") or None) and tokens["runtimeMs"] / 1000.0,
                    "turns": turns,
                })
    df = pd.DataFrame(rows)
    df["arm"] = pd.Categorical(df["arm"], categories=ARM_ORDER, ordered=True)
    return df


def restrict_to_common_success(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only (replicate, task_id) pairs that succeeded (no is_failure)
    in ALL three arms -- the same fixed task set is then compared for every
    arm, so no arm gets an easier surviving subset than another. This is a
    paired comparison of answer quality; it does NOT capture the
    reliability cost of each arm (see docs/benchmarks/swe-atlas-qa.md,
    "pilot30 amem study" section, failure-accounting note) -- an arm that
    fails more often is simply excluded from more of its own denominator,
    same as the others, rather than penalized for failing more."""
    ok_counts = (~df["is_failure"]).groupby([df["replicate"], df["task_id"]]).sum()
    common = ok_counts[ok_counts == len(ARM_ORDER)].index
    keep = pd.MultiIndex.from_frame(df[["replicate", "task_id"]]).isin(common)
    return df[keep].copy()


def fit_slopes(df, y):
    """Least-squares slope of mean(y) vs position, per arm -- the same
    per-position-mean-then-fit approach used earlier in this study's manual
    analysis. Printed for a quick flat-vs-trending read alongside the plot."""
    out = {}
    for arm in ARM_ORDER:
        sub = df[df["arm"] == arm].dropna(subset=["position", y])
        means = sub.groupby("position")[y].mean()
        if len(means) < 2:
            out[arm] = (float("nan"), float("nan"))
            continue
        x = means.index.to_numpy(dtype=float)
        yv = means.to_numpy(dtype=float)
        mx, my = x.mean(), yv.mean()
        denom = ((x - mx) ** 2).sum()
        slope = ((x - mx) * (yv - my)).sum() / denom if denom else float("nan")
        out[arm] = (slope, my)
    return out


def add_position_bin(df, bin_size=5):
    """Bucket the 1..15 within-repo position into fixed-width bins (default
    1-5/6-10/11-15) so each point on the binned plots pools ~3x more
    observations (all positions in the bin x both repos x all replicates)
    than the single-position view -- trading position resolution for a
    less noisy per-bin mean, since 15 individual positions each averaged
    over only ~5-6 paired tasks was too thin to read a trend off of."""
    df = df.copy()
    max_pos = int(df["position"].max())
    bin_starts = list(range(1, max_pos + 1, bin_size))
    labels = [str(s) if bin_size == 1 else f"{s}-{min(s + bin_size - 1, max_pos)}" for s in bin_starts]
    edges = bin_starts + [max_pos + 1]
    df["position_bin"] = pd.cut(df["position"], bins=edges, labels=labels, right=False, include_lowest=True)
    return df, labels


def fit_slopes_binned(df, group_col, group_order, bin_labels, y):
    """Slope of mean(y) vs. bin index (0, 1, 2, ...) per group -- same
    least-squares approach as fit_slopes/fit_slopes_generic, just over bins
    instead of raw positions, for a slope figure that matches what's
    plotted rather than the finer-grained unbinned one."""
    out = {}
    for g in group_order:
        sub = df[df[group_col] == g].dropna(subset=["position_bin", y])
        means = sub.groupby("position_bin", observed=True)[y].mean().reindex(bin_labels)
        means = means.dropna()
        if len(means) < 2:
            out[g] = (float("nan"), float("nan"))
            continue
        x = np.array([bin_labels.index(b) for b in means.index], dtype=float)
        yv = means.to_numpy(dtype=float)
        mx, my = x.mean(), yv.mean()
        denom = ((x - mx) ** 2).sum()
        slope = ((x - mx) * (yv - my)).sum() / denom if denom else float("nan")
        out[g] = (slope, my)
    return out


DELTA_ORDER = ["task-scope - control", "repo-scope - control"]
DELTA_PALETTE = {"task-scope - control": ARM_PALETTE["task-scope"], "repo-scope - control": ARM_PALETTE["repo-scope"]}


def build_paired_deltas(df, y="agg_score"):
    """Same-task paired delta vs. control (repo-scope - control, task-scope
    - control), by (replicate, task_id, position) -- this cancels out the
    shared task-difficulty-by-position pattern that a same-position,
    different-task comparison would not, matching the paired methodology
    used throughout this study. One row per (replicate, task_id) per
    comparison, long-form for seaborn."""
    piv = df.pivot_table(index=["replicate", "task_id", "position"], columns="arm", values=y).reset_index()
    piv["task-scope - control"] = piv["task-scope"] - piv["control"]
    piv["repo-scope - control"] = piv["repo-scope"] - piv["control"]
    long = piv.melt(
        id_vars=["replicate", "task_id", "position"],
        value_vars=DELTA_ORDER, var_name="comparison", value_name="delta",
    ).dropna(subset=["delta"])
    long["comparison"] = pd.Categorical(long["comparison"], categories=DELTA_ORDER, ordered=True)
    return long


def fit_slopes_generic(df, group_col, group_order, y):
    """Same fitting logic as fit_slopes, generalized to any grouping
    column (arm, or a comparison label like 'repo-scope - control')."""
    out = {}
    for g in group_order:
        sub = df[df[group_col] == g].dropna(subset=["position", y])
        means = sub.groupby("position")[y].mean()
        if len(means) < 2:
            out[g] = (float("nan"), float("nan"))
            continue
        x = means.index.to_numpy(dtype=float)
        yv = means.to_numpy(dtype=float)
        mx, my = x.mean(), yv.mean()
        denom = ((x - mx) ** 2).sum()
        slope = ((x - mx) * (yv - my)).sum() / denom if denom else float("nan")
        out[g] = (slope, my)
    return out


def plot_delta_vs_position(df, out_dir, bin_size=5):
    """agg_score delta vs. control (paired, same task), by BINNED position
    within repo -- isolates whether a memory arm's edge over control
    grows/shrinks with repeated exposure to a repo, independent of shared
    task-difficulty trends that a raw per-arm plot can't separate out (see
    plot_longitudinal's docstring). Binned rather than per-position: each
    raw position was only ~5-6 paired tasks (2 repos x 3 replicates, minus
    failures), too thin to read a trend off of -- pooling bin_size
    positions per bin multiplies that to ~15-18."""
    long = build_paired_deltas(df, "agg_score")
    long, bin_labels = add_position_bin(long, bin_size=bin_size)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.axhline(0, color="black", linewidth=1, alpha=0.4, zorder=1)
    sns.pointplot(
        data=long, x="position_bin", y="delta", hue="comparison", order=bin_labels, hue_order=DELTA_ORDER,
        palette=DELTA_PALETTE, errorbar=None, dodge=0.15, ax=ax,
    )
    slopes = fit_slopes_binned(long, "comparison", DELTA_ORDER, bin_labels, "delta")
    subtitle = "  |  ".join(f"{c} slope={s:+.4f}/bin (mean {m:+.3f})" for c, (s, m) in slopes.items())
    ax.set_xlabel(f"task position within repo, binned (execution order, bin size={bin_size})", fontsize=13)
    ax.set_ylabel("agg_score delta vs. control\n(same task, paired)", fontsize=13)
    fig.suptitle("Accuracy delta vs. control, by binned task position within repo", y=1.0)
    ax.set_title(subtitle, fontsize=10.5)
    ax.legend(title="", loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"accuracy_delta_by_position_bin{bin_size}.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / f'accuracy_delta_by_position_bin{bin_size}.png'}  ({subtitle})")


def plot_longitudinal(df, out_dir, bin_size=5):
    """Metric vs. BINNED task position within its repo, in that repo's
    actual execution order (reset to 1 at the start of each repo's block,
    per build_dataset's "position" column) -- the "does it get
    better/cheaper with repeated exposure to a given repo" view. Uses the
    same common-success, equal-n-per-arm dataset as the bar charts, so a
    gap here isn't an artifact of one arm failing more at a given position.
    Binned rather than per-position: each raw position was only ~5-6 paired
    tasks (2 repos x 3 replicates, minus failures), too thin to read a
    trend off of -- pooling bin_size positions per bin multiplies that to
    ~15-18."""
    df, bin_labels = add_position_bin(df, bin_size=bin_size)

    # --- Accuracy vs position ---
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.pointplot(
        data=df, x="position_bin", y="agg_score", hue="arm", order=bin_labels, hue_order=ARM_ORDER,
        palette=ARM_PALETTE, errorbar=None, dodge=0.2, ax=ax,
    )
    ax.set_xlabel(f"task position within repo, binned (execution order, bin size={bin_size})")
    ax.set_ylabel("agg_score")
    slopes = fit_slopes_binned(df, "arm", ARM_ORDER, bin_labels, "agg_score")
    subtitle = "  |  ".join(f"{arm} slope={s:+.4f}/bin" for arm, (s, _) in slopes.items())
    fig.suptitle("Accuracy (agg_score) vs. binned task position within repo", y=0.99)
    ax.set_title(subtitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"accuracy_by_position_bin{bin_size}.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / f'accuracy_by_position_bin{bin_size}.png'}  ({subtitle})")

    # --- Efficiency vs position (2x2 grid) ---
    metrics = [
        ("inputTokens", "input tokens"),
        ("outputTokens", "output tokens"),
        ("runtimeSec", "runtime (s)"),
        ("turns", "assistant turns"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (col, label) in zip(axes.flat, metrics):
        sns.pointplot(
            data=df, x="position_bin", y=col, hue="arm", order=bin_labels, hue_order=ARM_ORDER,
            palette=ARM_PALETTE, errorbar=None, dodge=0.2, ax=ax, legend=False,
        )
        slopes = fit_slopes_binned(df, "arm", ARM_ORDER, bin_labels, col)
        subtitle = "  |  ".join(f"{arm}={s:+.3g}/bin" for arm, (s, _) in slopes.items())
        ax.set_title(f"{label}\n{subtitle}", fontsize=10)
        ax.set_xlabel("position within repo, binned")
        ax.set_ylabel(label)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if not handles:
        # legend=False above suppressed per-axes legends; rebuild one from the last lineplot's lines instead.
        handles = [plt.Line2D([0], [0], color=ARM_PALETTE[a], marker="o") for a in ARM_ORDER]
        labels = ARM_ORDER
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Efficiency metrics vs. binned task position within repo", y=1.08)
    fig.tight_layout()
    fig.savefig(out_dir / f"efficiency_by_position_bin{bin_size}.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / f'efficiency_by_position_bin{bin_size}.png'}")


def annotate_bars(ax, df, y):
    stats = df.groupby("arm", observed=True)[y].agg(["mean", "std"])
    for i, arm in enumerate(ARM_ORDER):
        if arm in stats.index:
            mean, std = stats.loc[arm, "mean"], stats.loc[arm, "std"]
            top = mean + (std if pd.notna(std) else 0)
            ax.annotate(f"{mean:.3g}", (i, top), ha="center", va="bottom",
                        fontsize=9, xytext=(0, 4), textcoords="offset points")


def parse_task_windows(aries_log_path):
    """bare task_id (no -NNN occurrence suffix) -> (start Timestamp, end
    Timestamp), from the run's own "task started"/"task finished" log
    lines. aries.log's task_id field is the full directory name
    (task-<hash>-NNN); stripped here to match repo_order's bare task_id
    keys (from find_task_dirs_ordered, which returns TASK_DIR_RE's
    group(1) only). Tasks run at concurrency=1 (enforced by
    TestSWEAtlasQAStudyArmsDifferOnlyInAMEM) so these windows never
    overlap -- every memory-note timestamp falls inside exactly one."""
    starts, ends = {}, {}
    with open(aries_log_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            mo = TASK_DIR_RE.match(r.get("task_id", ""))
            bare_id = mo.group(1) if mo else r.get("task_id")
            if r.get("msg") == "task started":
                starts[bare_id] = pd.Timestamp(r["time"])
            elif r.get("msg") == "task finished":
                ends[bare_id] = pd.Timestamp(r["time"])
    return {tid: (starts[tid], ends[tid]) for tid in starts if tid in ends}


def task_id_for_timestamp(ts, windows):
    for tid, (start, end) in windows.items():
        if start <= ts <= end:
            return tid
    return None


def build_memory_growth(qa_root, out_dir):
    """Per-position memory size, reconstructed from each note's own
    "timestamp" field in the amem-memory.json exports:
    - repo-scope: ONE export per repository (the shared store is only
      exported once, when it's torn down at the end of the whole run --
      see amem_pool.go's CleanupSharedAMEMRepoScope), so growth over the
      run has to be reconstructed by bucketing each note's timestamp into
      whichever task's [started, finished) window contains it, then
      cumulative-summing by that task's within-repo position.
    - task-scope: one export PER TASK (store torn down every task), so
      there's no cumulative growth to reconstruct -- this is plotted
      un-cumulative, per task, as the contrast case (a flat, non-growing
      size at each position, since every task starts from empty)."""
    task_meta = ssa.load_task_metadata(qa_root)
    repo_rows, task_rows = [], []

    for rep in REPLICATES:
        control_dir = Path(rep["control"])
        repo_order: dict[str, list[str]] = {}
        for task_id, suffix, _ in find_task_dirs_ordered(control_dir):
            repository, _ = task_meta.get(task_id, (None, None))
            if repository:
                repo_order.setdefault(repository, []).append(task_id)

        # --- repo-scope: reconstruct cumulative growth from note timestamps ---
        repo_dir = Path(rep["repo"])
        windows = parse_task_windows(repo_dir / "aries.log")
        for mem_path in sorted((repo_dir / "amem-memory").glob("*/amem-memory.json")):
            notes = json.loads(mem_path.read_text())
            per_task_added = {}
            for note in notes:
                ts = pd.Timestamp(note["payload"]["timestamp"])
                tid = task_id_for_timestamp(ts, windows)
                if tid is None:
                    continue
                d = per_task_added.setdefault(tid, {"notes": 0, "chars": 0})
                d["notes"] += 1
                d["chars"] += len(note["payload"].get("content", ""))
            # attribute each task's additions to its within-repo position, using
            # whichever repository these task_ids actually belong to (should be one repo)
            positions_seen = set()
            for tid, added in per_task_added.items():
                repository, _ = task_meta.get(tid, (None, None))
                order = repo_order.get(repository)
                if not order or tid not in order:
                    continue
                pos = order.index(tid) + 1
                positions_seen.add((repository, pos))
                repo_rows.append({
                    "replicate": rep["name"], "repository": repository, "position": pos,
                    "notes_added": added["notes"], "chars_added": added["chars"],
                })
            # positions where nothing was added still count as 0 added, so the
            # cumulative curve doesn't skip a position
            if per_task_added:
                any_repo = next(iter(per_task_added))
                repository, _ = task_meta.get(any_repo, (None, None))
                order = repo_order.get(repository, [])
                for pos in range(1, len(order) + 1):
                    if (repository, pos) not in positions_seen:
                        repo_rows.append({
                            "replicate": rep["name"], "repository": repository, "position": pos,
                            "notes_added": 0, "chars_added": 0,
                        })

        # --- task-scope: per-task export size, NOT cumulative (contrast case) ---
        task_dir_run = Path(rep["task"])
        for task_id, suffix, task_dir in find_task_dirs_ordered(task_dir_run):
            mem_path = task_dir / "amem-memory.json"
            repository, _ = task_meta.get(task_id, (None, None))
            order = repo_order.get(repository)
            pos = (order.index(task_id) + 1) if order and task_id in order else None
            if pos is None:
                continue
            if mem_path.is_file():
                notes = json.loads(mem_path.read_text())
                n_notes = len(notes)
                n_chars = sum(len(n["payload"].get("content", "")) for n in notes)
            else:
                n_notes = n_chars = 0
            task_rows.append({
                "replicate": rep["name"], "repository": repository, "position": pos,
                "notes": n_notes, "chars": n_chars,
            })

    repo_df = pd.DataFrame(repo_rows).sort_values(["replicate", "repository", "position"])
    repo_df["cum_notes"] = repo_df.groupby(["replicate", "repository"])["notes_added"].cumsum()
    repo_df["cum_chars"] = repo_df.groupby(["replicate", "repository"])["chars_added"].cumsum()
    task_df = pd.DataFrame(task_rows)

    repo_df.to_csv(out_dir / "repo_scope_memory_growth.csv", index=False)
    task_df.to_csv(out_dir / "task_scope_memory_sizes.csv", index=False)
    return repo_df, task_df


def plot_memory_growth(qa_root, out_dir):
    repo_df, task_df = build_memory_growth(qa_root, out_dir)
    max_pos = int(repo_df["position"].max())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    ax = axes[0]
    series_id = repo_df["replicate"] + " / " + repo_df["repository"]
    for key, sub in repo_df.assign(series=series_id).groupby("series"):
        ax.plot(sub["position"], sub["cum_notes"], color=ARM_PALETTE["repo-scope"], alpha=0.25, linewidth=1.2)
    mean_curve = repo_df.groupby("position")["cum_notes"].mean()
    ax.plot(mean_curve.index, mean_curve.values, color=ARM_PALETTE["repo-scope"], linewidth=3, marker="o",
            label="mean across replicates x repos")
    ax.set_xlabel("task position within repo (execution order)")
    ax.set_ylabel("cumulative memory notes")
    ax.set_title("repo-scope: memory DOES accumulate\n(shared store, one export per repo, reconstructed from note timestamps)", fontsize=11)
    ax.set_xticks(range(1, max_pos + 1))
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    series_id2 = task_df["replicate"] + " / " + task_df["repository"]
    for key, sub in task_df.assign(series=series_id2).groupby("series"):
        ax.plot(sub["position"], sub["notes"], color=ARM_PALETTE["task-scope"], alpha=0.25, linewidth=1.2, marker=".")
    mean_curve2 = task_df.groupby("position")["notes"].mean()
    ax.plot(mean_curve2.index, mean_curve2.values, color=ARM_PALETTE["task-scope"], linewidth=3, marker="o",
            label="mean across replicates x repos")
    ax.set_xlabel("task position within repo (execution order)")
    ax.set_ylabel("memory notes in THIS task's store (not cumulative)")
    ax.set_title("task-scope: memory does NOT accumulate\n(fresh store every task, torn down after -- size shown is per-task, not running total)", fontsize=11)
    ax.set_xticks(range(1, max_pos + 1))
    ax.set_ylim(0, max(task_df["notes"].max(), mean_curve.max() if len(mean_curve) else 5) * 1.15)
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Memory size vs. task position within repo", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "memory_growth_by_position.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'memory_growth_by_position.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-size", type=int, default=3,
                         help="positions per bin for the longitudinal/delta-by-position plots (default: 3)")
    args = parser.parse_args()

    qa_root = REPO_ROOT / ".cache" / "swe-atlas-qa"
    df_all = build_dataset(qa_root)

    out_dir = REPO_ROOT / "scripts" / "out"
    out_dir.mkdir(exist_ok=True)
    df_all.to_csv(out_dir / "pilot30_combined_replicates.csv", index=False)
    print(f"wrote {out_dir / 'pilot30_combined_replicates.csv'} ({len(df_all)} rows)")
    print("--- all tasks, failures counted as agg_score=0 ---")
    print(df_all.groupby("arm", observed=True)[["agg_score", "inputTokens", "outputTokens", "runtimeSec", "turns"]]
          .agg(["mean", "std", "count"]))

    df = restrict_to_common_success(df_all)
    n_dropped_per_replicate = df_all.groupby("replicate")["task_id"].nunique() - df.groupby("replicate")["task_id"].nunique()
    print(f"\n--- restricted to tasks that succeeded in ALL 3 arms: {df['task_id'].nunique()} of "
          f"{df_all['task_id'].nunique()} unique tasks kept per replicate on average "
          f"(dropped per replicate: {dict(n_dropped_per_replicate)}) ---")
    print(df.groupby("arm", observed=True)[["agg_score", "inputTokens", "outputTokens", "runtimeSec", "turns"]]
          .agg(["mean", "std", "count"]))
    df.to_csv(out_dir / "pilot30_common_success_replicates.csv", index=False)
    print(f"wrote {out_dir / 'pilot30_common_success_replicates.csv'} ({len(df)} rows)")

    sns.set_theme(style="whitegrid", context="talk")

    # --- Accuracy figure ---
    fig, ax = plt.subplots(figsize=(7, 6.5))
    sns.barplot(
        data=df, x="arm", y="agg_score", order=ARM_ORDER,
        hue="arm", palette=ARM_PALETTE, legend=False,
        errorbar="sd", capsize=0.15, ax=ax,
    )
    annotate_bars(ax, df, "agg_score")
    n = len(df) // len(ARM_ORDER)
    ax.set_xlabel("")
    ax.set_ylabel("agg_score")
    ax.set_ylim(0, 1.15)
    fig.suptitle("Accuracy (agg_score) by arm", y=0.99)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_arm.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'accuracy_by_arm.png'}")

    # --- Efficiency figure (2x2 grid) ---
    metrics = [
        ("inputTokens", "input tokens"),
        ("outputTokens", "output tokens"),
        ("runtimeSec", "runtime (s)"),
        ("turns", "assistant turns"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (col, label) in zip(axes.flat, metrics):
        sns.barplot(
            data=df, x="arm", y=col, order=ARM_ORDER,
            hue="arm", palette=ARM_PALETTE, legend=False,
            errorbar="sd", capsize=0.15, ax=ax,
        )
        annotate_bars(ax, df, col)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(out_dir / "efficiency_by_arm.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'efficiency_by_arm.png'}")

    plot_longitudinal(df, out_dir, bin_size=args.bin_size)
    plot_delta_vs_position(df, out_dir, bin_size=args.bin_size)
    plot_memory_growth(qa_root, out_dir)


if __name__ == "__main__":
    main()
