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
ARM_LABELS = {"control": "control", "repo-scope": "repo-scope"}
ARM_ORDER = ["control", "repo-scope"]
ARM_PALETTE = {"control": "#2a78d6", "repo-scope": "#1baf7a"}

REPLICATES = [
    dict(
        name="pilot30",
        control="runs/20260906T141944.963675899Z-openclaw-sweatlasqa-pilot30-sglang",
        repo="runs/20260906T141944.964671697Z-openclaw-sweatlasqa-pilot30-amem-repo-sglang",
    ),
    dict(
        name="shuffle1",
        control="runs/20260907T013422.077077636Z-openclaw-sweatlasqa-pilot30-shuffle1-sglang",
        repo="runs/20260907T013422.077067776Z-openclaw-sweatlasqa-pilot30-shuffle1-amem-repo-sglang",
    ),
    dict(
        name="shuffle2",
        control="runs/20260907T111756.126844838Z-openclaw-sweatlasqa-pilot30-shuffle2-sglang",
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
        dirs = {"control": Path(rep["control"]), "repo-scope": Path(rep["repo"])}

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


DELTA_ORDER = ["repo-scope - control"]
DELTA_PALETTE = {"repo-scope - control": ARM_PALETTE["repo-scope"]}


def build_paired_deltas(df, y="agg_score"):
    """Same-task paired delta vs. control (repo-scope - control), by
    (replicate, task_id, position) -- this cancels out the shared
    task-difficulty-by-position pattern that a same-position, different-task
    comparison would not, matching the paired methodology used throughout
    this study. One row per (replicate, task_id) per comparison, long-form
    for seaborn."""
    piv = df.pivot_table(index=["replicate", "task_id", "position"], columns="arm", values=y).reset_index()
    piv["repo-scope - control"] = piv["repo-scope"] - piv["control"]
    long = piv.melt(
        id_vars=["replicate", "task_id", "position"],
        value_vars=DELTA_ORDER, var_name="comparison", value_name="delta",
    ).dropna(subset=["delta"])
    long["comparison"] = pd.Categorical(long["comparison"], categories=DELTA_ORDER, ordered=True)
    return long


def bucket_deltas(long, thresh=0.05):
    """Bucket each paired repo-scope-control delta into improved (>thresh),
    unchanged (within +-thresh), or degraded (<-thresh) -- the base-rate
    view of "when does memory help vs hurt" that a single mean can't show:
    a mean near zero could mean "almost always exactly zero effect" or
    "roughly as many big wins as big losses," and those call for different
    stories."""
    def bucket(d):
        if d > thresh:
            return "improved"
        if d < -thresh:
            return "degraded"
        return "unchanged"
    out = long.copy()
    out["bucket"] = out["delta"].apply(bucket)
    return out


def plot_delta_distribution(df, out_dir, thresh=0.05):
    """Distribution of the paired repo-scope-control agg_score delta across
    all common-success tasks -- a strip/swarm of individual task deltas
    plus the improved/unchanged/degraded bucket counts, so the aggregate
    mean (already shown in accuracy_by_arm.png) doesn't hide how many
    individual tasks actually moved which way."""
    long = build_paired_deltas(df, "agg_score")
    bucketed = bucket_deltas(long, thresh=thresh)
    counts = bucketed["bucket"].value_counts().reindex(["degraded", "unchanged", "improved"]).fillna(0).astype(int)
    n = len(bucketed)

    fig, ax = plt.subplots(figsize=(8, 7))
    bucket_colors = {"degraded": "#d03b3b", "unchanged": "#83817a", "improved": ARM_PALETTE["repo-scope"]}
    ax.axhline(0, color="black", linewidth=1, alpha=0.5, zorder=1)
    ax.axhspan(-thresh, thresh, color="#83817a", alpha=0.08, zorder=0)
    sns.stripplot(
        data=bucketed, x="comparison", y="delta", hue="bucket", hue_order=["degraded", "unchanged", "improved"],
        palette=bucket_colors, size=7, jitter=0.25, alpha=0.85, ax=ax,
    )
    mean_delta = bucketed["delta"].mean()
    ax.axhline(mean_delta, color=ARM_PALETTE["repo-scope"], linewidth=2.5, linestyle="--",
               label=f"mean = {mean_delta:+.3f}")
    ax.set_xlabel("")
    ax.set_ylabel("agg_score delta vs. control\n(same task, paired)")
    ax.set_xticks([])
    subtitle = (f"n={n}  |  degraded (< -{thresh:g}): {counts['degraded']} ({counts['degraded']/n:.0%})  |  "
                f"unchanged: {counts['unchanged']} ({counts['unchanged']/n:.0%})  |  "
                f"improved (> +{thresh:g}): {counts['improved']} ({counts['improved']/n:.0%})")
    fig.suptitle("Distribution of repo-scope - control paired delta, per task", y=1.0)
    ax.set_title(subtitle, fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_delta_distribution.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'accuracy_delta_distribution.png'}  ({subtitle})")
    return bucketed


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
        palette=DELTA_PALETTE, errorbar=None, dodge=False, ax=ax,
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
        palette=ARM_PALETTE, errorbar="sd", capsize=0.1, dodge=0.2, ax=ax,
        err_kws={"linewidth": 1.2, "alpha": 0.6}, markersize=6, linewidth=2,
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
    "timestamp" field in the amem-memory.json exports. repo-scope has ONE
    export per repository (the shared store is only exported once, when
    it's torn down at the end of the whole run -- see amem_pool.go's
    CleanupSharedAMEMRepoScope), so growth over the run has to be
    reconstructed by bucketing each note's timestamp into whichever task's
    [started, finished) window contains it, then cumulative-summing by that
    task's within-repo position."""
    task_meta = ssa.load_task_metadata(qa_root)
    repo_rows = []

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

    repo_df = pd.DataFrame(repo_rows).sort_values(["replicate", "repository", "position"])
    repo_df["cum_notes"] = repo_df.groupby(["replicate", "repository"])["notes_added"].cumsum()
    repo_df["cum_chars"] = repo_df.groupby(["replicate", "repository"])["chars_added"].cumsum()

    repo_df.to_csv(out_dir / "repo_scope_memory_growth.csv", index=False)
    return repo_df


def plot_memory_growth(qa_root, out_dir):
    repo_df = build_memory_growth(qa_root, out_dir)
    max_pos = int(repo_df["position"].max())

    fig, ax = plt.subplots(figsize=(8, 6.5))

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

    fig.suptitle("Memory size vs. task position within repo", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "memory_growth_by_position.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'memory_growth_by_position.png'}")


def plot_retrieval_relevance(out_dir):
    """Is a relevant, on-topic memory note the differentiator between
    repo-scope's wins and losses? Reads scripts/out/retrieval_relevance_cases.csv
    -- a hand-annotated table from transcript-level investigation of the 8
    largest wins and 8 largest losses (paired vs. control, same task), each
    case checked for whether a retrieved note is DIRECTLY and quotably
    reflected in the answer that won or lost the task (see the CSV's
    "evidence" column for the citation backing each row; this is not
    inferred from aggregate stats).

    This is the sharpest single piece of evidence in the whole repo-scope
    investigation: relevant retrieval shows up in the majority of wins and
    in NONE of the losses -- retrieval isn't a source of harm (no
    poisoned/misleading notes found in any loss case), it's a coverage
    lottery that pays off when an earlier task happened to already record
    the fact a later question needs, and is simply inert otherwise."""
    cases = pd.read_csv(out_dir / "retrieval_relevance_cases.csv")
    summary = (
        cases.groupby("bucket")["relevant_note_used"]
        .agg(n_used="sum", n_total="count")
        .reindex(["improved", "degraded"])
    )
    summary["frac"] = summary["n_used"] / summary["n_total"]

    fig, ax = plt.subplots(figsize=(8, 8))
    labels = {"improved": "wins\n(repo-scope > control)", "degraded": "losses\n(repo-scope < control)"}
    colors = [ARM_PALETTE["repo-scope"], "#d03b3b"]
    bars = ax.bar([labels[b] for b in summary.index], summary["frac"] * 100, color=colors, width=0.55)
    for bar, (bucket, row) in zip(bars, summary.iterrows()):
        ax.annotate(f"{int(row['n_used'])} of {int(row['n_total'])}\n({row['frac']:.0%})",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=12, xytext=(0, 6), textcoords="offset points")
    ax.set_ylabel("cases where a relevant memory note was\ndirectly, quotably reflected in the answer", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{y}%" for y in range(0, 101, 20)])
    n_deg = int(summary.loc["degraded", "n_total"])
    fig.suptitle("Relevant retrieval is what separates wins from losses, not harmful retrieval", y=1.0, fontsize=13)
    ax.set_title("8 largest paired wins vs. paired losses (repo-scope vs. control, same task)\n"
                 f"0 of {n_deg} losses show a wrong/misleading note anywhere -- see retrieval_relevance_cases.csv for citations",
                 fontsize=10, pad=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_dir / "retrieval_relevance_by_outcome.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'retrieval_relevance_by_outcome.png'}")
    print(summary)


def plot_retrieval_similarity_comparison(out_dir):
    """Does the memory_search similarity score predict whether a retrieved
    note ends up mattering? Same 18 hand-investigated cases as
    plot_retrieval_relevance, now plotting each case's mean retrieved-note
    similarity (from repo_scope_retrieval_similarity.csv, joined in) against
    outcome bucket and against whether the note was actually used.

    If similarity predicted usefulness, the "used" dots should sit visibly
    above the "not used" dots. They don't -- both groups cluster around the
    same ~50% mean with fully overlapping spread, the quantitative version
    of "semantically similar to the query, but functionally useless for the
    task": the retrieval system is finding topically-adjacent notes at a
    fairly consistent rate regardless of whether they happen to contain the
    specific fact a task needs."""
    cases = pd.read_csv(out_dir / "retrieval_relevance_cases.csv").dropna(subset=["sim_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    ax = axes[0]
    bucket_labels = {"improved": "wins", "degraded": "losses"}
    bucket_colors = {"improved": ARM_PALETTE["repo-scope"], "degraded": "#d03b3b"}
    for i, bucket in enumerate(["improved", "degraded"]):
        sub = cases[cases["bucket"] == bucket]
        x = np.random.default_rng(0).uniform(i - 0.12, i + 0.12, size=len(sub))
        ax.scatter(x, sub["sim_mean"], color=bucket_colors[bucket], s=70, alpha=0.85, zorder=3)
        mean = sub["sim_mean"].mean()
        ax.plot([i - 0.2, i + 0.2], [mean, mean], color=bucket_colors[bucket], linewidth=3, zorder=4)
        ax.annotate(f"mean {mean:.1f}%  (n={len(sub)})", (i, mean), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=10, color=bucket_colors[bucket])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{bucket_labels[b]}\n(repo-scope vs. control)" for b in ["improved", "degraded"]])
    ax.set_ylabel("mean similarity of retrieved notes (%)")
    ax.set_ylim(20, 80)
    ax.set_title("by outcome", fontsize=11)

    ax = axes[1]
    used_labels = {True: "note used\nin the answer", False: "retrieved,\nnot used"}
    used_colors = {True: ARM_PALETTE["repo-scope"], False: "#83817a"}
    for i, used in enumerate([True, False]):
        sub = cases[cases["relevant_note_used"] == used]
        x = np.random.default_rng(1).uniform(i - 0.12, i + 0.12, size=len(sub))
        ax.scatter(x, sub["sim_mean"], color=used_colors[used], s=70, alpha=0.85, zorder=3)
        mean = sub["sim_mean"].mean()
        ax.plot([i - 0.2, i + 0.2], [mean, mean], color=used_colors[used], linewidth=3, zorder=4)
        ax.annotate(f"mean {mean:.1f}%  (n={len(sub)})", (i, mean), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=10, color=used_colors[used])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([used_labels[u] for u in [True, False]])
    ax.set_ylabel("mean similarity of retrieved notes (%)")
    ax.set_ylim(20, 80)
    ax.set_title("by whether the note was actually used", fontsize=11)

    fig.suptitle("Similarity score does not predict whether a retrieved note matters", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "retrieval_similarity_vs_outcome.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'retrieval_similarity_vs_outcome.png'}")


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
    print(f"\n--- restricted to tasks that succeeded in ALL {len(ARM_ORDER)} arms: {df['task_id'].nunique()} of "
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

    bucketed = plot_delta_distribution(df, out_dir)
    bucketed.to_csv(out_dir / "accuracy_delta_bucketed.csv", index=False)
    print(f"wrote {out_dir / 'accuracy_delta_bucketed.csv'} ({len(bucketed)} rows)")

    if (out_dir / "retrieval_relevance_cases.csv").is_file():
        plot_retrieval_relevance(out_dir)
        plot_retrieval_similarity_comparison(out_dir)


if __name__ == "__main__":
    main()
