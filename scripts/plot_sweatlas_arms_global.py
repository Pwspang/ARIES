#!/usr/bin/env python3
"""Compare control (stateless baseline) vs. amem-global (cross-repository
memory) for the pilot30 study, pooling all three replicates (pilot30,
shuffle1, shuffle2).

This mirrors scripts/plot_sweatlas_arms.py (control vs. repo-scope) but for
the global-scope arm, and adds the analysis specific to global scope's
hypothesis: does memory written while working on repo A transfer to help on
repo B, a repository the agent has never touched before in this run?
Repo-scope memory can't do this by construction (its store is keyed per
repo); only global scope can. The sharpest test is "position 1 of the
SECOND repo" -- the very first task against a never-before-seen repository,
where a global store already holds notes from the first repo and a
repo-scope store would start from empty exactly like control.

Run with:
  uv run --with seaborn --with pandas --with matplotlib \\
      scripts/plot_sweatlas_arms_global.py
"""
import argparse
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
import plot_sweatlas_arms as base  # noqa: E402

ARM_LABELS = {"control": "control (stateless)", "global": "amem-global (cross-repo)"}
ARM_ORDER = ["control", "global"]
ARM_PALETTE = {"control": "#2a78d6", "global": "#c9781b"}

REPLICATES = [
    dict(
        name="pilot30",
        control="runs/20260906T141944.963675899Z-openclaw-sweatlasqa-pilot30-sglang",
        glob="runs/20260913T063321.846516847Z-openclaw-sweatlasqa-pilot30-amem-global-sglang",
    ),
    dict(
        name="shuffle1",
        control="runs/20260907T013422.077077636Z-openclaw-sweatlasqa-pilot30-shuffle1-sglang",
        glob="runs/20260914T035338.436341279Z-openclaw-sweatlasqa-pilot30-shuffle1-amem-global-sglang",
    ),
    dict(
        name="shuffle2",
        control="runs/20260907T111756.126844838Z-openclaw-sweatlasqa-pilot30-shuffle2-sglang",
        glob="runs/20260914T035345.813663967Z-openclaw-sweatlasqa-pilot30-shuffle2-amem-global-sglang",
    ),
]


def build_dataset(qa_root: Path) -> pd.DataFrame:
    task_meta = ssa.load_task_metadata(qa_root)
    rows = []
    for rep in REPLICATES:
        dirs = {"control": Path(rep["control"]), "global": Path(rep["glob"])}

        global_order = [tid for tid, suffix, _ in base.find_task_dirs_ordered(dirs["control"])]
        repo_order: dict[str, list[str]] = {}
        for task_id in global_order:
            repository, _ = task_meta.get(task_id, (None, None))
            if repository:
                repo_order.setdefault(repository, []).append(task_id)
        # which repo is "first" (seen first) vs "second" (unseen until block 2)
        repos_in_order = list(dict.fromkeys(
            task_meta.get(tid, (None, None))[0] for tid in global_order if task_meta.get(tid, (None, None))[0]
        ))
        repo_block_index = {r: i for i, r in enumerate(repos_in_order)}  # 0 = first repo, 1 = second repo

        for arm, run_dir in dirs.items():
            for task_id, suffix, task_dir in base.find_task_dirs_ordered(run_dir):
                agg, is_failure = base.agg_score_with_failures(task_dir)
                tokens = ssa.load_session_tokens(task_dir) or {}
                turns = base.count_assistant_turns(task_dir)
                repository, _ = task_meta.get(task_id, (None, None))
                global_position = (global_order.index(task_id) + 1) if task_id in global_order else None
                order = repo_order.get(repository)
                position = (order.index(task_id) + 1) if order and task_id in order else None
                rows.append({
                    "replicate": rep["name"],
                    "arm": arm,
                    "task_id": task_id,
                    "repository": repository,
                    "repo_block": repo_block_index.get(repository),
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
    ok_counts = (~df["is_failure"]).groupby([df["replicate"], df["task_id"]]).sum()
    common = ok_counts[ok_counts == len(ARM_ORDER)].index
    keep = pd.MultiIndex.from_frame(df[["replicate", "task_id"]]).isin(common)
    return df[keep].copy()


def mean_std_n(s):
    s = s.dropna()
    return s.mean(), s.std(), len(s)


def print_group(df, label, group_mask):
    sub = df[group_mask]
    print(f"\n--- {label} ---")
    for arm in ARM_ORDER:
        a = sub[sub["arm"] == arm]["agg_score"]
        m, sd, n = mean_std_n(a)
        print(f"  {ARM_LABELS[arm]:<28} n={n:<4} mean_agg_score={m:.3f}" if n else f"  {ARM_LABELS[arm]:<28} n=0")


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    qa_root = REPO_ROOT / ".cache" / "swe-atlas-qa"
    out_dir = REPO_ROOT / "scripts" / "out"
    out_dir.mkdir(exist_ok=True)

    df_all = build_dataset(qa_root)
    df_all.to_csv(out_dir / "pilot30_global_combined_replicates.csv", index=False)
    print(f"wrote {out_dir / 'pilot30_global_combined_replicates.csv'} ({len(df_all)} rows)")

    print("=== ALL TASKS (failures counted as agg_score=0) ===")
    print(df_all.groupby("arm", observed=True)[["agg_score", "inputTokens", "outputTokens", "runtimeSec", "turns"]]
          .agg(["mean", "std", "count"]))

    df = restrict_to_common_success(df_all)
    n_dropped = df_all.groupby("replicate")["task_id"].nunique() - df.groupby("replicate")["task_id"].nunique()
    print(f"\n=== RESTRICTED to common-success tasks: {df['task_id'].nunique()} kept "
          f"(dropped per replicate: {dict(n_dropped)}) ===")
    print(df.groupby("arm", observed=True)[["agg_score", "inputTokens", "outputTokens", "runtimeSec", "turns"]]
          .agg(["mean", "std", "count"]))
    df.to_csv(out_dir / "pilot30_global_common_success_replicates.csv", index=False)

    # --- The key cross-repo transfer test ---
    print("\n" + "=" * 78)
    print("CROSS-REPOSITORY TRANSFER TEST")
    print("Only global-scope memory can carry information from repo A into repo B;")
    print("repo-scope and control both start block 2 with nothing.")
    print("=" * 78)
    print_group(df_all, "repo_block==0 (first repo in run, all positions) -- ALL TASKS", df_all["repo_block"] == 0)
    print_group(df_all, "repo_block==1 (SECOND, never-before-seen repo, all positions) -- ALL TASKS", df_all["repo_block"] == 1)
    print_group(df_all, "repo_block==1 AND position==1 (very first task on the unseen repo) -- ALL TASKS",
                (df_all["repo_block"] == 1) & (df_all["position"] == 1))
    print_group(df_all, "repo_block==1 AND position>=2 (unseen repo, after its own warm-up) -- ALL TASKS",
                (df_all["repo_block"] == 1) & (df_all["position"] >= 2))

    # paired per-task deltas, second-repo block only
    print("\n--- Paired per-task deltas, repo_block==1 (second/unseen repo), all positions ---")
    piv = df_all[df_all["repo_block"] == 1].pivot_table(
        index=["replicate", "task_id", "position"], columns="arm", values="agg_score", observed=True
    ).reset_index()
    piv = piv.dropna(subset=["control", "global"])
    piv["delta"] = piv["global"] - piv["control"]
    print(f"  n={len(piv)}  mean_delta(global-control)={piv['delta'].mean():+.4f}  "
          f"wins={(piv['delta']>0.05).sum()} losses={(piv['delta']<-0.05).sum()} "
          f"unchanged={((piv['delta']>=-0.05)&(piv['delta']<=0.05)).sum()}")
    piv.to_csv(out_dir / "pilot30_global_crossrepo_paired_deltas.csv", index=False)

    print("\n--- Paired per-task deltas, repo_block==1, position==1 only (strongest transfer test) ---")
    piv1 = piv[piv["position"] == 1]
    print(f"  n={len(piv1)}  mean_delta(global-control)={piv1['delta'].mean():+.4f}" if len(piv1) else "  n=0")
    print(piv1[["replicate", "task_id", "control", "global", "delta"]].to_string(index=False))

    # --- Plots ---
    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(7, 6.5))
    sns.barplot(data=df, x="arm", y="agg_score", order=ARM_ORDER, hue="arm",
                palette=ARM_PALETTE, legend=False, errorbar="sd", capsize=0.15, ax=ax)
    base.annotate_bars(ax, df, "agg_score", arm_order=ARM_ORDER)
    ax.set_xticks(range(len(ARM_ORDER)))
    ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER])
    ax.set_xlabel("")
    ax.set_ylabel("agg_score")
    ax.set_ylim(0, 1.15)
    fig.suptitle("Overall accuracy: control vs. amem-global", y=0.99)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_arm_global.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {out_dir / 'accuracy_by_arm_global.png'}")

    # --- Efficiency figure (2x2 grid): input/output tokens, runtime, turns -- paired completed tasks ---
    metrics = [
        ("inputTokens", "input tokens"),
        ("outputTokens", "output tokens"),
        ("runtimeSec", "runtime (s)"),
        ("turns", "assistant turns"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (col, label) in zip(axes.flat, metrics):
        sns.barplot(data=df, x="arm", y=col, order=ARM_ORDER, hue="arm",
                    palette=ARM_PALETTE, legend=False, errorbar="sd", capsize=0.15, ax=ax)
        base.annotate_bars(ax, df, col, arm_order=ARM_ORDER)
        ax.set_xticks(range(len(ARM_ORDER)))
        ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER], fontsize=10)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel(label)
    fig.suptitle("Efficiency: control vs. amem-global (paired completed tasks)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "efficiency_by_arm_global.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'efficiency_by_arm_global.png'}")

    # cross-repo focused bar chart: first repo vs second repo, by arm
    df["repo_block_label"] = df["repo_block"].map({0: "1st repo (seen)", 1: "2nd repo (unseen until now)"})
    block_order = ["1st repo (seen)", "2nd repo (unseen until now)"]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.barplot(data=df, x="repo_block_label", y="agg_score", order=block_order, hue="arm", hue_order=ARM_ORDER,
                palette=ARM_PALETTE, errorbar="sd", capsize=0.1, ax=ax)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, [ARM_LABELS[a] for a in ARM_ORDER], title="", loc="lower center")

    stats_by_group = df.groupby(["repo_block_label", "arm"], observed=True)["agg_score"].agg(["mean", "std"])
    # seaborn draws bars hue-major: all x-groups for arm0, then all x-groups for arm1
    ordered = [stats_by_group.loc[(g, a)] for a in ARM_ORDER for g in block_order]
    for patch, row in zip(ax.patches, ordered):
        x = patch.get_x() + patch.get_width() / 2
        top = row["mean"] + (row["std"] if pd.notna(row["std"]) else 0)
        ax.annotate(f"{row['mean']:.3f} ± {row['std']:.3f}", (x, top), ha="center", va="bottom",
                    fontsize=9, xytext=(0, 4), textcoords="offset points")
    ax.set_xlabel("")
    ax.set_ylabel("agg_score")
    ax.set_ylim(0, 1.25)
    fig.suptitle("Accuracy by repo novelty: does cross-repo memory help on an unseen repo?", y=0.99)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_repo_novelty_global.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'accuracy_by_repo_novelty_global.png'}")

    # --- Paired completed tasks only (both arms produced a real judge score) ---
    fig, ax = plt.subplots(figsize=(7, 6.5))
    sns.barplot(data=df, x="arm", y="agg_score", order=ARM_ORDER, hue="arm",
                palette=ARM_PALETTE, legend=False, errorbar="sd", capsize=0.15, ax=ax)
    base.annotate_bars(ax, df, "agg_score", arm_order=ARM_ORDER)
    ax.set_xticks(range(len(ARM_ORDER)))
    ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER])
    ax.set_xlabel("")
    ax.set_ylabel("agg_score")
    ax.set_ylim(0, 1.15)
    fig.suptitle("Paired completed tasks: control vs. amem-global", y=0.99)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_arm_all_vs_completed_global.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'accuracy_by_arm_all_vs_completed_global.png'}")

    # --- Paired-delta strip plot on the unseen repo (completed-only, matches reported n=39) ---
    unseen = df[df["repo_block"] == 1]
    piv2 = unseen.pivot_table(index=["replicate", "task_id", "position"], columns="arm",
                               values="agg_score", observed=True).reset_index()
    piv2 = piv2.dropna(subset=["control", "global"])
    piv2["delta"] = piv2["global"] - piv2["control"]
    piv2.sort_values("delta").to_csv(out_dir / "accuracy_delta_unseen_repo_global.csv", index=False)
    print(f"wrote {out_dir / 'accuracy_delta_unseen_repo_global.csv'} ({len(piv2)} rows)")
    mean_delta = piv2["delta"].mean()
    std_delta = piv2["delta"].std()
    control_mean, control_std = piv2["control"].mean(), piv2["control"].std()
    global_mean, global_std = piv2["global"].mean(), piv2["global"].std()
    try:
        from scipy import stats
        t_stat, p_val = stats.ttest_rel(piv2["global"], piv2["control"])
        stat_str = f"paired t={t_stat:+.2f}, p={p_val:.2f}"
    except ImportError:
        stat_str = "(scipy unavailable for t-test)"
    wins = (piv2["delta"] > 0.05).sum()
    losses = (piv2["delta"] < -0.05).sum()
    unchanged = len(piv2) - wins - losses

    fig, ax = plt.subplots(figsize=(9, 5.8))
    bucket_color = piv2["delta"].apply(lambda d: ARM_PALETTE["global"] if d > 0.05
                                        else ("#d03b3b" if d < -0.05 else "#9a988f"))
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.3, 0.3, size=len(piv2))
    ax.scatter(piv2["delta"], jitter, c=bucket_color, s=70, alpha=0.85, zorder=3, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1, alpha=0.5, zorder=1)
    ax.axvline(mean_delta, color="#d03b3b", linewidth=2, linestyle="--", zorder=2,
               label=f"mean Δ = {mean_delta:+.3f} ± {std_delta:.3f}")
    ax.set_yticks([])
    ax.set_xlabel("agg_score delta (global − control), same task")
    ax.set_xlim(-1.05, 1.05)
    fig.suptitle("No accuracy edge on the unseen repository", y=1.0)
    ax.set_title(
        f"control = {control_mean:.3f} ± {control_std:.3f}   |   global = {global_mean:.3f} ± {global_std:.3f}\n"
        f"n={len(piv2)}  |  wins={wins} losses={losses} unchanged={unchanged}  |  {stat_str}",
        fontsize=11,
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_delta_unseen_repo_global.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'accuracy_delta_unseen_repo_global.png'}")


if __name__ == "__main__":
    main()
