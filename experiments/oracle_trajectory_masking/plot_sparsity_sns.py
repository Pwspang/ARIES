"""Seaborn/matplotlib plots of the minimal-context sparsity results.

This sandbox has no matplotlib/seaborn/pandas/numpy installed and no pip to add them, so this
script is untested here -- run it in an environment that has those four packages (`pip install
matplotlib seaborn pandas numpy`).

Reads a rows file with columns dataset, task_id, turn_index, total_tokens, kept_tokens,
token_ratio, unit_ratio (produced by token_sparsity.py) and, for the average-kept-vs-available
panel, also minimal_units/total_units (merge these in from the turns.jsonl records -- see
render_strips.py / analyze_all.py for how those are read -- token_sparsity.py doesn't emit them
by default since it's scoped to token counting).

Usage:
  python plot_sparsity_sns.py --rows results/sparsity_merged_fanout.json --dataset fanout --out-dir plots/
"""
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)
UNIT_COLOR = "#2a78d6"
TOKEN_COLOR = "#eb6834"


def load(rows_path, dataset):
    rows = json.load(open(rows_path))
    if dataset:
        rows = [r for r in rows if r["dataset"] == dataset]
    df = pd.DataFrame(rows)
    df["label"] = df["task_id"] + " turn " + df["turn_index"].astype(str)
    return df


def plot_distribution(df, out_dir):
    long = pd.concat(
        [
            df[["unit_ratio"]].rename(columns={"unit_ratio": "ratio"}).assign(metric="unit-count"),
            df[["token_ratio"]].rename(columns={"token_ratio": "ratio"}).assign(metric="token-weighted"),
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(
        data=long,
        x="ratio",
        hue="metric",
        bins=10,
        binrange=(0, 1),
        multiple="dodge",
        shrink=0.85,
        palette={"unit-count": UNIT_COLOR, "token-weighted": TOKEN_COLOR},
        ax=ax,
    )
    ax.set_xlabel("fraction of context kept")
    ax.set_ylabel("turns")
    ax.set_title("Distribution of context kept per turn")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/distribution.png", dpi=150)
    plt.close(fig)


def plot_scatter(df, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=df, x="unit_ratio", y="token_ratio", color=UNIT_COLOR, alpha=0.75, s=60, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("unit-count ratio")
    ax.set_ylabel("token-weighted ratio")
    ax.set_title("Unit-count vs. token-weighted ratio, per turn")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/scatter.png", dpi=150)
    plt.close(fig)


def plot_avg_kept_vs_available(df, out_dir):
    """Tokens and units live on very different scales, so this is two panels (one axis each),
    not one chart with two y-axes."""
    if "minimal_units" not in df or "total_units" not in df:
        print("skipping avg_kept_vs_available -- rows file has no minimal_units/total_units columns")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, kept_col, total_col, label, fmt in [
        (axes[0], "kept_tokens", "total_tokens", "tokens", lambda v: f"{v:,.0f}"),
        (axes[1], "minimal_units", "total_units", "units", lambda v: f"{v:.2f}"),
    ]:
        bars = pd.DataFrame({"which": ["available", "kept"], "value": [df[total_col].mean(), df[kept_col].mean()]})
        sns.barplot(data=bars, x="which", y="value", color=UNIT_COLOR, ax=ax)
        ax.patches[0].set_alpha(0.35)
        for i, v in enumerate(bars["value"]):
            ax.text(i, v, fmt(v), ha="center", va="bottom", fontsize=9)
        ax.set_title(f"average {label}")
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Average kept vs. average available")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/avg_kept_vs_available.png", dpi=150)
    plt.close(fig)


def plot_best_worst_mean(df, out_dir):
    groups = pd.DataFrame(
        [
            {"case": "best (sparsest)", "metric": "unit-count", "value": df["unit_ratio"].min()},
            {"case": "best (sparsest)", "metric": "token-weighted", "value": df["token_ratio"].min()},
            {"case": "mean", "metric": "unit-count", "value": df["unit_ratio"].mean()},
            {"case": "mean", "metric": "token-weighted", "value": df["token_ratio"].mean()},
            {"case": "worst (densest)", "metric": "unit-count", "value": df["unit_ratio"].max()},
            {"case": "worst (densest)", "metric": "token-weighted", "value": df["token_ratio"].max()},
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        data=groups,
        x="case",
        y="value",
        hue="metric",
        palette={"unit-count": UNIT_COLOR, "token-weighted": TOKEN_COLOR},
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9)
    ax.set_ylabel("ratio")
    ax.set_xlabel("")
    ax.set_title("Best, worst, and mean case")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/best_worst_mean.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="results/token_sparsity_rows.json")
    ap.add_argument("--dataset", default=None, help="filter to one dataset (e.g. 'fanout' or 'aries'); omit for both")
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()

    import os

    os.makedirs(args.out_dir, exist_ok=True)
    df = load(args.rows, args.dataset)
    plot_distribution(df, args.out_dir)
    plot_scatter(df, args.out_dir)
    plot_avg_kept_vs_available(df, args.out_dir)
    plot_best_worst_mean(df, args.out_dir)
    print(f"wrote 3 plots to {args.out_dir}/ from {len(df)} turns")
