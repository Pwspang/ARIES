"""Seaborn/matplotlib plots for "Recent, Not Relevant" -- recency, lexical relevance, and
call-vs-output necessity by tool type.

Same environment constraint as plot_sparsity_sns.py: this sandbox has no matplotlib/seaborn/
pandas/numpy and no pip to install them, so this script is untested here. Run it in an
environment that has those four packages (`pip install matplotlib seaborn pandas numpy`).
Reads results/recency_relevance_rows.json, produced by recency_relevance.py.

Usage:
  python plot_recency_relevance_sns.py --rows results/recency_relevance_rows.json --out-dir plots/
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)
KEPT_COLOR = "#2a78d6"
DROPPED_COLOR = "#eb6834"
TOOL_ORDER = ["web_search", "web_fetch", "mixed/other"]


def load(rows_path):
    rows = json.load(open(rows_path))
    df = pd.DataFrame(rows)
    df["status"] = df["kept_any"].map({True: "kept", False: "dropped"})
    return df


def plot_recency_bars(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=df,
        x="tool_type",
        y="rounds_back",
        hue="status",
        order=TOOL_ORDER,
        hue_order=["kept", "dropped"],
        palette={"kept": KEPT_COLOR, "dropped": DROPPED_COLOR},
        estimator="mean",
        errorbar=None,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("mean rounds back")
    ax.set_title("Recency: how far back is kept vs. dropped context?")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/recency_bars.png", dpi=150)
    plt.close(fig)


def plot_relevance_bars(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=df,
        x="tool_type",
        y="relevance",
        hue="status",
        order=TOOL_ORDER,
        hue_order=["kept", "dropped"],
        palette={"kept": KEPT_COLOR, "dropped": DROPPED_COLOR},
        estimator="mean",
        errorbar=None,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("mean lexical relevance")
    ax.set_title("Relevance to the current search: kept vs. dropped")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/relevance_bars.png", dpi=150)
    plt.close(fig)


def plot_scatter(df, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sns.scatterplot(
        data=df,
        x="rounds_back",
        y="relevance",
        hue="status",
        hue_order=["kept", "dropped"],
        palette={"kept": KEPT_COLOR, "dropped": DROPPED_COLOR},
        alpha=0.7,
        s=60,
        ax=ax,
    )
    ax.set_xlabel("rounds back (recency)")
    ax.set_ylabel("relevance to current search")
    ax.set_title("Recency vs. relevance, per candidate round")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/recency_vs_relevance_scatter.png", dpi=150)
    plt.close(fig)


WHOLE_COLOR = "#1baf7a"


def plot_tooltype_call_output(out_dir):
    """Necessity rate for every unit kind -- tool types split into call/output, plus the three
    atomic (unsplit) units: compaction, task instruction, system prompt. From the aggregate
    necessity table (not raw per-round rows -- a different unit of analysis than the other
    three plots). x-axis ordered by category (atomic units on the outside, tools in the middle)
    to match the report's chart."""
    data = pd.DataFrame(
        [
            {"unit": "compaction", "which": "whole", "necessity": 0.595},
            {"unit": "task", "which": "whole", "necessity": 0.543},
            {"unit": "web_fetch", "which": "call", "necessity": 0.612},
            {"unit": "web_fetch", "which": "output", "necessity": 0.343},
            {"unit": "read", "which": "call", "necessity": 0.426},
            {"unit": "read", "which": "output", "necessity": 0.630},
            {"unit": "web_search", "which": "call", "necessity": 0.381},
            {"unit": "web_search", "which": "output", "necessity": 0.429},
            {"unit": "mixed", "which": "call", "necessity": 0.304},
            {"unit": "mixed", "which": "output", "necessity": 0.222},
            {"unit": "system prompt", "which": "whole", "necessity": 0.377},
        ]
    )
    order = ["compaction", "task", "web_fetch", "read", "web_search", "mixed", "system prompt"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(
        data=data,
        x="unit",
        y="necessity",
        hue="which",
        order=order,
        hue_order=["call", "output", "whole"],
        palette={"call": KEPT_COLOR, "output": DROPPED_COLOR, "whole": WHOLE_COLOR},
        dodge=True,
        ax=ax,
    )
    for container in ax.containers:
        labels = [f"{v*100:.0f}%" if v > 0 else "" for v in container.datavalues]
        ax.bar_label(container, labels=labels, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("necessity rate")
    ax.set_title("Necessity by unit kind: call, output, or whole unit")
    ax.legend(title="", loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/tooltype_call_output.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="results/recency_relevance_rows.json")
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load(args.rows)
    plot_recency_bars(df, args.out_dir)
    plot_relevance_bars(df, args.out_dir)
    plot_scatter(df, args.out_dir)
    plot_tooltype_call_output(args.out_dir)
    print(f"wrote 4 plots to {args.out_dir}/ from {len(df)} candidate rounds")
