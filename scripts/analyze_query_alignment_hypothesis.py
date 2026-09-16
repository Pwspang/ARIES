#!/usr/bin/env python3
"""Test the hypothesis that the agent-generated memory_search `query` string is
poorly aligned with the task's actual <question>, and that this misalignment
(not just amem's own similarity ranking) drives irrelevant retrieval and worse
task outcomes.

Joins scripts/out/memory_search_events_with_overlap.csv (call-level:
query_question_overlap, sim_mean, delta, bucket -- from
analyze_query_relevance_gap.py) with scripts/out/memory_search_relevance_judged.csv
(note-level: llm_relevant -- from judge_memory_relevance.py) on
(run_dir, task_id, occurrence, call_seq), neither of which currently correlates
query_question_overlap against llm_relevant or delta.

Usage:
  scripts/analyze_query_alignment_hypothesis.py \\
      --events scripts/out/memory_search_events_with_overlap.csv \\
      --judged scripts/out/memory_search_relevance_judged.csv
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def median_split(df: pd.DataFrame, col: str) -> pd.Series:
    med = df[col].median()
    return df[col].apply(lambda v: "high" if v >= med else "low")


def report_group_rates(df: pd.DataFrame, group_col: str, rate_col: str, label: str) -> None:
    print(f"\n=== {label}: {rate_col} rate by {group_col} ===")
    for g, sub in df.groupby(group_col, dropna=False):
        rated = sub[rate_col].dropna()
        rate = rated.mean() if len(rated) else float("nan")
        print(f"{str(g):<10} n={len(sub):>4} rated={len(rated):>4} rate={rate:.3f}")


def chi_square_test(df: pd.DataFrame, group_col: str, outcome_col: str) -> None:
    """Pearson's chi-square test of independence, computed by hand (no scipy).
    p-value via the Wilson-Hilferty chi-square approximation (accurate to ~1e-3
    for the small dof used here), which is adequate for this exploratory check."""
    ct = pd.crosstab(df[group_col], df[outcome_col].astype(bool))
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        print(f"(not enough variation to run chi-square on {group_col} x {outcome_col})")
        return
    observed = ct.values.astype(float)
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected = row_sums @ col_sums / total
    chi2 = ((observed - expected) ** 2 / expected).sum()
    dof = (ct.shape[0] - 1) * (ct.shape[1] - 1)
    # Wilson-Hilferty: (chi2/dof)^(1/3) approx N(1 - 2/(9dof), 2/(9dof))
    if dof > 0:
        h = 2.0 / (9 * dof)
        z = ((chi2 / dof) ** (1 / 3) - (1 - h)) / np.sqrt(h)
        p = 0.5 * math.erfc(z / math.sqrt(2))
    else:
        p = float("nan")
    print(f"chi2({group_col} x {outcome_col}) = {chi2:.3f}, dof={dof}, p ~ {p:.4f}, n = {int(total)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=Path, default=Path("scripts/out/memory_search_events_with_overlap.csv"))
    parser.add_argument("--judged", type=Path, default=Path("scripts/out/memory_search_relevance_judged.csv"))
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    judged = pd.read_csv(args.judged)

    key = ["run_dir", "task_id", "occurrence", "call_seq"]
    call_relevance = judged.groupby(key, as_index=False).agg(
        llm_relevant_rate=("llm_relevant", "mean"),
        n_notes_judged=("llm_relevant", "count"),
    )
    merged = events.merge(call_relevance, on=key, how="inner")
    merged["llm_relevant_majority"] = merged["llm_relevant_rate"] >= 0.5

    print(f"Joined {len(merged)} calls with judged notes (of {len(events)} total events, "
          f"{call_relevance.shape[0]} calls judged).")

    # 1. Direct test: low query_question_overlap -> lower llm_relevant rate?
    merged["overlap_group"] = median_split(merged, "query_question_overlap")
    merged["overlap_zero_group"] = merged["query_question_overlap"].fillna(0).apply(
        lambda v: "zero" if v == 0 else "nonzero"
    )
    report_group_rates(merged, "overlap_group", "llm_relevant_rate", "Median split")
    report_group_rates(merged, "overlap_zero_group", "llm_relevant_rate", "Zero vs nonzero overlap")
    chi_square_test(merged, "overlap_group", "llm_relevant_majority")
    chi_square_test(merged, "overlap_zero_group", "llm_relevant_majority")

    # 2. Control for amem's own similarity score: does overlap predict relevance/delta
    #    beyond sim_mean? Compare correlations and a similarity-matched tercile check.
    print("\n=== Correlations (controlling informally for sim_mean via terciles) ===")
    corr_cols = ["query_question_overlap", "sim_mean", "llm_relevant_rate", "delta"]
    print(merged[corr_cols].corr(numeric_only=True))

    merged["sim_tercile"] = pd.qcut(merged["sim_mean"], 3, labels=["low_sim", "mid_sim", "high_sim"], duplicates="drop")
    for tercile, sub in merged.groupby("sim_tercile", observed=True):
        sub_corr = sub[["query_question_overlap", "llm_relevant_rate"]].corr().iloc[0, 1]
        print(f"within {tercile}: n={len(sub)} corr(query_question_overlap, llm_relevant_rate) = {sub_corr:.3f}")

    # 3. Outcome-level check: overlap vs delta directly.
    print("\n=== Outcome-level: query_question_overlap vs delta ===")
    print(f"corr(query_question_overlap, delta) = {merged[['query_question_overlap', 'delta']].corr().iloc[0, 1]:.3f}")
    degraded = merged[merged["bucket"] == "degraded"].sort_values("query_question_overlap")
    print(f"\nLowest-overlap 'degraded' bucket calls (n={len(degraded)} total in bucket):")
    for _, row in degraded.head(10).iterrows():
        print(f"- {row['run_dir']} / {row['task_id']} occ={row['occurrence']} call={row['call_seq']} "
              f"delta={row['delta']:.3f} overlap={row['query_question_overlap']:.3f} "
              f"sim_mean={row['sim_mean']:.1f} relevant_rate={row['llm_relevant_rate']:.2f}")
        print(f"  query: {row['query']!r}")
        print(f"  question: {str(row['question'])[:200]!r}")

    # 4. Qualitative spot-check: 10 lowest-overlap cases overall, regardless of bucket.
    print("\n=== 10 lowest query_question_overlap cases (qualitative spot-check) ===")
    lowest = merged.sort_values("query_question_overlap").head(10)
    for _, row in lowest.iterrows():
        print(f"- overlap={row['query_question_overlap']:.3f} relevant_rate={row['llm_relevant_rate']:.2f} "
              f"bucket={row['bucket']} delta={row['delta']:.3f}")
        print(f"  query: {row['query']!r}")
        print(f"  question: {str(row['question'])[:200]!r}")

    out_path = args.events.parent / "query_alignment_hypothesis_merged.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
