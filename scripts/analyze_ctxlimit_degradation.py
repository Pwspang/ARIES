#!/usr/bin/env python3
"""Does limiting the context window hurt the stateless baseline more than
repo-scoped AMEM?

Joins the ctxlimit study (runs/ctxlimit/, context_window_tokens=96000 so
OpenClaw's own auto-compaction fires routinely) against the unconstrained
pilot30 study (runs/, no compaction pressure) for the same two arms
(control / repo-scope amem), and computes each arm's own accuracy
degradation (full - ctxlimit), then tests whether that degradation is
smaller for repo-scope than for control (difference-in-differences).

Run with:
  uv run --with seaborn --with pandas --with matplotlib --with scipy \\
      scripts/analyze_ctxlimit_degradation.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import summarize_sweatlas_arms as ssa  # noqa: E402
import plot_sweatlas_arms_ctxlimit as psac  # noqa: E402 (agg_score_with_failures, find_task_dirs_ordered)

ARM_ORDER = ["control", "repo-scope"]
ARM_PALETTE = {"control": "#2a78d6", "repo-scope": "#1baf7a"}
COND_ORDER = ["full", "ctxlimit"]
COND_PALETTE = {"full": "#83817a", "ctxlimit": "#d03b3b"}

# Unconstrained ("full") replicates -- from plot_sweatlas_arms.py's REPLICATES.
FULL_REPLICATES = [
    dict(name="pilot30", control="runs/20260906T141944.963675899Z-openclaw-sweatlasqa-pilot30-sglang",
         repo="runs/20260906T141944.964671697Z-openclaw-sweatlasqa-pilot30-amem-repo-sglang"),
    dict(name="shuffle1", control="runs/20260907T013422.077077636Z-openclaw-sweatlasqa-pilot30-shuffle1-sglang",
         repo="runs/20260907T013422.077067776Z-openclaw-sweatlasqa-pilot30-shuffle1-amem-repo-sglang"),
    dict(name="shuffle2", control="runs/20260907T111756.126844838Z-openclaw-sweatlasqa-pilot30-shuffle2-sglang",
         repo="runs/20260907T111756.126379309Z-openclaw-sweatlasqa-pilot30-shuffle2-amem-repo-sglang"),
]

# Ctxlimit replicates -- exactly plot_sweatlas_arms_ctxlimit.py's REPLICATES
# (one shared unshuffled control paired with 3 separate repo-scope reruns,
# plus one shuffle1 and one shuffle2 pair -- see that module's comment for
# why the shared control is a valid like-for-like comparison).
CTXLIMIT_REPLICATES = psac.REPLICATES


def classify_failure(task_dir: Path) -> str:
    """Why a failed instance never produced a gradeable answer. The harness's
    own give-up message ("Context overflow: prompt too large for the model")
    is written straight into gateway.log (not just run-result.json's
    final_response), so this only needs the task's own log -- no separate
    run-result.json parse. "context_overflow" is the ctxlimit condition
    doing exactly what it's designed to do (the agent ran out of budget and
    gave up); "gateway_stall" is the OpenClaw gateway's own memory-pressure
    watchdog force-aborting a stuck model call, an unrelated infra defect;
    anything else is "other" (e.g. the 5 malformed-answer cases already
    fixed by scripts/recover_ctxlimit_malformed_answers.py)."""
    gw = task_dir / "harness-turn-01" / "gateway.log"
    if not gw.is_file():
        return "other"
    text = gw.read_text(errors="ignore")
    if "Context overflow: prompt too large for the model" in text:
        return "context_overflow"
    if "stalled session" in text or "stuck session recovery" in text:
        return "gateway_stall"
    return "other"


def count_compactions(task_dir: Path) -> int:
    """Number of times OpenClaw's own auto-compaction fired mid-task."""
    gw = task_dir / "harness-turn-01" / "gateway.log"
    if not gw.is_file():
        return 0
    return gw.read_text(errors="ignore").count("auto-compaction succeeded")


def is_judge_timeout(task_dir: Path) -> bool:
    """True when evaluation_results.json exists but every rubric is
    UNSCORED (num_scored == 0) -- confirmed via judge_errors.log to be the
    DeepSeek judge API itself timing out ("context deadline exceeded") on
    every call, not a real 0 score. classify_failure()/agg_score_with_failures()
    can't catch this: the file exists and json-decodes fine, it's just
    empty of any actual judging. Treated as excluded, like a gateway-stall,
    rather than a real answer-quality data point."""
    eval_path = task_dir / "evaluation" / "evaluation_results.json"
    if not eval_path.is_file():
        return False
    try:
        data = json.loads(eval_path.read_text())
    except Exception:
        return False
    return data.get("num_scored") == 0


def load_replicate_set(replicates, condition, task_meta):
    """One row per (replicate, arm, task_id): raw agg_score, is_failure,
    failure_cause (only meaningful when is_failure), corrected totalTokens,
    runtime, and assistant-turn count, tagged with the given condition
    label. Efficiency fields are collected for every instance regardless of
    outcome, since cost is incurred whether or not the task ultimately gets
    judged -- unlike agg_score, they are not zeroed/excluded on failure."""
    rows = []
    for rep in replicates:
        dirs = {"control": Path(rep["control"]), "repo-scope": Path(rep["repo"])}
        for arm, run_dir in dirs.items():
            for task_id, _suffix, task_dir in psac.find_task_dirs_ordered(run_dir):
                agg, is_failure = psac.agg_score_with_failures(task_dir)
                if not is_failure and is_judge_timeout(task_dir):
                    agg, is_failure = None, True
                    judge_timed_out = True
                else:
                    judge_timed_out = False
                tokens = ssa.load_session_tokens(task_dir) or {}
                runtime_ms = tokens.get("runtimeMs")
                rows.append({
                    "condition": condition,
                    "replicate": rep["name"],
                    "arm": arm,
                    "task_id": task_id,
                    "agg_score": agg,
                    "is_failure": is_failure,
                    "failure_cause": ("judge_timeout" if judge_timed_out
                                      else classify_failure(task_dir) if is_failure else None),
                    "totalTokens": psac.corrected_total_tokens(task_dir),
                    "totalTokens_raw": tokens.get("totalTokens"),
                    "runtimeSec": (runtime_ms / 1000.0) if runtime_ms else None,
                    "turns": psac.count_assistant_turns(task_dir),
                    "n_compact": count_compactions(task_dir),
                })
    return rows


def build_dataset() -> pd.DataFrame:
    qa_root = REPO_ROOT / ".cache" / "swe-atlas-qa"
    task_meta = ssa.load_task_metadata(qa_root)
    rows = load_replicate_set(FULL_REPLICATES, "full", task_meta)
    rows += load_replicate_set(CTXLIMIT_REPLICATES, "ctxlimit", task_meta)
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=COND_ORDER, ordered=True)
    df["arm"] = pd.Categorical(df["arm"], categories=ARM_ORDER, ordered=True)
    return df


def per_task_means(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Average across replicates within each (condition, arm, task_id) cell
    first (so a replicate-rich cell doesn't outweigh a replicate-poor one),
    then pivot to one row per task_id with 4 score columns."""
    cell = df.groupby(["condition", "arm", "task_id"], observed=True)[score_col].mean().reset_index()
    piv = cell.pivot_table(index="task_id", columns=["arm", "condition"], values=score_col)
    piv.columns = [f"{arm}_{cond}" for arm, cond in piv.columns]
    return piv


def sign_flip_permutation_p(diff: np.ndarray, n_perm: int = 200_000, seed: int = 0) -> float:
    """One-sided permutation p-value for mean(diff) > 0, under the null
    that each task's diff sign is equally likely +/- (the appropriate
    exchangeability null when the "which arm is which" labeling is what's
    being tested for a systematic effect)."""
    rng = np.random.default_rng(seed)
    observed = diff.mean()
    n = len(diff)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = (signs * diff).mean(axis=1)
    return float((perm_means >= observed).mean())


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def analyze_convention(df: pd.DataFrame, score_col: str, label: str) -> dict:
    print(f"\n{'=' * 70}\nConvention: {label}\n{'=' * 70}")

    piv = per_task_means(df, score_col)
    needed = ["control_full", "control_ctxlimit", "repo-scope_full", "repo-scope_ctxlimit"]
    complete = piv.dropna(subset=needed)
    print(f"tasks with data in all 4 cells: {len(complete)} of {len(piv)}")

    for col in needed:
        print(f"  mean {col:<22} = {piv[col].mean():.4f}  (n={piv[col].notna().sum()})")

    deg_control = complete["control_full"] - complete["control_ctxlimit"]
    deg_repo = complete["repo-scope_full"] - complete["repo-scope_ctxlimit"]
    diff = (deg_control - deg_repo).to_numpy()  # >0 supports the hypothesis

    print(f"\nmean degradation, control    (full - ctxlimit) = {deg_control.mean():+.4f}")
    print(f"mean degradation, repo-scope (full - ctxlimit) = {deg_repo.mean():+.4f}")
    print(f"mean DiD (control_degradation - repo_degradation) = {diff.mean():+.4f}  (n={len(diff)} tasks)")

    if len(diff) >= 2 and np.any(diff != 0):
        w_stat, w_p_two = stats.wilcoxon(diff)
        w_p_one = w_p_two / 2 if diff.mean() > 0 else 1 - w_p_two / 2
        t_stat, t_p_two = stats.ttest_1samp(diff, 0)
        t_p_one = t_p_two / 2 if diff.mean() > 0 else 1 - t_p_two / 2
        perm_p = sign_flip_permutation_p(diff)
        ci_lo, ci_hi = bootstrap_ci(diff)
        print(f"Wilcoxon signed-rank: stat={w_stat:.3f}  one-sided p={w_p_one:.4f} (two-sided {w_p_two:.4f})")
        print(f"Paired t-test:        stat={t_stat:.3f}  one-sided p={t_p_one:.4f} (two-sided {t_p_two:.4f})")
        print(f"Sign-flip permutation one-sided p={perm_p:.4f}")
        print(f"Bootstrap 95% CI on mean DiD: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    else:
        w_p_one = t_p_one = perm_p = ci_lo = ci_hi = float("nan")
        print("Not enough variation in per-task diffs to run inferential tests.")

    return {
        "label": label, "n": len(diff),
        "deg_control": deg_control.mean(), "deg_repo": deg_repo.mean(), "mean_diff": diff.mean(),
        "wilcoxon_p_one_sided": w_p_one, "ttest_p_one_sided": t_p_one, "perm_p_one_sided": perm_p,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "complete": complete, "diff": diff,
    }


ARM_DISPLAY = {"control": "baseline", "repo-scope": "repo-scope"}


def plot_2x2(df: pd.DataFrame, out_dir: Path):
    """Mean agg_score by arm x condition. A context-overflow give-up counts
    as 0 -- it's the ctxlimit condition doing exactly what it's designed to
    do -- while a gateway-stall abort or other unclassified infra defect is
    excluded rather than scored, since that's a reliability fault unrelated
    to context limiting itself."""
    plot_df = df.dropna(subset=["agg_score_excl"]).assign(
        arm_display=lambda d: d["arm"].map(ARM_DISPLAY)
    )
    hue_order = [ARM_DISPLAY[a] for a in ARM_ORDER]
    palette = {ARM_DISPLAY[a]: ARM_PALETTE[a] for a in ARM_ORDER}

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.barplot(
        data=plot_df, x="condition", y="agg_score_excl", hue="arm_display", order=COND_ORDER, hue_order=hue_order,
        palette=palette, errorbar="sd", capsize=0.12, ax=ax,
    )
    ax.legend(title="", loc="lower right")
    stats = plot_df.groupby(["condition", "arm"], observed=True)["agg_score_excl"].agg(["mean", "std"])
    for i, cond in enumerate(COND_ORDER):
        for j, arm in enumerate(ARM_ORDER):
            if (cond, arm) in stats.index:
                mean, std = stats.loc[(cond, arm), "mean"], stats.loc[(cond, arm), "std"]
                top = mean + (std if pd.notna(std) else 0)
                x = i + (j - 0.5) * 0.4
                ax.annotate(f"{mean:.3f}", (x, top), ha="center", va="bottom",
                            fontsize=11, fontweight="bold", xytext=(0, 6), textcoords="offset points")
    ax.set_xticklabels(["full (262k)", "ctxlimit (96k)"])
    ax.set_xlabel("")
    ax.set_ylabel("agg_score")
    ax.set_ylim(0, 1.15)
    fig.suptitle("Accuracy by arm x context-limit condition", y=0.99)
    fig.tight_layout()
    fig.savefig(out_dir / "degradation_2x2_by_arm.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'degradation_2x2_by_arm.png'}")


def plot_efficiency_2x2(df: pd.DataFrame, out_dir: Path):
    """Cost side of the same 2x2: total tokens, runtime, and assistant-turn
    count by arm x condition. Unlike agg_score, cost is incurred whether or
    not a task ultimately gets judged, so every instance with a reading is
    included here regardless of is_failure/failure_cause."""
    plot_df = df.assign(arm_display=lambda d: d["arm"].map(ARM_DISPLAY))
    hue_order = [ARM_DISPLAY[a] for a in ARM_ORDER]
    palette = {ARM_DISPLAY[a]: ARM_PALETTE[a] for a in ARM_ORDER}

    metrics = [("totalTokens", "total tokens"), ("runtimeSec", "runtime (s)"), ("turns", "assistant turns")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    for ax, (col, label) in zip(axes, metrics):
        sub = plot_df.dropna(subset=[col])
        sns.barplot(
            data=sub, x="condition", y=col, hue="arm_display", order=COND_ORDER, hue_order=hue_order,
            palette=palette, errorbar="sd", capsize=0.12, ax=ax, legend=False,
        )
        stats = sub.groupby(["condition", "arm"], observed=True)[col].agg(["mean", "std"])
        for i, cond in enumerate(COND_ORDER):
            for j, arm in enumerate(ARM_ORDER):
                if (cond, arm) in stats.index:
                    mean, std = stats.loc[(cond, arm), "mean"], stats.loc[(cond, arm), "std"]
                    top = mean + (std if pd.notna(std) else 0)
                    x = i + (j - 0.5) * 0.4
                    ax.annotate(f"{mean:,.0f}", (x, top), ha="center", va="bottom",
                                fontsize=9.5, fontweight="bold", xytext=(0, 5), textcoords="offset points")
        ax.set_xticklabels(["full (262k)", "ctxlimit (96k)"])
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.margins(y=0.18)
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[h]) for h in hue_order]
    fig.legend(handles, hue_order, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.suptitle("Efficiency by arm x context-limit condition", y=1.12)
    fig.tight_layout()
    fig.savefig(out_dir / "degradation_efficiency_2x2_by_arm.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'degradation_efficiency_2x2_by_arm.png'}")


def analyze_compaction_resilience(df: pd.DataFrame, out_dir: Path) -> dict:
    """Does AMEM's persisted-outside-the-prompt store cushion the specific
    loss mechanism auto-compaction causes? A context-overflow give-up is the
    most extreme case of compaction failing -- the harness's own retry loop
    (up to 3 attempts) exhausts itself trying to compact the prompt back
    under budget and never succeeds -- so it belongs in the "compacted"
    bucket, scored 0, rather than being dropped: excluding it would hide
    exactly the tail-risk a compaction-resistance claim needs to cover. A
    gateway-stall abort or judge-timeout still has no meaningful "compacted"
    story (an unrelated infra/measurement defect, not a compaction outcome)
    and stays excluded. This compares each arm's own mean agg_score with vs.
    without at least one mid-task compaction. If persisted notes are
    compaction-resistant, the within-arm compacted-vs-not gap should be
    smaller for repo-scope than for baseline -- tested here as an unpaired
    two-sample comparison per arm (compaction status isn't something you can
    pair the same task on, since whether a task compacts is itself an
    outcome) plus a permutation test on the interaction (baseline's gap
    minus repo-scope's gap)."""
    sub = df[
        (df["condition"] == "ctxlimit")
        & (~df["is_failure"] | (df["failure_cause"] == "context_overflow"))
    ].copy()
    sub["compacted"] = (sub["n_compact"] > 0) | (sub["failure_cause"] == "context_overflow")

    cell = sub.groupby(["arm", "compacted"], observed=True)["agg_score"].agg(["mean", "std", "count"])
    print(f"\n{'=' * 70}\nCompaction resilience (ctxlimit, judged + context-overflow scored 0)\n{'=' * 70}")
    print(cell)

    def arr(arm, compacted):
        return sub.loc[(sub["arm"] == arm) & (sub["compacted"] == compacted), "agg_score"].to_numpy()

    cc, cn = arr("control", True), arr("control", False)
    rc, rn = arr("repo-scope", True), arr("repo-scope", False)

    t_c, p_c = stats.ttest_ind(cn, cc, equal_var=False)
    t_r, p_r = stats.ttest_ind(rn, rc, equal_var=False)
    print(f"\nbaseline:   not-compacted vs compacted  t={t_c:.3f} p={p_c:.4f}")
    print(f"repo-scope: not-compacted vs compacted  t={t_r:.3f} p={p_r:.4f}")

    observed = (cn.mean() - cc.mean()) - (rn.mean() - rc.mean())
    rng = np.random.default_rng(0)
    n_perm = 20_000
    control_pool, repo_pool = np.concatenate([cc, cn]), np.concatenate([rc, rn])
    n_cc, n_rc = len(cc), len(rc)
    count = 0
    for _ in range(n_perm):
        perm_c = rng.permutation(control_pool)
        perm_r = rng.permutation(repo_pool)
        g = (perm_c[n_cc:].mean() - perm_c[:n_cc].mean()) - (perm_r[n_rc:].mean() - perm_r[:n_rc].mean())
        if g >= observed:
            count += 1
    perm_p = count / n_perm
    print(f"\ninteraction (baseline's compaction gap - repo-scope's) = {observed:+.4f}")
    print(f"permutation p (one-sided, interaction >= observed) = {perm_p:.4f}")

    sub.to_csv(out_dir / "compaction_resilience_instances.csv", index=False)
    print(f"wrote {out_dir / 'compaction_resilience_instances.csv'}")

    return dict(cell=cell, t_c=t_c, p_c=p_c, t_r=t_r, p_r=p_r, observed=observed, perm_p=perm_p)


def plot_compaction_resilience(result: dict, out_dir: Path):
    cell = result["cell"]
    comp_order = [False, True]
    comp_labels = {False: "not compacted", True: "compacted"}
    comp_palette = {False: "#83817a", True: "#d03b3b"}

    fig, ax = plt.subplots(figsize=(8, 6.5))
    x = np.arange(len(ARM_ORDER))
    width = 0.35
    for i, compacted in enumerate(comp_order):
        means, stds = [], []
        for arm in ARM_ORDER:
            row = cell.loc[(arm, compacted)] if (arm, compacted) in cell.index else None
            means.append(row["mean"] if row is not None else 0)
            stds.append(row["std"] if row is not None else 0)
        xpos = x + (i - 0.5) * width
        ax.bar(xpos, means, width, yerr=stds, capsize=6, color=comp_palette[compacted],
               label=comp_labels[compacted], error_kw={"linewidth": 1.3})
        for xi, m, s in zip(xpos, means, stds):
            ax.annotate(f"{m:.3f}", (xi, m + s), ha="center", va="bottom",
                        fontsize=11, fontweight="bold", xytext=(0, 5), textcoords="offset points")

    ax.set_xticks(x)
    ax.set_xticklabels([ARM_DISPLAY[a] for a in ARM_ORDER])
    ax.set_ylabel("agg_score")
    ax.set_ylim(0, 1.15)
    ax.legend(title="", loc="lower center", ncol=2, frameon=True)
    subtitle = (f"baseline gap={result['cell'].loc[('control', False), 'mean'] - result['cell'].loc[('control', True), 'mean']:+.3f} "
                f"(p={result['p_c']:.3f})  |  repo-scope gap="
                f"{result['cell'].loc[('repo-scope', False), 'mean'] - result['cell'].loc[('repo-scope', True), 'mean']:+.3f} "
                f"(p={result['p_r']:.3f})  |  interaction p={result['perm_p']:.3f}")
    fig.suptitle("Does compaction hurt repo-scope less than baseline?", y=1.02)
    ax.set_title(subtitle, fontsize=10.5, pad=12)
    fig.tight_layout()
    fig.savefig(out_dir / "compaction_resilience_by_arm.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'compaction_resilience_by_arm.png'}")


def plot_diff_distribution(diff: np.ndarray, out_dir: Path, label: str, suffix: str):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    x = np.random.default_rng(0).uniform(-0.15, 0.15, size=len(diff))
    colors = ["#1baf7a" if d > 0 else ("#d03b3b" if d < 0 else "#83817a") for d in diff]
    ax.scatter(x, diff, color=colors, s=70, alpha=0.85, zorder=3)
    mean_diff = diff.mean()
    ax.plot([-0.25, 0.25], [mean_diff, mean_diff], color="black", linewidth=2.5, linestyle="--",
            label=f"mean = {mean_diff:+.3f}")
    n_supports = int((diff > 0).sum())
    ax.set_xticks([])
    ax.set_ylabel("per-task DiD: control degradation - repo-scope degradation\n(> 0 supports the hypothesis)")
    ax.set_title(f"{label}  |  n={len(diff)}  |  {n_supports}/{len(diff)} tasks support the hypothesis (DiD > 0)",
                 fontsize=11)
    fig.suptitle("Distribution of per-task difference-in-differences", y=1.0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = out_dir / f"degradation_diff_distribution_{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


def print_verdict(result: dict):
    print(f"\n{'=' * 70}\nVERDICT\n{'=' * 70}")
    supported = result["mean_diff"] > 0 and min(result["wilcoxon_p_one_sided"], result["perm_p_one_sided"]) < 0.05
    status = "SUPPORTED" if supported else ("INCONCLUSIVE" if result["mean_diff"] > 0 else "NOT SUPPORTED")
    print(f"control degrades {result['deg_control']:+.3f}, repo-scope degrades {result['deg_repo']:+.3f} "
          f"-> DiD={result['mean_diff']:+.3f} (n={result['n']}), Wilcoxon p={result['wilcoxon_p_one_sided']:.3f}, "
          f"permutation p={result['perm_p_one_sided']:.3f}  =>  {status}")
    print("Caveats: n<=30 tasks, only 2-5 replicates per cell; a context-overflow give-up scores 0 (it's the "
          "ctxlimit condition working as intended), but gateway-stall aborts and other unclassified infra "
          "defects are still excluded rather than scored -- treat the p-value as indicative, not confirmatory.")


def main():
    out_dir = REPO_ROOT / "scripts" / "out_ctxlimit"
    out_dir.mkdir(exist_ok=True)

    df = build_dataset()
    # agg_score_with_failures() already fills 0.0 for judge failures (reward.txt
    # present, no evaluation_results.json). Two different things hide behind
    # that: "context_overflow" is the ctxlimit condition doing exactly what
    # it's designed to do -- the agent ran out of budget and gave up -- so
    # it's a real, in-scope consequence of limiting context and is scored 0
    # rather than dropped. "gateway_stall" (the OpenClaw gateway's own
    # memory-pressure watchdog force-aborting a stuck model call) and
    # anything unclassified are unrelated infra defects, not something a
    # context-limit-vs-not comparison should be measuring, so those stay
    # excluded.
    df["agg_score_excl"] = df["agg_score"].where(
        ~df["is_failure"] | (df["failure_cause"] == "context_overflow"), np.nan
    )

    join_path = out_dir / "degradation_join_task_level.csv"
    df.to_csv(join_path, index=False)
    print(f"wrote {join_path} ({len(df)} rows)")

    print("\n--- failure-cause breakdown by (condition, arm) ---")
    print(df.loc[df["is_failure"]].groupby(["condition", "arm", "failure_cause"], observed=True).size())
    print("\n--- excluded-from-analysis rate (gateway_stall + judge_timeout + other) by (condition, arm) ---")
    excluded = df["is_failure"] & (df["failure_cause"] != "context_overflow")
    print(excluded.groupby([df["condition"], df["arm"]], observed=True).agg(["mean", "sum", "count"]))

    result = analyze_convention(df, "agg_score_excl", "context-overflow scored 0, other infra failures excluded")
    result["complete"].to_csv(out_dir / "degradation_diff_in_diff.csv")
    print(f"wrote {out_dir / 'degradation_diff_in_diff.csv'}")

    print_verdict(result)

    sns.set_theme(style="whitegrid", context="talk")
    plot_2x2(df, out_dir)
    plot_diff_distribution(result["diff"], out_dir, "context-overflow scored 0, other infra failures excluded", "excl")
    plot_efficiency_2x2(df, out_dir)

    # --- efficiency: cost side of the same 2x2, all instances (failures included -- cost is
    # incurred regardless of whether the task ends up judged) ---
    print(f"\n{'=' * 70}\nEFFICIENCY (all instances, failures included -- cost is paid either way)\n{'=' * 70}")
    for col in ["totalTokens", "runtimeSec", "turns"]:
        sub = df.dropna(subset=[col])
        if sub.empty:
            continue
        print(f"\n--- {col} by (condition, arm) ---")
        print(sub.groupby(["condition", "arm"], observed=True)[col].agg(["mean", "std", "count"]))

    compaction_result = analyze_compaction_resilience(df, out_dir)
    plot_compaction_resilience(compaction_result, out_dir)


if __name__ == "__main__":
    main()
