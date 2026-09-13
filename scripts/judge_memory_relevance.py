#!/usr/bin/env python3
"""LLM-judge pass over memory_search_notes.csv: for every retrieved note,
ask an LLM whether the note's content is actually relevant/useful for
answering the task's real <question> (as opposed to just being amem's own
vector-similarity match). This replaces the crude lexical-overlap proxy in
analyze_query_relevance_gap.py with an actual judgment call, at a scale
(dozens of notes) that would be tedious but not impossible to do by hand --
scripts/out/memory_search_relevance_hand_review.csv holds a by-hand review
of the same repo-scope cases for cross-validation.

Judge model: whatever OpenAI-compatible chat endpoint is passed via
--base-url/--model (defaults to the local SGLang endpoint already used by
amem itself in these runs, since no DEEPSEEK_API_KEY is available in this
environment -- note this means the judge and the note-writer may share the
same underlying model, a caveat worth keeping in mind when reading verdicts).

Usage:
  scripts/judge_memory_relevance.py \\
      --events scripts/out/memory_search_events_with_overlap.csv \\
      --notes scripts/out/memory_search_notes.csv \\
      --out scripts/out/memory_search_relevance_judged.csv
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DEFAULT_BASE_URL = "http://192.168.111.200:8100/v1"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

JUDGE_SYSTEM_PROMPT = (
    "You are grading whether a retrieved memory note helped answer a specific "
    "question about a codebase. You will be shown the question an engineer "
    "asked, and one note that a memory-search tool returned for it. Judge "
    "ONLY whether this note's content would actually help answer THIS "
    "specific question -- not whether it's about the same repository or "
    "vaguely related area. A note about the right subsystem but the wrong "
    "specific concern (e.g. covers request routing when the question is "
    "about a memory leak) is NOT relevant. Respond with strict JSON only: "
    '{"relevant": true|false, "reason": "<one sentence>"}'
)


def call_judge(base_url: str, model: str, question: str, note_content: str, timeout: int = 60) -> dict:
    user_prompt = f"QUESTION:\n{question}\n\nRETRIEVED NOTE:\n{note_content}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 150,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    text = body["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"relevant": None, "reason": f"unparseable judge output: {text!r}"}
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"relevant": None, "reason": f"unparseable judge JSON: {text!r}"}
    return {"relevant": parsed.get("relevant"), "reason": parsed.get("reason")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=Path, default=Path("scripts/out/memory_search_events_with_overlap.csv"))
    parser.add_argument("--notes", type=Path, default=Path("scripts/out/memory_search_notes.csv"))
    parser.add_argument("--out", type=Path, default=Path("scripts/out/memory_search_relevance_judged.csv"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    events = pd.read_csv(args.events)[
        ["run_dir", "task_id", "occurrence", "call_seq", "question", "bucket", "delta", "sim_mean"]
    ]
    notes = pd.read_csv(args.notes)
    notes = notes.merge(events, on=["run_dir", "task_id", "occurrence", "call_seq"], how="left")

    results = []
    for i, row in notes.iterrows():
        question = row.get("question")
        content = row.get("note_text_snippet")
        if not isinstance(question, str) or not isinstance(content, str):
            verdict = {"relevant": None, "reason": "missing question or note text"}
        else:
            try:
                verdict = call_judge(args.base_url, args.model, question, content)
            except Exception as e:  # noqa: BLE001 -- best-effort batch judging, log and continue
                verdict = {"relevant": None, "reason": f"judge call failed: {e}"}
        print(
            f"[{i + 1}/{len(notes)}] task={row['task_id']} call={row['call_seq']} note_rank={row['note_rank']} "
            f"-> relevant={verdict['relevant']}",
            file=sys.stderr,
        )
        results.append({**row.to_dict(), "llm_relevant": verdict["relevant"], "llm_reason": verdict["reason"]})

    out_df = pd.DataFrame(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"wrote {len(out_df)} rows to {args.out}", file=sys.stderr)

    print("\n=== Per-bucket relevant rate (LLM judge, note-level) ===")
    for bucket, g in out_df.groupby("bucket", dropna=False):
        rated = g["llm_relevant"].dropna()
        rate = rated.mean() if len(rated) else None
        print(f"{bucket}: n={len(g)} rated={len(rated)} relevant_rate={rate}")


if __name__ == "__main__":
    main()
