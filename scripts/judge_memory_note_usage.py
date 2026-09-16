#!/usr/bin/env python3
"""LLM-judge pass measuring whether a retrieved memory note was actually
*used* by the agent, as opposed to merely topically relevant to the task
question (which judge_memory_relevance.py already measures).

For each memory_search call, this reads the raw telemetry for that task and
reconstructs what the agent did *after* retrieving each note: the shell
commands it ran, its own reasoning text, and its final answer -- up to the
next memory_search call in the same session (or end of session). It then
asks an LLM judge whether the note's specific content shows up (echoed,
acted upon, or contradicted-and-then-corrected) in that subsequent
trajectory, which is a much stronger signal of usefulness than "could this
plausibly have helped."

Usage:
  scripts/judge_memory_note_usage.py \\
      --notes scripts/out/memory_search_notes.csv \\
      --events scripts/out/memory_search_events_with_overlap.csv \\
      --runs-root runs \\
      --out scripts/out/memory_note_usage_judged.csv \\
      [--relevance-csv scripts/out/memory_search_relevance_judged.csv]
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from extract_memory_retrieval_events import find_telemetry_session_files  # noqa: E402

DEFAULT_BASE_URL = "http://192.168.111.200:8100/v1"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

MAX_EXEC_COMMANDS = 25
MAX_REASONING_SNIPPETS = 15
MAX_EXEC_CHARS = 200
MAX_REASONING_CHARS = 300
MAX_FINAL_ANSWER_CHARS = 2000

JUDGE_SYSTEM_PROMPT = (
    "You are grading whether a coding agent actually USED a specific memory "
    "note it retrieved, not merely whether the note was topically related to "
    "the question. You will see: the question the agent was asked, the note "
    "content, a trailing summary of what the agent did AFTER retrieving this "
    "note (shell commands it ran, its own reasoning snippets, in order), and "
    "the agent's final answer. Judge whether the note's specific facts, "
    "identifiers, file paths, or claims appear to have driven what the agent "
    "did next or what it wrote in its final answer -- reflected verbatim, "
    "paraphrased, acted on (e.g. it ran a command targeting a file/function "
    "the note named), or explicitly contradicted and then corrected. A note "
    "that was retrieved but never shows up in the subsequent actions or "
    "answer, even if it looks relevant, was NOT used. Respond with strict "
    "JSON only: {\"used\": true|false, \"confidence\": \"high\"|\"medium\"|\"low\", "
    "\"evidence\": \"<one sentence pointing to what shows usage, or why not>\"}"
)


def call_judge(base_url: str, model: str, question: str, note_content: str,
                trajectory: str, final_answer: str, timeout: int = 90) -> dict:
    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED NOTE:\n{note_content}\n\n"
        f"AGENT'S SUBSEQUENT ACTIONS:\n{trajectory or '(none captured)'}\n\n"
        f"AGENT'S FINAL ANSWER:\n{final_answer or '(none captured)'}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 200,
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
        return {"used": None, "confidence": None, "evidence": f"unparseable judge output: {text!r}"}
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"used": None, "confidence": None, "evidence": f"unparseable judge JSON: {text!r}"}
    return {
        "used": parsed.get("used"),
        "confidence": parsed.get("confidence"),
        "evidence": parsed.get("evidence"),
    }


def load_session_records(session_path: Path) -> list[dict | None]:
    records = []
    with session_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                records.append(None)
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append(None)
    return records


def extract_call_contexts(task_dir: Path) -> dict[int, dict]:
    """Returns call_seq -> {exec_commands, reasoning_snippets, final_answer},
    mirroring extract_memory_retrieval_events.iter_memory_search_calls's
    per-session call_seq numbering (resets to 1 at the start of each session
    file) so keys line up with memory_search_notes.csv's call_seq column."""
    contexts: dict[int, dict] = {}
    for session_path in find_telemetry_session_files(task_dir):
        records = load_session_records(session_path)

        result_indices = []
        for i, rec in enumerate(records):
            if rec is None:
                continue
            msg = rec.get("message", {})
            if msg.get("role") == "toolResult" and msg.get("toolName") == "memory_search":
                result_indices.append(i)
        if not result_indices:
            continue

        final_answer = None
        for rec in reversed(records):
            if rec is None:
                continue
            msg = rec.get("message", {})
            if msg.get("role") != "assistant":
                continue
            texts = [
                item.get("text") for item in (msg.get("content") or [])
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
            ]
            if texts:
                final_answer = texts[-1]
                break

        for call_seq, start_idx in enumerate(result_indices, start=1):
            end_idx = result_indices[call_seq] if call_seq < len(result_indices) else len(records)
            exec_commands, reasoning_snippets = [], []
            for i in range(start_idx + 1, end_idx):
                rec = records[i]
                if rec is None:
                    continue
                msg = rec.get("message", {})
                if msg.get("role") != "assistant":
                    continue
                for item in (msg.get("content") or []):
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text" and item.get("text"):
                        reasoning_snippets.append(item["text"])
                    elif item.get("type") == "toolCall" and item.get("name") == "exec":
                        cmd = (item.get("arguments") or {}).get("command")
                        if cmd:
                            exec_commands.append(cmd)
            contexts[call_seq] = {
                "exec_commands": exec_commands,
                "reasoning_snippets": reasoning_snippets,
                "final_answer": final_answer,
            }
    return contexts


def render_trajectory(ctx: dict) -> str:
    parts = []
    commands = ctx["exec_commands"][:MAX_EXEC_COMMANDS]
    if commands:
        parts.append("Shell commands run after this note was retrieved:")
        parts.extend(f"  $ {c[:MAX_EXEC_CHARS]}" for c in commands)
    snippets = ctx["reasoning_snippets"][:MAX_REASONING_SNIPPETS]
    if snippets:
        parts.append("Agent reasoning excerpts after this note was retrieved:")
        parts.extend(f"  - {s[:MAX_REASONING_CHARS]}" for s in snippets)
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notes", type=Path, default=Path("scripts/out/memory_search_notes.csv"))
    parser.add_argument("--events", type=Path, default=Path("scripts/out/memory_search_events_with_overlap.csv"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("scripts/out/memory_note_usage_judged.csv"))
    parser.add_argument("--relevance-csv", type=Path, default=None,
                         help="optional memory_search_relevance_judged.csv to cross-tab used vs relevant")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="only judge the first N notes (debugging)")
    args = parser.parse_args()

    notes = pd.read_csv(args.notes)
    events = pd.read_csv(args.events)[["run_dir", "task_id", "occurrence", "call_seq", "question", "bucket", "delta"]]
    notes = notes.merge(events, on=["run_dir", "task_id", "occurrence", "call_seq"], how="left")
    if args.limit:
        notes = notes.head(args.limit)

    call_key_cols = ["run_dir", "task_id", "occurrence", "call_seq"]
    context_cache: dict[tuple, dict] = {}

    results = []
    for i, row in notes.iterrows():
        call_key = tuple(row[c] for c in call_key_cols)
        if call_key not in context_cache:
            task_dir = args.runs_root / row["run_dir"] / f"{row['task_id']}-{int(row['occurrence']):03d}"
            all_contexts = extract_call_contexts(task_dir) if task_dir.is_dir() else {}
            context_cache[call_key] = all_contexts

        ctx = context_cache[call_key].get(int(row["call_seq"]), {
            "exec_commands": [], "reasoning_snippets": [], "final_answer": None,
        })
        trajectory = render_trajectory(ctx)
        final_answer = (ctx["final_answer"] or "")[:MAX_FINAL_ANSWER_CHARS]

        question = row.get("question")
        content = row.get("note_text_snippet")
        if not isinstance(question, str) or not isinstance(content, str):
            verdict = {"used": None, "confidence": None, "evidence": "missing question or note text"}
        else:
            try:
                verdict = call_judge(args.base_url, args.model, question, content, trajectory, final_answer)
            except Exception as e:  # noqa: BLE001 -- best-effort batch judging, log and continue
                verdict = {"used": None, "confidence": None, "evidence": f"judge call failed: {e}"}

        print(
            f"[{i + 1}/{len(notes)}] task={row['task_id']} call={row['call_seq']} note_rank={row['note_rank']} "
            f"-> used={verdict['used']} ({verdict['confidence']})",
            file=sys.stderr,
        )
        results.append({
            **row.to_dict(),
            "llm_used": verdict["used"],
            "llm_used_confidence": verdict["confidence"],
            "llm_used_evidence": verdict["evidence"],
        })

    out_df = pd.DataFrame(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"wrote {len(out_df)} rows to {args.out}", file=sys.stderr)

    print("\n=== Per-bucket used rate (LLM judge, note-level) ===")
    for bucket, g in out_df.groupby("bucket", dropna=False):
        rated = g["llm_used"].dropna()
        rate = rated.mean() if len(rated) else None
        print(f"{bucket}: n={len(g)} rated={len(rated)} used_rate={rate}")

    if args.relevance_csv and args.relevance_csv.is_file():
        rel = pd.read_csv(args.relevance_csv)[
            ["run_dir", "task_id", "occurrence", "call_seq", "note_rank", "llm_relevant"]
        ]
        joined = out_df.merge(rel, on=["run_dir", "task_id", "occurrence", "call_seq", "note_rank"], how="inner")
        print(f"\n=== Relevant (topical) vs Used (behavioral): n={len(joined)} jointly-judged notes ===")
        ct = pd.crosstab(joined["llm_relevant"], joined["llm_used"], dropna=False)
        print(ct)
        relevant_not_used = joined[(joined["llm_relevant"] == True) & (joined["llm_used"] == False)]  # noqa: E712
        print(f"\nrelevant-but-not-used: {len(relevant_not_used)} / {len(joined)} "
              f"({len(relevant_not_used) / len(joined):.1%})" if len(joined) else "")


if __name__ == "__main__":
    main()
