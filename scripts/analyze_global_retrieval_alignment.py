#!/usr/bin/env python3
"""Is the agent's memory_search query -- and what it retrieves -- actually
aligned with the task it's working on, under global amem scope?

extract_memory_retrieval_events.py's structured details.memories field is
silently truncated by the telemetry writer once a call's payload exceeds
~8KB (persistedDetailsTruncated=true), which happens on most global-scope
calls (the shared store is bigger than repo/task-scope's), leaving
memory_search_notes.csv nearly empty for this arm. The free-text
toolResult.content[0].text is NOT truncated and contains the same
"N. <content> (similarity NN%, id: <uuid>)" listing, so this script
re-parses that instead of relying on the structured field.

For every memory_search call under global scope, this computes:
  - query <-> task-question lexical overlap (does the agent's own
    self-generated query even resemble what it was actually asked?)
  - each retrieved note's origin repository (via keyword match against the
    note's own content -- global scope's store spans both repositories, so
    this is the only way to tell a same-repo hit from a cross-repo one)
  - note <-> task-question lexical overlap, split by same-repo vs cross-repo
    origin, since a cross-repo note is a much stronger prior for "irrelevant"
    than topic overlap alone

Usage:
  scripts/analyze_global_retrieval_alignment.py \\
      --run runs/<amem-global-run> [--run ... x3] \\
      --qa-root .cache/swe-atlas-qa \\
      --out scripts/out/global_retrieval/query_alignment.csv
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import summarize_sweatlas_arms as ssa  # noqa: E402
from extract_memory_retrieval_events import find_telemetry_session_files, infer_replicate  # noqa: E402
from analyze_query_relevance_gap import extract_question, tokenize, jaccard  # noqa: E402

import pandas as pd

NOTE_RE = re.compile(
    r"(?:^|\n)(\d+)\.\s(.*?)\s\(similarity (\d+)%, id: ([0-9a-f-]+)\)",
    re.DOTALL,
)

# Keyword sets used only to tag which repo a note's content is ABOUT --
# distinct from (and independent of) which task/repo it was retrieved for.
REPO_KEYWORDS = {
    "simple-login/app": [
        "simplelogin", "simple login", "email_handler", "job_runner.py",
        "reverse-alias", "reply_email", "mailbox", "alias",
    ],
    "paperless-ngx/paperless-ngx": [
        "paperless-ngx", "paperless ngx", "django-q", "django_q", "consume_file",
        "consumption_dir", "documents.tasks", "whoosh", "document_consumer",
    ],
}


def tag_origin(note_content: str) -> str | None:
    text = note_content.lower()
    hits = [repo for repo, kws in REPO_KEYWORDS.items() if any(k in text for k in kws)]
    return hits[0] if len(hits) == 1 else None


def parse_notes_from_text(text: str) -> list[dict]:
    if not text:
        return []
    return [
        {"rank": int(m.group(1)), "content": m.group(2).strip(), "similarity": int(m.group(3)), "note_id": m.group(4)}
        for m in NOTE_RE.finditer(text)
    ]


def iter_calls_with_full_text(task_dir: Path):
    """Like extract_memory_retrieval_events.iter_memory_search_calls, but
    keeps the free-text toolResult (never truncated) instead of the
    structured details field (often truncated for global scope)."""
    for session_path in find_telemetry_session_files(task_dir):
        tool_call_queries, call_order, tool_texts = {}, [], {}
        with session_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = record.get("message", {})
                role = message.get("role")
                if role == "assistant":
                    for item in message.get("content") or []:
                        if isinstance(item, dict) and item.get("type") == "toolCall" and item.get("name") == "memory_search":
                            tool_call_id = item["id"]
                            call_order.append(tool_call_id)
                            tool_call_queries[tool_call_id] = (item.get("arguments") or {}).get("query")
                elif role == "toolResult" and message.get("toolName") == "memory_search":
                    tool_call_id = message.get("toolCallId")
                    content = message.get("content") or []
                    tool_texts[tool_call_id] = content[0]["text"] if content else None

        for call_seq, tool_call_id in enumerate(call_order, start=1):
            yield {
                "session_id": session_path.stem,
                "call_seq": call_seq,
                "query": tool_call_queries.get(tool_call_id),
                "notes": parse_notes_from_text(tool_texts.get(tool_call_id)),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, type=Path, dest="runs")
    parser.add_argument("--qa-root", default=Path(".cache/swe-atlas-qa"), type=Path)
    parser.add_argument("--out", default=Path("scripts/out/global_retrieval/query_alignment.csv"), type=Path)
    args = parser.parse_args()

    task_meta = ssa.load_task_metadata(args.qa_root)

    call_rows, note_rows = [], []
    for run_dir in args.runs:
        replicate = infer_replicate(run_dir.name)
        for task_id, task_dir in ssa.find_task_dirs(run_dir):
            repository, _ = task_meta.get(task_id, (None, None))
            question = extract_question(task_dir)
            q_tokens = tokenize(question)
            for call in iter_calls_with_full_text(task_dir):
                query_tokens = tokenize(call["query"])
                q_overlap = jaccard(q_tokens, query_tokens)
                notes = call["notes"]
                origins = [tag_origin(n["content"]) for n in notes]
                cross = [o for o in origins if o and repository and o != repository]
                same = [o for o in origins if o and repository and o == repository]
                call_rows.append({
                    "replicate": replicate, "task_id": task_id, "repository": repository,
                    "call_seq": call["call_seq"], "query": call["query"],
                    "n_notes_parsed": len(notes),
                    "query_question_overlap": q_overlap,
                    "n_cross_repo": len(cross), "n_same_repo": len(same),
                    "n_origin_unknown": len(notes) - len(cross) - len(same),
                    "cross_repo_frac": (len(cross) / len(notes)) if notes else None,
                })
                for n in notes:
                    origin = tag_origin(n["content"])
                    note_rows.append({
                        "replicate": replicate, "task_id": task_id, "repository": repository,
                        "call_seq": call["call_seq"], "rank": n["rank"], "similarity": n["similarity"],
                        "note_id": n["note_id"], "origin_repo": origin,
                        "cross_repo": bool(origin and repository and origin != repository),
                        "note_question_overlap": jaccard(q_tokens, tokenize(n["content"])),
                        "content_snippet": n["content"][:160],
                    })

    calls_df = pd.DataFrame(call_rows)
    notes_df = pd.DataFrame(note_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    calls_df.to_csv(args.out, index=False)
    notes_df.to_csv(args.out.parent / "query_alignment_notes.csv", index=False)
    print(f"wrote {len(calls_df)} calls -> {args.out}")
    print(f"wrote {len(notes_df)} notes -> {args.out.parent / 'query_alignment_notes.csv'}")

    print("\n=== Query <-> actual task question, lexical overlap (jaccard) ===")
    print(calls_df["query_question_overlap"].describe())
    low = (calls_df["query_question_overlap"].fillna(0) < 0.05).mean()
    print(f"fraction of queries sharing <5% of question vocabulary: {low:.0%}  (n={len(calls_df)})")

    print("\n=== Retrieved notes: same-repo vs cross-repo (global scope only) ===")
    print(notes_df["cross_repo"].value_counts(dropna=False))
    print("\nmean note<->question overlap, by origin:")
    print(notes_df.groupby("cross_repo")["note_question_overlap"].agg(["mean", "count"]))
    print("\nmean similarity score, by origin (amem's own score -- does it know the difference?):")
    print(notes_df.groupby("cross_repo")["similarity"].agg(["mean", "count"]))

    print("\n=== Worst-aligned queries (lowest query<->question overlap, with notes returned) ===")
    worst = calls_df[calls_df["n_notes_parsed"] > 0].sort_values("query_question_overlap").head(8)
    print(worst[["replicate", "task_id", "call_seq", "query", "query_question_overlap", "cross_repo_frac"]].to_string(index=False))


if __name__ == "__main__":
    main()
