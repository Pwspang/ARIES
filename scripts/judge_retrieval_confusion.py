#!/usr/bin/env python3
"""LLM-judge confusion matrix for retrieved memory, control vs both.

Extracts every memory_search result note directly from the raw session
transcript's content[0].text ("Found N matching memories, plus M linked to
them: 1. <content> (similarity XX%, id: <uuid>) 2. ..."), NOT from the
toolResult's structured `details.memories` field -- that field gets replaced
with a `{persistedDetailsTruncated: true}` stub by OpenClaw's own session-log
persistence once a result set gets large, which silently drops the note
content extract_memory_retrieval_events.py relies on. content[0].text is
never truncated this way, so this is the reliable path for this study's data
(confirmed: most calls in this study's runs have truncated `details`).

For each extracted note, asks the same LLM-judge question as
judge_memory_relevance.py: given the task's real question, is this note's
content actually relevant to answering it? Reuses that script's judge
prompt/call for consistency with the earlier study's methodology.

Usage:
  scripts/judge_retrieval_confusion.py --out-dir scripts/out_retrieval_filter
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from judge_memory_relevance import call_judge, DEFAULT_BASE_URL, DEFAULT_MODEL  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

RUNS = {
    "control": [
        "runs/20260921T112121.240847222Z-openclaw-sweatlasqa-pilot30-amem-repo-control-sglang",
        "runs/20260921T185635.159523170Z-openclaw-sweatlasqa-pilot30-shuffle1-amem-repo-control-sglang",
    ],
    "both": [
        "runs/20260921T133243.122989422Z-openclaw-sweatlasqa-pilot30-amem-repo-both-sglang",
        "runs/20260921T230545.829562372Z-openclaw-sweatlasqa-pilot30-shuffle1-amem-repo-both-sglang",
    ],
}

HEADER_RE = re.compile(r"^Found (\d+) matching memories(?:, plus (\d+) linked to them)?:")
# Each real entry is "<n>. <content>(similarity NN%, id: <uuid>)" for a direct
# match, or "<n>. <content>(similarity NN%, linked -- did not match the query
# itself, id: <uuid>)" for a BFS-expanded note -- match up to that specific,
# unambiguous terminator rather than splitting on embedded "\ndigit. "
# patterns, which also occur INSIDE a note's own prose (many notes contain
# their own numbered steps, e.g. "consumer.py: 1. File placed... 2.
# Pre-checks...") and would otherwise fragment one note into several fake
# ones. The two terminator shapes were first treated as one pattern, which
# silently swallowed every linked entry into the content of the match before
# it (its `.*?` skipped straight past "linked -- ... id:" looking for the
# plain-match shape) -- confirmed via a run with n=637 extracted notes and
# via='match' on all of them, when independent call-count data said ~40% of
# results were via='link'.
ENTRY_RE = re.compile(
    r"(\d+)\.\s(.*?)\(similarity (\d+)%, (?:linked [—-]+ did not match the query itself, )?id: ([0-9a-f-]+)\)",
    re.DOTALL,
)


QUESTION_TAG_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL)


def load_task_question(task_id: str, qa_root: Path) -> str | None:
    for sub in ("qa", "tw"):
        instr_path = qa_root / "data" / sub / task_id / "instruction.md"
        if instr_path.is_file():
            text = instr_path.read_text()
            m = QUESTION_TAG_RE.search(text)
            if m:
                return m.group(1).strip()
    return None


def extract_notes_from_run(run_dir: Path, arm: str, qa_root: Path):
    rows = []
    question_cache: dict[str, str | None] = {}
    for task_dir in sorted(run_dir.glob("task-*/")):
        mo = re.match(r"^(task-[0-9a-f]+)-(\d+)$", task_dir.name)
        if not mo:
            continue
        task_id, occurrence = mo.group(1), mo.group(2)
        if task_id not in question_cache:
            question_cache[task_id] = load_task_question(task_id, qa_root)
        question = question_cache[task_id]
        if not question:
            continue

        for f in sorted(task_dir.glob("harness-turn-*/telemetry/*.jsonl")):
            if ".trajectory." in f.name:
                continue
            current_task_by_call_id: dict[str, str | None] = {}
            call_seq = 0
            for line in f.read_text().splitlines():
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                msg = e.get("message", {})
                if msg.get("role") == "assistant":
                    for c in (msg.get("content") or []):
                        if isinstance(c, dict) and c.get("type") == "toolCall" and c.get("name") == "memory_search":
                            call_id = c.get("id")
                            args = c.get("arguments") or {}
                            if call_id:
                                current_task_by_call_id[call_id] = args.get("current_task")
                    continue
                if msg.get("role") != "toolResult" or msg.get("toolName") != "memory_search":
                    continue
                text = (msg.get("content") or [{}])[0].get("text", "")
                header = HEADER_RE.match(text)
                if not header:
                    continue
                call_seq += 1
                current_task = current_task_by_call_id.get(msg.get("toolCallId"))
                n_match = int(header.group(1))
                body = text[header.end():]
                for m in ENTRY_RE.finditer(body):
                    rank = int(m.group(1))
                    content = m.group(2).strip()
                    sim = int(m.group(3))
                    note_id = m.group(4)
                    via = "match" if rank <= n_match else "link"
                    rows.append({
                        "arm": arm, "run_dir": str(run_dir.name), "task_id": task_id, "occurrence": occurrence,
                        "call_seq": call_seq,
                        "note_id": note_id, "rank": rank, "via": via, "similarity": sim,
                        "current_task": current_task,
                        "question": question, "note_content": content.strip(),
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "scripts" / "out_retrieval_filter")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-per-arm", type=int, default=0, help="0 = no cap")
    args = parser.parse_args()

    qa_root = REPO_ROOT / ".cache" / "swe-atlas-qa"
    all_rows = []
    for arm, run_dirs in RUNS.items():
        for rd in run_dirs:
            rows = extract_notes_from_run(REPO_ROOT / rd, arm, qa_root)
            print(f"{arm} / {rd}: {len(rows)} notes extracted", file=sys.stderr)
            all_rows.extend(rows)

    # de-dupe identical (arm, task_id, occurrence, note_id, rank) in case a
    # note appears in >1 search call within the same task -- keep first sighting
    seen = set()
    deduped = []
    for r in all_rows:
        key = (r["arm"], r["task_id"], r["occurrence"], r["note_id"], r["rank"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    all_rows = deduped
    print(f"total unique notes: {len(all_rows)}", file=sys.stderr)

    if args.max_per_arm:
        by_arm: dict[str, list] = {}
        for r in all_rows:
            by_arm.setdefault(r["arm"], []).append(r)
        capped = []
        for arm, rows in by_arm.items():
            capped.extend(rows[: args.max_per_arm])
        all_rows = capped
        print(f"capped to {len(all_rows)} notes ({args.max_per_arm}/arm)", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(all_rows):
        try:
            verdict = call_judge(args.base_url, args.model, row["question"], row["note_content"])
        except Exception as e:  # noqa: BLE001
            verdict = {"relevant": None, "reason": f"judge call failed: {e}"}
        row["llm_relevant"] = verdict["relevant"]
        row["llm_reason"] = verdict["reason"]
        print(f"[{i + 1}/{len(all_rows)}] arm={row['arm']} task={row['task_id']} rank={row['rank']} "
              f"via={row['via']} -> relevant={verdict['relevant']}", file=sys.stderr)

    out_path = args.out_dir / "retrieval_relevance_judged.csv"
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows to {out_path}")

    print("\n=== Relevant rate by arm ===")
    from collections import defaultdict
    counts = defaultdict(lambda: defaultdict(int))
    for r in all_rows:
        counts[r["arm"]][r["llm_relevant"]] += 1
    for arm, d in counts.items():
        total = sum(d.values())
        rel = d.get(True, 0)
        print(f"{arm}: n={total} relevant={rel} ({rel/total:.1%}) not_relevant={d.get(False,0)} unparsed={d.get(None,0)}")


if __name__ == "__main__":
    main()
