#!/usr/bin/env python3
"""Share of CodeRAG (claude-context MCP) calls among all agent tool calls.

Reads OpenClaw session exports under each task's harness[-turn-NN]/telemetry/
(the only place MCP calls are recorded; they bypass the SSH ToolBridge) and
joins evaluation results from run-result.json.

Usage: retrieval_share.py [--json] [--repos repos.json] RUN_DIR [RUN_DIR...]

retrieval = search_code calls / all tool calls. OpenClaw names MCP tools
<server>__<tool>, e.g. coderag__search_code.
"""
import argparse
import collections
import json
import os
import pathlib
import sys

CODERAG_TOOLS = {"search_code", "index_codebase", "get_indexing_status", "clear_index"}
RETRIEVAL_TOOLS = {"search_code"}


def coderag_tool(name: str) -> str | None:
    """Base claude-context tool name, accepting an MCP server prefix (e.g. coderag__search_code)."""
    for sep in ("__", ".", ":", "/", "-"):
        if sep in name:
            tail = name.rsplit(sep, 1)[-1]
            if tail in CODERAG_TOOLS and name != tail:
                return tail
    return name if name in CODERAG_TOOLS else None


def session_files(task_dir: pathlib.Path):
    # Current layout: harness/telemetry; older runs: harness-turn-NN/telemetry.
    paths = list(task_dir.glob("harness/telemetry/*.jsonl")) + list(task_dir.glob("harness-turn-*/telemetry/*.jsonl"))
    for path in sorted(paths):
        if ".trajectory" not in path.name:
            yield path


def count_task(task_dir: pathlib.Path):
    calls = collections.Counter()
    errors = collections.Counter()
    pending = {}
    for path in session_files(task_dir):
        for line in path.open():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = entry.get("message") or {}
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "toolCall":
                        name = part.get("name") or "?"
                        calls[name] += 1
                        pending[part.get("id")] = name
            if message.get("role") == "toolResult" and message.get("isError"):
                name = pending.get(message.get("toolCallId"), message.get("toolName") or "?")
                errors[name] += 1
    return calls, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=pathlib.Path)
    state = pathlib.Path(os.environ.get("CODERAG_STATE") or pathlib.Path(__file__).resolve().parents[2] / ".cache" / "coderag")
    parser.add_argument("--repos", type=pathlib.Path, default=state / "repos.json",
                        help="slug -> tasks map from `coderag.sh repos` (default: $CODERAG_STATE/repos.json)")
    parser.add_argument("--json", action="store_true", help="emit per-task rows and aggregates as JSON")
    args = parser.parse_args()

    repo_of = {}
    if args.repos.exists():
        for slug, entry in json.loads(args.repos.read_text()).items():
            for task in entry["tasks"]:
                repo_of[task] = slug

    rows = []
    for run in args.runs:
        result_path = run / "run-result.json"
        results = {}
        if result_path.exists():
            for task in json.loads(result_path.read_text()).get("tasks", []):
                results[task["task_id"]] = task
        for task_dir in sorted(p for p in run.glob("task-*") if p.is_dir()):
            calls, errors = count_task(task_dir)
            if not calls and task_dir.name not in results:
                continue
            per_tool = collections.Counter()
            for name, n in calls.items():
                base = coderag_tool(name)
                if base:
                    per_tool[base] += n
            total = sum(calls.values())
            retrieval = sum(per_tool[t] for t in RETRIEVAL_TOOLS)
            coderag = sum(per_tool.values())
            evaluation = (results.get(task_dir.name) or {}).get("evaluation") or {}
            base_id = task_dir.name.rsplit("-", 1)[0]
            rows.append({
                "run": run.name,
                "task": task_dir.name,
                "repo": repo_of.get(base_id, "?"),
                "total_calls": total,
                "coderag_calls": coderag,
                "retrieval_calls": retrieval,
                "retrieval_share": retrieval / total if total else None,
                "coderag_share": coderag / total if total else None,
                "coderag_by_tool": dict(per_tool),
                "coderag_errors": sum(n for name, n in errors.items() if coderag_tool(name)),
                "index_mutation_attempts": per_tool["index_codebase"] + per_tool["clear_index"],
                "all_tools": dict(calls),
                "score": evaluation.get("score"),
                "reward": evaluation.get("reward"),
                "eval_status": evaluation.get("status"),
            })

    def aggregate(group):
        total = sum(r["total_calls"] for r in group)
        retrieval = sum(r["retrieval_calls"] for r in group)
        coderag = sum(r["coderag_calls"] for r in group)
        shares = [r["retrieval_share"] for r in group if r["retrieval_share"] is not None]
        scores = [r["score"] for r in group if r["score"] is not None]
        return {
            "tasks": len(group),
            "total_calls": total,
            "retrieval_calls": retrieval,
            "coderag_calls": coderag,
            "retrieval_share_micro": retrieval / total if total else None,
            "retrieval_share_macro": sum(shares) / len(shares) if shares else None,
            "tasks_using_retrieval": sum(1 for r in group if r["retrieval_calls"]),
            "index_mutation_attempts": sum(r["index_mutation_attempts"] for r in group),
            "mean_score": sum(scores) / len(scores) if scores else None,
        }

    by_repo = collections.defaultdict(list)
    for row in rows:
        by_repo[row["repo"]].append(row)
    summary = {"overall": aggregate(rows), "by_repo": {k: aggregate(v) for k, v in sorted(by_repo.items())}}

    if args.json:
        json.dump({"tasks": rows, "summary": summary}, sys.stdout, indent=2)
        print()
        return 0

    pct = lambda v: "   -  " if v is None else f"{100 * v:5.1f}%"
    print(f"{'task':40} {'repo':28} {'calls':>5} {'rag':>4} {'share':>6} {'score':>5}")
    for r in rows:
        score = "-" if r["score"] is None else f"{r['score']:.2f}"
        flag = "  !index-mutation" if r["index_mutation_attempts"] else ""
        print(f"{r['task']:40} {r['repo']:28} {r['total_calls']:5} {r['retrieval_calls']:4} {pct(r['retrieval_share'])} {score:>5}{flag}")
    print()
    print(f"{'repo':28} {'tasks':>5} {'calls':>6} {'rag':>5} {'micro':>6} {'macro':>6} {'used':>4} {'score':>5}")
    for name, agg in list(summary["by_repo"].items()) + [("OVERALL", summary["overall"])]:
        score = "-" if agg["mean_score"] is None else f"{agg['mean_score']:.2f}"
        print(f"{name:28} {agg['tasks']:5} {agg['total_calls']:6} {agg['retrieval_calls']:5} "
              f"{pct(agg['retrieval_share_micro'])} {pct(agg['retrieval_share_macro'])} {agg['tasks_using_retrieval']:4} {score:>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
