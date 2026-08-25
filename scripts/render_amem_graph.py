#!/usr/bin/env python3
"""Render an interactive amem memory graph (nodes = memories, edges =
memory_relations) as a self-contained HTML file, from a run's captured amem
SQLite state (see pkg/harness/openclaw/harness.go's collectAMEMDatabase).

Usage:
    render_amem_graph.py <path/to/memory.db> [output.html]

With no output path, writes graph.html next to memory.db (the same
directory) — the file the runtime containers copy out to
<run>/<task>/harness-turn-01/amem/.amem/.

To render every task in a run at once:
    for db in runs/<run>/*/harness-turn-01/amem/.amem/memory.db; do
        scripts/render_amem_graph.py "$db"
    done
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / "amem-graph-template.html"


def short_label(content, limit=20):
    head = re.split(r"[：:]", content, maxsplit=1)[0]
    if len(head) > limit:
        head = head[:limit] + "…"
    return head


def load_graph(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT id, content, type, tags, confidence, tier FROM memories ORDER BY created_at")
    memories = [dict(r) for r in cur.fetchall()]
    cur.execute("SELECT id, from_id, to_id, relationship_type, strength FROM memory_relations ORDER BY created_at")
    relations = [dict(r) for r in cur.fetchall()]
    con.close()
    return memories, relations


def render(db_path, task_label):
    memories, relations = load_graph(db_path)
    relation_types = sorted(set(r["relationship_type"] for r in relations))

    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    if relations:
        subtitle = (
            f"{len(memories)} facts the agent stored via amem__memory_store during "
            f"Deep Research Bench task {task_label}, linked by amem__memory_relate "
            f"({len(relations)} relations)."
        )
        aria = (
            f"Force-directed graph of {len(memories)} stored memories connected by "
            f"{len(relations)} typed relations ({', '.join(relation_types)})."
        )
    else:
        subtitle = (
            f"{len(memories)} facts the agent stored via amem__memory_store during "
            f"Deep Research Bench task {task_label} — no memory_relate calls were "
            f"made this run, so no relations exist to show."
        )
        aria = f"{len(memories)} stored memories with no relations between them."

    data_json = json.dumps({"memories": memories, "relations": relations}, ensure_ascii=False)
    data_json = data_json.replace("</", "<\\/")

    out = template
    out = out.replace("__TITLE__", f"amem memory graph — task {task_label}")
    out = out.replace("__SUBTITLE__", subtitle)
    out = out.replace("__MEMORY_COUNT__", str(len(memories)))
    out = out.replace("__RELATION_COUNT__", str(len(relations)))
    out = out.replace("__RELATION_TYPE_COUNT__", str(len(relation_types)))
    out = out.replace("__ARIA_LABEL__", aria)
    out = out.replace("__DATA_JSON__", data_json)
    return out


def task_label_from_path(db_path):
    # .../<run>/<task_id>/harness-turn-01/amem/.amem/memory.db
    parts = db_path.resolve().parts
    try:
        idx = parts.index("harness-turn-01")
        return parts[idx - 1]
    except ValueError:
        return db_path.parent.name


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    db_path = Path(sys.argv[1])
    if not db_path.is_file():
        print(f"no such file: {db_path}", file=sys.stderr)
        sys.exit(1)
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else db_path.parent / "graph.html"

    task_label = task_label_from_path(db_path)
    html = render(db_path, task_label)
    out_path.write_text(html, encoding="utf-8")
    print(f"{task_label}: wrote {out_path}")


if __name__ == "__main__":
    main()
