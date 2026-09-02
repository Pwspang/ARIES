"""Map each turn's minimal surviving unit set (from turns.jsonl) back to the source nodes those
units came from, producing a turn-level dependency graph over a UNIFIED TIMELINE that includes
not just assistant turns but the user messages and compaction summaries interleaved between them
(previously these were flattened into per-turn "external input" flags; they're now first-class
timeline nodes so a sliding-window / pinning policy can be evaluated against them directly).

A unit id is "call:<node_id>" / "result:<node_id>" / "chunk:<node_id>", where <node_id> is the
raw telemetry node id of the chunk it was built from (context_chunks.unit_ids, trajectory_extractor
.node_to_chunk). For an assistant_action chunk, that node_id is exactly the node_id of some earlier
Turn (trajectory_extractor.extract_turns sets Turn.node_id = anode["id"]). A "chunk:<id>" unit is
either a user message or a compaction summary -- also given its own timeline slot, keyed by the
same node_id.

turns.jsonl only stores the human-readable label (unit_label), not the raw unit id, so this
re-extracts turns from the same telemetry dir and recomputes unit_ids()/unit_label() per turn.
Labels are NOT guaranteed unique within a turn -- two distinct chunks can land in the same
millisecond and produce byte-identical "call@<ts> (N tool calls)" strings (observed in practice,
e.g. turn 37 of dr_009, from an eviction/fanout burst). A label->id dict would silently collapse
such collisions to whichever id was seen last. Since minimal_unit_labels is always a subsequence
of unit_ids(turn) in original order (build_request preserves chunk order; ddmin only removes
units, never reorders them), duplicates are instead resolved positionally: walk the turn's full
ordered (label, id) list and the minimal label list in lockstep, consuming a full-list entry only
when its label matches the next minimal label still to be consumed.

Usage:
  python dependency_graph.py --telemetry-dir <run>/.../conversation/sessions \
      --turns-jsonl results/eviction-fanout-run-20260819T020623Z/dr_009/turns.jsonl
"""
import argparse
import heapq
import json
import re
from collections import defaultdict

from context_chunks import unit_ids, unit_label
from trajectory_extractor import extract_turns

_TS_RE = re.compile(r"@([0-9T:.\-Z]+)")


def resolve_minimal_ids(turn, minimal_labels):
    """Match each label in minimal_labels to its unit id, positionally: minimal_labels is always
    a subsequence of [unit_label(turn, u) for u in unit_ids(turn)] in the same order, so walk both
    lists in lockstep rather than using a label->id dict (which breaks on duplicate labels)."""
    full_ids = unit_ids(turn)
    full_labels = [unit_label(turn, u) for u in full_ids]
    resolved = []
    fi = 0
    for label in minimal_labels:
        while fi < len(full_labels) and full_labels[fi] != label:
            fi += 1
        if fi < len(full_labels):
            resolved.append((label, full_ids[fi]))
            fi += 1
        else:
            resolved.append((label, None))
    return resolved


def _tool_type(turn):
    """Classify a turn's node type by the distinct tool names it called, e.g. "web_search",
    "web_search+web_fetch" for a turn that called both. Turns with no tool calls (a final text
    answer) are labeled "text"."""
    names = sorted(set(tc["name"] for tc in turn.oracle_tool_calls))
    return "+".join(names) if names else "text"


def _label_timestamp(label):
    m = _TS_RE.search(label)
    return m.group(1) if m else None


def build_timeline(turns):
    """Unify every assistant turn plus every distinct user/compaction chunk referenced anywhere
    into one causally ordered timeline. Returns (timeline, node_id_to_pos) where timeline is a
    list of {pos, node_id, kind, type_label, timestamp, turn_index (or None)}.

    Ordering is NOT a plain sort by timestamp: this telemetry has bursts where many nodes --
    including a turn and its own ancestor chunks -- share the exact same millisecond timestamp
    (observed e.g. in dr_009 turn 34, whose own timestamp is identical to three of its ancestor
    call chunks'). A naive (timestamp, turn_index-or-sentinel) sort breaks ties by shoving every
    non-assistant chunk after every assistant node at that timestamp, which can place an ancestor
    AFTER the turn that depends on it -- producing backwards-pointing "dependency" arcs. Instead,
    each turn's own chunk list is a real parent-child chain (guaranteed correctly ordered, oldest
    to newest, ending at the turn itself), so those adjacent pairs become "happens-before"
    constraints and the whole timeline is built with a topological sort over them. Timestamp is
    used only as a tiebreak among nodes with no ordering constraint between them (parallel
    fanout branches, which have no true relative order anyway)."""
    entries = {}  # node_id -> entry
    before = defaultdict(set)  # node_id -> set of node_ids it must come before

    for t in turns:
        entries[t.node_id] = {
            "node_id": t.node_id,
            "kind": "assistant",
            "type_label": _tool_type(t),
            "timestamp": t.timestamp,
            "turn_index": t.index,
        }
        chain_ids = []
        for c in t.chunks:
            if c.kind != "assistant_action" and c.id not in entries:
                entries[c.id] = {
                    "node_id": c.id,
                    "kind": c.kind,  # "user" | "compaction"
                    "type_label": c.kind,
                    "timestamp": _label_timestamp(c.label),
                    "turn_index": None,
                }
            chain_ids.append(c.id)
        chain_ids.append(t.node_id)
        for a, b in zip(chain_ids, chain_ids[1:]):
            before[a].add(b)

    def sort_key(node_id):
        e = entries[node_id]
        return (e["timestamp"] or "", e["turn_index"] if e["turn_index"] is not None else 1 << 30, node_id)

    indegree = {nid: 0 for nid in entries}
    for a, bs in before.items():
        for b in bs:
            indegree[b] += 1

    ready = [(sort_key(nid), nid) for nid, d in indegree.items() if d == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, nid = heapq.heappop(ready)
        order.append(nid)
        for nxt in before.get(nid, ()):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(ready, (sort_key(nxt), nxt))

    if len(order) != len(entries):
        # a real cycle shouldn't be possible (constraints come from acyclic parent chains), but
        # don't silently drop nodes if telemetry ever violates that -- append whatever's left in
        # timestamp order so the timeline stays complete, just with a possibly-wrong tail.
        missing = sorted(set(entries) - set(order), key=sort_key)
        order.extend(missing)

    timeline = []
    node_id_to_pos = {}
    for pos, nid in enumerate(order):
        e = entries[nid]
        e["pos"] = pos
        timeline.append(e)
        node_id_to_pos[nid] = pos
    return timeline, node_id_to_pos


def turn_dependencies(turns, node_id_to_pos):
    """Returns {turn_index: [record, ...]} where record is one of:
    {"depends_on_turn": M, "via": "call"|"result", "label": <label>, "pos": <timeline pos>}
    {"depends_on_chunk": "user"|"compaction", "label": <label>, "pos": <timeline pos>}
    parsed from each turn's own minimal_unit_labels (must be attached as turn._minimal_labels)."""
    node_id_to_turn_index = {t.node_id: t.index for t in turns}
    deps = {}
    for t in turns:
        minimal_labels = getattr(t, "_minimal_labels", None)
        if minimal_labels is None:
            continue
        records = []
        for label, unit_id in resolve_minimal_ids(t, minimal_labels):
            if label == "system_prompt":
                continue
            if unit_id is None:
                records.append({"unresolved_label": label})
                continue
            kind, _, node_id = unit_id.partition(":")
            pos = node_id_to_pos.get(node_id)
            if kind in ("call", "result") and node_id in node_id_to_turn_index:
                records.append({"depends_on_turn": node_id_to_turn_index[node_id], "via": kind, "label": label, "pos": pos})
            elif kind == "chunk":
                ext_kind = "compaction" if label.startswith("compaction@") else "user"
                records.append({"depends_on_chunk": ext_kind, "label": label, "pos": pos})
            else:
                records.append({"unresolved_label": label})
        deps[t.index] = records
    return deps


def build_task_graph(telemetry_dir, turns_jsonl_path):
    """Returns {"timeline": [...], "turns": {turn_index: {...}}} for one task. `timeline` is the
    unified node sequence (see build_timeline); `turns[idx]["deps"]` entries carry `pos`, the
    timeline position of the depended-on node, so a caller can compute window/pin coverage
    without re-deriving turn-index vs. chunk-position arithmetic."""
    turns = extract_turns(telemetry_dir)
    by_index = {t.index: t for t in turns}
    timeline, node_id_to_pos = build_timeline(turns)

    records_by_index = {}
    with open(turns_jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            records_by_index[rec["turn_index"]] = rec
            t = by_index.get(rec["turn_index"])
            if t is not None and rec.get("minimal_unit_labels") is not None:
                t._minimal_labels = rec["minimal_unit_labels"]

    deps = turn_dependencies(turns, node_id_to_pos)
    out_turns = {}
    for idx, rec in records_by_index.items():
        out_turns[idx] = {
            "deps": deps.get(idx, []),
            "pos": node_id_to_pos.get(by_index[idx].node_id) if idx in by_index else None,
            "oracle_command": rec.get("oracle_command"),
            "reference_command": rec.get("reference_command"),
            "reproducible": rec.get("reproducible"),
            "oracle_matches_reference": rec.get("oracle_matches_reference"),
            "total_units": rec.get("total_units"),
            "minimal_units": rec.get("minimal_units"),
            "error": rec.get("error"),
        }
    return {"timeline": timeline, "turns": out_turns}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--telemetry-dir", required=True)
    p.add_argument("--turns-jsonl", required=True)
    args = p.parse_args()

    graph = build_task_graph(args.telemetry_dir, args.turns_jsonl)
    deps = {idx: v["deps"] for idx, v in graph["turns"].items()}
    for idx in sorted(deps):
        print(f"turn {idx}:")
        if not deps[idx]:
            print("  (no dependencies -- minimal context was empty or system-prompt only)")
        for r in deps[idx]:
            if "depends_on_turn" in r:
                print(f"  -> depends on turn {r['depends_on_turn']} ({r['via']}): {r['label']}")
            elif "depends_on_chunk" in r:
                print(f"  -> depends on {r['depends_on_chunk']} chunk: {r['label']}")
            else:
                print(f"  -> UNRESOLVED: {r['unresolved_label']}")


if __name__ == "__main__":
    main()
