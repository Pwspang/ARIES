"""Parse OpenClaw telemetry for one session into an ordered list of Turns.

A "turn" is one assistant message node (possibly containing multiple tool
calls) together with the ancestor chain of context that preceded it: prior
user/assistant/toolResult nodes, and compaction summaries. Reads two files
per session:

  <sessionId>.trajectory.jsonl  -> context.compiled events (systemPrompt, tools)
  <sessionId>.jsonl             -> parent-linked message chain (the real per-turn detail)
"""
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    id: str
    kind: str  # "user" | "assistant_action" | "compaction"
    label: str
    openai_messages: list = None  # "user"/"compaction": the single-unit representation
    assistant_message: dict = None  # "assistant_action": just the assistant's message (text + tool_calls)
    tool_results: list = None  # "assistant_action": [{"tool_call_id":..., "content":...}, ...], independently maskable
    call_label: str = None
    result_label: str = None


@dataclass
class Turn:
    index: int
    node_id: str
    timestamp: str
    system_prompt: str
    tools: list  # OpenAI-format tool schemas
    chunks: list  # ordered list[Chunk], oldest first, up to but excluding this turn
    oracle_tool_calls: list  # [{"name":..., "arguments":...}, ...]
    oracle_text: str


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _tool_schema_to_openai(tools):
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def _latest_context_before(trajectory_events, timestamp):
    """Return (systemPrompt, tools) from the last context.compiled event at or before `timestamp`."""
    best = None
    for ev in trajectory_events:
        if ev.get("type") != "context.compiled":
            continue
        if ev.get("timestamp", "") <= timestamp:
            best = ev
    if best is None:
        # fall back to the first context.compiled event if none strictly precede
        for ev in trajectory_events:
            if ev.get("type") == "context.compiled":
                best = ev
                break
    data = best["data"]
    return data["systemPrompt"], _tool_schema_to_openai(data.get("tools", []))


def _normalize_content(content):
    """Message content is a plain string in some sessions, a list of {"type":"text","text":...}
    content-parts in others (seen in Agent_Bench-style runs) -- always return a plain string."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if part.get("type") == "text")


def _assistant_message_to_openai(node):
    parts = node["message"]["content"]
    text = "".join(p["text"] for p in parts if p.get("type") == "text")
    tool_calls = [p for p in parts if p.get("type") == "toolCall"]
    msg = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
            }
            for tc in tool_calls
        ]
    return msg, tool_calls, text


def extract_turns(telemetry_dir):
    telemetry_dir = Path(telemetry_dir)
    # ".checkpoint.<uuid>.jsonl" files are truncated mid-run snapshots of the main session
    # file (seen in Agent_Bench-style runs) -- not a second session, so exclude them too.
    session_files = [
        p
        for p in telemetry_dir.glob("*.jsonl")
        if not p.name.endswith(".trajectory.jsonl") and ".checkpoint." not in p.name
    ]
    if len(session_files) != 1:
        raise ValueError(f"expected exactly one session jsonl in {telemetry_dir}, found {session_files}")
    session_path = session_files[0]
    trajectory_path = telemetry_dir / (session_path.stem + ".trajectory.jsonl")

    nodes = _load_jsonl(session_path)
    trajectory_events = _load_jsonl(trajectory_path)
    by_id = {n["id"]: n for n in nodes if "id" in n}

    # Group toolResult nodes by the assistant node whose tool call they answer, matched by
    # toolCallId -- NOT by parentId. OpenClaw chains parallel tool results sequentially (each
    # result's parentId is the PREVIOUS result, not the original assistant node), so only the
    # first of several parallel results would ever be grouped correctly by parentId alone.
    tool_call_owner = {}
    for n in nodes:
        if n.get("type") == "message" and n["message"].get("role") == "assistant":
            for c in n["message"]["content"]:
                if c.get("type") == "toolCall":
                    tool_call_owner[c["id"]] = n["id"]

    tool_results_by_parent = {}
    for n in nodes:
        if n.get("type") == "message" and n["message"].get("role") == "toolResult":
            owner = tool_call_owner.get(n["message"]["toolCallId"])
            if owner:
                tool_results_by_parent.setdefault(owner, []).append(n)

    def node_to_chunk(node):
        if node["type"] == "compaction":
            return Chunk(
                id=node["id"],
                kind="compaction",
                label=f"compaction@{node['timestamp']}",
                openai_messages=[{"role": "user", "content": "[Compacted prior context]\n" + node["summary"]}],
            )
        role = node["message"]["role"]
        if role == "user":
            return Chunk(
                id=node["id"],
                kind="user",
                label=f"user@{node['timestamp']}",
                openai_messages=[{"role": "user", "content": _normalize_content(node["message"]["content"])}],
            )
        if role == "assistant":
            msg, tool_calls, _ = _assistant_message_to_openai(node)
            results_by_call_id = {}
            for tr in tool_results_by_parent.get(node["id"], []):
                tr_msg = tr["message"]
                text = "".join(c.get("text", "") for c in tr_msg.get("content", []) if c.get("type") == "text")
                results_by_call_id[tr_msg["toolCallId"]] = text
            # order results to match the tool_calls order, not whatever order they completed in
            tool_results = [
                {"tool_call_id": tc["id"], "content": results_by_call_id[tc["id"]]}
                for tc in tool_calls
                if tc["id"] in results_by_call_id
            ]
            call_label = f"call@{node['timestamp']}" + (f" ({len(tool_calls)} tool calls)" if tool_calls else "")
            result_label = f"result@{node['timestamp']}" + (f" ({len(tool_results)} outputs)" if tool_results else "")
            return Chunk(
                id=node["id"],
                kind="assistant_action",
                label=call_label,
                assistant_message=msg,
                tool_results=tool_results,
                call_label=call_label,
                result_label=result_label,
            )
        raise ValueError(f"unexpected role for chunk conversion: {role}")

    def ancestor_chain(node_id):
        """Walk parentId back to root, or to the nearest compaction node (inclusive), whichever is nearer.

        A compaction event replaces everything before it with a summary, so history
        further back than the nearest compaction was never actually seen by the model.
        """
        chain = []
        cur = by_id.get(node_id)
        while cur is not None:
            chain.append(cur)
            if cur["type"] == "compaction":
                break
            parent_id = cur.get("parentId")
            cur = by_id.get(parent_id) if parent_id else None
        chain.reverse()
        return chain

    assistant_nodes = [n for n in nodes if n.get("type") == "message" and n["message"].get("role") == "assistant"]

    turns = []
    for idx, anode in enumerate(assistant_nodes):
        ancestors = ancestor_chain(anode["parentId"])
        # toolResult nodes are already folded into the preceding assistant_action chunk
        chunks = [
            node_to_chunk(n)
            for n in ancestors
            if n["type"] == "compaction" or (n["type"] == "message" and n["message"].get("role") != "toolResult")
        ]
        system_prompt, tools = _latest_context_before(trajectory_events, anode["timestamp"])
        _, oracle_tool_calls, oracle_text = _assistant_message_to_openai(anode)
        turns.append(
            Turn(
                index=idx,
                node_id=anode["id"],
                timestamp=anode["timestamp"],
                system_prompt=system_prompt,
                tools=tools,
                chunks=chunks,
                oracle_tool_calls=[{"name": tc["name"], "arguments": tc["arguments"]} for tc in oracle_tool_calls],
                oracle_text=oracle_text,
            )
        )
    return turns
