"""Turn a Turn's system prompt + conversation chunks into an addressable, maskable unit list.

The system prompt is one maskable unit ("sys"). Each prior user/compaction turn is one unit
("chunk:<id>"). Each prior assistant turn is split into two INDEPENDENTLY maskable units --
its tool call(s) ("call:<id>") and the resulting tool output(s) ("result:<id>") -- so the
search can tell apart "does the model need to remember making this call" from "does the
model need to have seen this output."

A native OpenAI tool-call/tool-result pair is only valid together (a `tool` role message must
answer a `tool_calls` entry in the immediately preceding assistant message), so the two
"mismatched" cases fall back to a safe representation instead of producing an invalid request:
  - call kept, output masked  -> the assistant message keeps its real tool_calls, but each
    `tool` message's content is replaced with a redacted placeholder (still a valid pair).
  - output kept, call masked  -> there's no tool_calls to answer, so the output is instead
    folded into a plain user-role message ("here is a result from an earlier, unseen action").
"""

SYSTEM_UNIT_ID = "sys"


def unit_ids(turn):
    """All maskable unit ids for a turn, in order: system prompt, then per prior turn --
    one id for a user/compaction chunk, or two (call + result) for an assistant chunk."""
    ids = [SYSTEM_UNIT_ID]
    for c in turn.chunks:
        if c.kind == "assistant_action":
            ids.append(f"call:{c.id}")
            if c.tool_results:
                ids.append(f"result:{c.id}")
        else:
            ids.append(f"chunk:{c.id}")
    return ids


def pinned_unit_ids(turn):
    """Unit id(s) belonging to the single most recent prior turn -- always kept, never a
    candidate for masking. The model needs *some* immediately-preceding turn to anchor how it
    continues the conversation and formats its next tool call; searching over whether to drop
    it too would conflate "is earlier history necessary" with "can the model even parse a
    request with no immediately-prior turn at all," which isn't the question this experiment
    is asking. Returns [] if there's no prior turn (turn 0)."""
    if not turn.chunks:
        return []
    last = turn.chunks[-1]
    if last.kind == "assistant_action":
        ids = [f"call:{last.id}"]
        if last.tool_results:
            ids.append(f"result:{last.id}")
        return ids
    return [f"chunk:{last.id}"]


def _find_chunk(turn, chunk_id):
    for c in turn.chunks:
        if c.id == chunk_id:
            return c
    return None


def unit_label(turn, unit_id):
    if unit_id == SYSTEM_UNIT_ID:
        return "system_prompt"
    kind, _, chunk_id = unit_id.partition(":")
    c = _find_chunk(turn, chunk_id)
    if c is None:
        return unit_id
    if kind == "call":
        return c.call_label
    if kind == "result":
        return c.result_label
    return c.label


def build_request(turn, kept_unit_ids):
    """Build {system_prompt, messages, tools} for the model call, keeping only `kept_unit_ids`."""
    kept = set(kept_unit_ids)
    system_prompt = turn.system_prompt if SYSTEM_UNIT_ID in kept else ""

    messages = []
    for c in turn.chunks:
        if c.kind != "assistant_action":
            if f"chunk:{c.id}" in kept:
                messages.extend(c.openai_messages)
            continue

        call_kept = f"call:{c.id}" in kept
        result_kept = bool(c.tool_results) and f"result:{c.id}" in kept

        if call_kept and result_kept:
            messages.append(c.assistant_message)
            for tr in c.tool_results:
                messages.append({"role": "tool", "tool_call_id": tr["tool_call_id"], "content": tr["content"]})
        elif call_kept:  # result masked (or no results ever existed)
            messages.append(c.assistant_message)
            for tr in c.tool_results:
                messages.append({"role": "tool", "tool_call_id": tr["tool_call_id"], "content": "[output redacted]"})
        elif result_kept:  # call masked, output kept
            combined = "\n\n".join(tr["content"] for tr in c.tool_results)
            messages.append(
                {"role": "user", "content": "[Result of an earlier, unseen action]\n" + combined}
            )
        # else: both masked -> nothing for this chunk

    return {"system_prompt": system_prompt, "messages": messages, "tools": turn.tools}
