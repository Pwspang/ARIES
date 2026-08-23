"""Run the minimal-working-memory search over one task's oracle trajectory.

Usage:
  python run_experiment.py --telemetry-dir <run>/<task>/harness/telemetry \
      --task-id 71-008 --base-url http://localhost:8100/v1 \
      --model Qwen/Qwen3.6-35B-A3B-FP8 [--api-key-env SGLANG_API_KEY]

Writes results/<task_id>/turns.jsonl incrementally, one line per completed turn,
so progress can be inspected on disk while the run is still in flight.

Each turn's minimization target is NOT the historical oracle action from the telemetry --
it's a self-consistent reference established by sampling the full (unmasked) context several
times and taking the majority-agreeing cluster. The recorded oracle action may itself have been
an arbitrary, non-majority sample (the collection run wasn't temperature-pinned), so asking
replay to reproduce it exactly conflates "is this context sufficient" with "did we get the same
sample as last time." Minimizing against this run's own stable answer isolates the former.
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from context_chunks import build_request, unit_ids, unit_label
from ddmin import ddmin
from match import actions_match, establish_reference, fuzzy_actions_match, render_action
from model_client import ModelClient
from trajectory_extractor import extract_turns


def make_test_batch_fn(client, turn, pool, match_fn, target_tool_calls, target_text, pinned, samples=3):
    """`samples` repeated calls per candidate, majority vote against the established reference --
    a single temperature=0 call on this server isn't reliably reproducible even for byte-identical
    input (confirmed: the same request fired twice returned two different actions), so voting
    smooths over that per-call noise.

    `pinned` is unioned in before every call (currently always empty -- nothing is pinned, so ddmin
    can remove the most recent prior turn too, unlike earlier runs).

    Every individual (candidate, sample) HTTP call is flattened into one flat list of tasks and
    submitted to `pool` together -- NOT one pool.map call per candidate with samples run
    sequentially inside each worker. The latter means a ddmin round with few candidates (early
    rounds only produce ~4) can never use more than ~4 of the pool's workers even if it's sized
    for many more, since each worker is tied up running its candidate's 5 samples one at a time."""

    def test_batch(candidates):
        reqs = [build_request(turn, list(pinned) + list(c)) for c in candidates]
        flat_tasks = [(ci, req) for ci, req in enumerate(reqs) for _ in range(samples)]
        flat_predictions = list(pool.map(lambda item: client.complete(item[1]["system_prompt"], item[1]["messages"], item[1]["tools"]), flat_tasks))

        grouped = [[] for _ in candidates]
        for (ci, _), predicted in zip(flat_tasks, flat_predictions):
            grouped[ci].append(predicted)

        return [sum(match_fn(target_tool_calls, target_text, p) for p in preds) > samples / 2 for preds in grouped]

    return test_batch


def process_turn(client, turn, pool, task_id, match_fn, samples):
    ids = unit_ids(turn)
    req = build_request(turn, ids)

    full_predictions = list(pool.map(lambda _: client.complete(req["system_prompt"], req["messages"], req["tools"]), range(samples)))
    reference, stable, agreement = establish_reference(full_predictions, match_fn)

    oracle_command = render_action(turn.oracle_tool_calls, turn.oracle_text)
    if reference is None:
        reference_command = "(no self-consistent action -- all samples disagreed or errored)"
        oracle_matches_reference = False
    else:
        reference_command = render_action(reference.get("tool_calls") or [], reference.get("text") or "")
        oracle_matches_reference = match_fn(turn.oracle_tool_calls, turn.oracle_text, reference)

    # nothing pinned: every unit, including the most recent prior turn, is a candidate for removal
    pinned = []
    candidates = [u for u in ids if u not in pinned]

    record = {
        "task_id": task_id,
        "turn_index": turn.index,
        "total_units": len(ids),
        "pinned_unit_labels": [unit_label(turn, u) for u in pinned],
        "reproducible": stable,  # "stable" = the model's own full-context answer is self-consistent
        "agreement_fraction": round(agreement, 2),
        "oracle_command": oracle_command,
        "reference_command": reference_command,
        "oracle_matches_reference": oracle_matches_reference,
    }
    if stable:
        target_tool_calls = reference.get("tool_calls") or []
        target_text = reference.get("text") or ""
        test_batch = make_test_batch_fn(client, turn, pool, match_fn, target_tool_calls, target_text, pinned, samples)
        minimal_candidates = ddmin(candidates, test_batch) if candidates else []
        minimal = [u for u in ids if u in pinned or u in minimal_candidates]
        record["minimal_units"] = len(minimal)
        record["minimal_unit_labels"] = [unit_label(turn, u) for u in minimal]
    else:
        record["minimal_units"] = None
        record["minimal_unit_labels"] = None

    print(
        f"turn {turn.index}: stable={stable} (agreement={agreement:.2f}) "
        f"total={len(ids)} minimal={record['minimal_units']}\n"
        f"  oracle:    {oracle_command}\n"
        f"  reference: {reference_command}"
    )
    return record


def run(telemetry_dir, task_id, base_url, model, api_key_env, out_path, turn_workers, calls_per_turn_workers, use_fuzzy, tool_filter=None, samples=3, timeout=300):
    turns = extract_turns(telemetry_dir)
    if tool_filter:
        # require every call in the turn to be an allowed tool, not just one -- a turn mixing
        # e.g. web_search with web_fetch would still be dragged down by web_fetch's inherent
        # pick-one-of-many-equally-good-candidates instability
        turns = [t for t in turns if t.oracle_tool_calls and all(tc["name"] in tool_filter for tc in t.oracle_tool_calls)]
    client = ModelClient(base_url=base_url, model=model, api_key_env=api_key_env, temperature=0, timeout=timeout)
    match_fn = fuzzy_actions_match if use_fuzzy else actions_match

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    out_file = open(out_path, "w")

    def process_and_write(turn):
        try:
            record = process_turn(client, turn, call_pool, task_id, match_fn, samples)
        except Exception as e:  # noqa: BLE001 -- one flaky turn must not take down the whole run
            print(f"turn {turn.index}: FAILED ({e!r})")
            record = {"task_id": task_id, "turn_index": turn.index, "error": repr(e)}
        with write_lock:
            out_file.write(json.dumps(record) + "\n")
            out_file.flush()
        return record

    # two pool levels: one turn's ddmin rounds fan out concurrently (calls_per_turn_workers),
    # and multiple turns' searches run concurrently too (turn_workers) -- both are independent
    # HTTP calls against the same model server, which can serve them concurrently.
    with ThreadPoolExecutor(max_workers=calls_per_turn_workers) as call_pool:
        with ThreadPoolExecutor(max_workers=turn_workers) as turn_pool:
            list(turn_pool.map(process_and_write, turns))

    out_file.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--telemetry-dir", required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--base-url", default="http://localhost:8100/v1")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    p.add_argument("--api-key-env", default="SGLANG_API_KEY")
    p.add_argument("--out", default=None)
    p.add_argument("--turn-workers", type=int, default=4, help="turns processed concurrently")
    p.add_argument("--calls-per-turn-workers", type=int, default=6, help="concurrent model calls within one turn's ddmin round")
    p.add_argument("--exact-match", action="store_true", help="use strict exact-match instead of the fuzzy comparator")
    p.add_argument(
        "--tools",
        default="web_search,web_fetch",
        help="comma-separated tool names to restrict the search to (turns whose oracle action doesn't call one of "
        "these are skipped entirely); pass an empty string to process all turns",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=3,
        help="repeated model calls used both to establish the self-consistent reference action and, per "
        "candidate, to vote on whether it reproduces that reference",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="per-request timeout in seconds -- late, context-heavy turns can take a while to prefill",
    )
    args = p.parse_args()

    # each invocation gets its own timestamped results folder so a later run never overwrites
    # an earlier one -- pass --out explicitly to opt out (e.g. to append into an existing subset run)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or f"results/run-{run_stamp}/{args.task_id}/turns.jsonl"
    tool_filter = set(t.strip() for t in args.tools.split(",") if t.strip()) or None
    run(
        args.telemetry_dir,
        args.task_id,
        args.base_url,
        args.model,
        args.api_key_env,
        out,
        args.turn_workers,
        args.calls_per_turn_workers,
        use_fuzzy=not args.exact_match,
        tool_filter=tool_filter,
        samples=args.samples,
        timeout=args.timeout,
    )
