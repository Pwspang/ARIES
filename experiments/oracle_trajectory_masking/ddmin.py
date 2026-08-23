"""Generic delta-debugging minimization (Zeller's ddmin), operating on a list of opaque ids.

`test_batch(list_of_candidate_id_lists)` must return a same-length list of bools, one per
candidate. Candidates within a round are independent (the model call for one masked context
doesn't depend on another), so `test_batch` is expected to evaluate them concurrently.

The caller must already have confirmed the full id list reproduces the oracle action -- ddmin
does NOT re-verify this itself. At temperature=0 on a batched inference server, re-querying the
exact same full context can occasionally return a different result than the first check (batch
composition affects numerics), so re-testing it here would make minimization spuriously flaky
for a case the caller already established.
"""


def ddmin(ids, test_batch):
    ids = list(ids)

    n = 2
    current = ids
    while len(current) >= 2:
        chunk_size = max(1, len(current) // n)
        subsets = [current[i : i + chunk_size] for i in range(0, len(current), chunk_size)]
        complements = [[x for x in current if x not in subset] for subset in subsets]

        # evaluate every subset and complement for this round in one batch (parallel-friendly)
        candidates = subsets + complements
        results = test_batch(candidates)
        subset_results, complement_results = results[: len(subsets)], results[len(subsets) :]

        reduced = False
        for subset, ok in zip(subsets, subset_results):
            if ok:
                current = subset
                n = max(n - 1, 2)
                reduced = True
                break
        if reduced:
            continue

        for complement, ok in zip(complements, complement_results):
            if complement and ok:
                current = complement
                n = max(n - 1, 2)
                reduced = True
                break
        if reduced:
            continue

        if n >= len(current):
            break
        n = min(n * 2, len(current))

    # the main loop only ever compares subsets/complements of >=2 candidates, so a single
    # remaining candidate is never itself tested for removal -- check that boundary case here
    if len(current) == 1 and test_batch([[]])[0]:
        current = []

    return current
