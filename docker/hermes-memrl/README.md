# Hermes + MemRL memory provider

This directory builds a Hermes image with a MemRL memory provider baked in.
The base image is the pinned `docker.io/nousresearch/hermes-agent:v2026.5.29.2`,
and the provider is added as the bundled plugin `plugins/memory/memrl`.

MemRL ([MemTensor/MemRL](https://github.com/MemTensor/MemRL), MIT) is
non-parametric runtime reinforcement learning. The model weights stay frozen,
and the agent learns by changing what it retrieves into context. Each memory is
an Intent-Experience-Utility triplet stored in SQLite at
`$HERMES_HOME/memrl/memrl.db`:
`(id, intent, experience, embedding, q_value, usage_count, created_at, updated_at)`.

```sh
docker build -f docker/hermes-memrl/Dockerfile -t <tag> .   # from the repo root
```

The build runs the plugin tests against the real `agent.memory_provider`. It
then checks that `load_memory_provider("memrl")` finds the plugin and reports it
available. The embedding model `BAAI/bge-small-en-v1.5` is downloaded at build
time, and the image sets `HF_HUB_OFFLINE=1`, so the plugin works with
`--network none`.

## Algorithm

| Step | Where | What |
| --- | --- | --- |
| Phase A | `prefetch` | Compare the query with every stored intent embedding by cosine similarity; keep those `>= delta`, at most `k1` |
| Phase B | `prefetch` | Z-score similarity and Q over the candidates (Q z-score clamped to ±3). Score = `(1-λ)·ẑ_sim + λ·ẑ_Q`; inject the top `k2` |
| Utility update | commit | Session reward `r ∈ [-1, 1]`. For each injected memory, `Q ← Q + α(r − Q)` |
| New triplet | commit | `r ≥ 0` stores the session as an experience with `Q = 0`. `r < 0` stores a `[PATTERN TO AVOID]` reflection with `Q = 0.5`, so it is retrieved immediately |

**Intent.** When the task prompt contains a `<question>…</question>` block, as
SWE-Atlas QA prompts do, the intent is only that block. Otherwise it is the whole
prompt. The same text is both the memory's key and the search query.

This departs from MemRL, which embeds each benchmark's task text as given. For
SWE-Atlas that text is half shared wrapper, which makes every task look alike.
Set `MEMRL_INTENT_TAG=` (empty) to use the whole prompt, as MemRL does.

**Experience.** This follows MemRL's `proceduralization` build step:
- On success, the task model writes a 3–5 step high-level script from the
  trajectory, using MemRL's `generate_script` prompt. The memory stores
  `Task:` + `SCRIPT:` + `TRAJECTORY:`.
- On failure, the model writes a reflection, using MemRL's `_generate_reflection`
  prompt. The memory stores `[PATTERN TO AVOID]` + `TASK REFLECTION:` +
  `What went wrong:` + `Failed approach:`.

The trajectory, including tool calls and results, is read from Hermes's own
session store (`$HERMES_HOME/state.db`). In one-shot mode the provider otherwise
sees only the final answer. The model sees up to 60,000 characters of it, with
each message capped at 2,000. The stored copy is capped at 6,000 characters, with
each message capped at 400, so that injecting three memories doesn't flood the
context. When a trajectory is too long, its middle is dropped.

The model call goes to the `model` section of Hermes's `config.yaml`, which is
the same OpenAI-compatible endpoint and credential the agent uses. If the call
fails, the memory is still stored, just without the script or reflection.

The reward comes from the `memrl_feedback` tool, which the system prompt asks
the agent to call before its final answer. If the agent doesn't call it, a
heuristic is used:
- With a transcript, the heuristic is `1 − 2·(failed tool results / classified tool results)`.
- Without one, it looks only at the final answer: an empty or give-up answer scores −0.5, anything else scores +0.25.

The commit runs once per session. It is triggered by the first of
`on_session_end`, `shutdown`, or an `atexit` hook. The `atexit` hook exists
because Hermes one-shot (`hermes -z`) exits without calling the other two.

Pure retrieval and update math lives in `memrl/retrieval.py`, ported from
MemRL's `MemoryService.retrieve_query` and `QValueUpdater.update`. MemRL itself
depends on MemOS, so it isn't imported. One deliberate difference: similarity
is z-scored over the candidate set, not with MemRL's fixed corpus statistics.

Background writes use `spawn_context_thread`, `RecallStatus` and
`is_trivial_prompt` from `agent.memory_provider` when Hermes provides them.
Neither this Hermes version nor MemRL ships them, so `memrl/_compat.py`
supplies small local versions.

## Configuration

Enable the provider with `memory.provider: memrl` in Hermes `config.yaml`.
Tuning comes from environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MEMRL_DELTA` | 0.38 | Phase A cosine threshold |
| `MEMRL_K1` | 15 | Phase A candidate cap |
| `MEMRL_K2` | 3 | Memories injected per turn |
| `MEMRL_LAM` | 0.5 | Phase B weight on Q |
| `MEMRL_ALPHA` | 0.3 | EMA step size |
| `MEMRL_EPSILON` | 0.0 | MemRL ε-greedy exploration |
| `MEMRL_Q_INIT` | 0.0 | Q of a new experience |
| `MEMRL_Q_REFLECTION` | 0.5 | Q of a new failure reflection |
| `MEMRL_INTENT_TAG` | `question` | Keep only `<tag>…</tag>` as the intent; empty uses the whole prompt |

With `bge-small-en-v1.5`, unrelated task descriptions still score about 0.4–0.6
cosine against each other (paraphrases score about 0.99). The default
`delta = 0.38` therefore filters very little on its own, and Phase B does most
of the selection. Raise `MEMRL_DELTA` (around 0.7) for stricter topical
matching.

## Tests

```sh
uv run --no-project --with numpy --with pytest --with pyyaml pytest docker/hermes-memrl/tests -q
```

Outside the image, `tests/conftest.py` installs a minimal stand-in for
`agent.memory_provider` and a deterministic fake embedder. The model is replaced
by a fake chat function.

## Use in ARIES

Build and tag the image that `configs/versions-memrl.json` pins, then run a
profile that sets `harness.memrl.enabled`:

```sh
docker build -f docker/hermes-memrl/Dockerfile -t aries/hermes-memrl:v2026.5.29.2-memrl1 .
./bin/aries profiles/hermes-tb2-fix-git-memrl-deepseek.json
```

The Hermes harness keeps one store per run at `<run>/memrl/memrl.db`:
- It stages the store into each task's container.
- After the one-shot exits, it copies the store back. Each task's copy is also
  retained as `harness/memrl/memrl.db`.

The hand-off is sequential, so the profile must use `execution.concurrency` 1.
In one-shot mode the heuristic sees only the final answer. The reward is
therefore meaningful mainly when the agent calls `memrl_feedback`.
