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
| Phase A | `prefetch` | Compare the query with every retrieval key's intent embedding by cosine similarity; keep keys `>= delta`, at most `k1` |
| Phase B | `prefetch` | Every memory filed under those keys is a candidate, scored `(1-λ)·ẑ_sim + λ·ẑ_Q`. Similarity uses fixed corpus statistics (`sim_norm_mean`, `sim_norm_std`); Q uses the candidates' own, clamped to ±3. Inject the top `k2`, or with probability `epsilon` a random `k2` |
| Utility update | after the reward | Reward `r ∈ [-1, 1]`. For each injected memory, `Q ← Q + α(r − Q)` |
| New memory | after the reward | `r ≥ 0` stores an experience; `r < 0` stores a `[PATTERN TO AVOID]` reflection. Both start at `Q = 0`. When the task matched a retrieved key with similarity `>= add_similarity`, the memory joins that key; otherwise it starts a new one |

These are MemRL's `retrieve_query`, `QValueUpdater` and `dict_memory` rules.
Defaults come from its BigCodeBench config (`k_retrieve` 5, `topk` 5, ε 0.1,
`q_init_pos`/`q_init_neg` 0, `add_similarity_threshold` 0.9, and an LLM
temperature of 0).

`delta` and the similarity statistics were re-measured for `bge-small-en-v1.5`:
- Over the 124 SWE-Atlas QA questions, pairwise cosine has mean 0.679 and std
  0.057. Same-repository pairs average 0.77, cross-repository pairs 0.67.
- `delta = 0.72` is mean + 0.75 std, the same placement MemRL's thresholds have
  against its own statistics (BigCodeBench 0.38 at mean + 0.7 std, ALFWorld 0.62
  at mean + 0.84 std).
- Other benchmarks should re-measure and set `MEMRL_DELTA`, `MEMRL_SIM_NORM_MEAN`
  and `MEMRL_SIM_NORM_STD`.

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

**Reward.** `MEMRL_REWARD_SOURCE` selects where the reward comes from:

- **`external`**, which ARIES sets. The benchmark's verdict is used, as in MemRL:
  - At exit, the session is parked as pending under `MEMRL_TASK_ID`.
  - Once evaluation finishes, the host writes `{task_id: reward}` to
    `$HERMES_HOME/memrl/rewards.json`. The reward is +1 for full reward, −1
    otherwise, and `null` when the benchmark reached no verdict.
  - The next session applies the recorded rewards before its first recall, then
    builds the memories.
  
  A `null` reward drops the session. The last task of a run stays pending,
  because no later task exists to use it.
- **`agent`**, the default outside ARIES. The agent reports the reward with the
  `memrl_feedback` tool. Otherwise a heuristic is used:
  - With a transcript: `1 − 2·(failed tool results / classified tool results)`.
  - Without one, only the final answer counts: an empty or give-up answer scores
    −0.5, anything else +0.25.

A session is closed once, by the first of `on_session_end`, `shutdown`, or an
`atexit` hook. The `atexit` hook exists because Hermes one-shot (`hermes -z`)
exits without calling the other two.

Pure retrieval and update math lives in `memrl/retrieval.py`, ported from
MemRL's `MemoryService.retrieve_query` and `QValueUpdater.update`. MemRL itself
depends on MemOS, so it isn't imported.

Remaining differences from MemRL:
- The `<question>` intent (above).
- A local embedder, where MemRL uses OpenAI `text-embedding-3-large`.
- The stored trajectory is capped at 6,000 characters.
- Updates happen per task, with no training epochs.

Background writes use `spawn_context_thread`, `RecallStatus` and
`is_trivial_prompt` from `agent.memory_provider` when Hermes provides them.
Neither this Hermes version nor MemRL ships them, so `memrl/_compat.py`
supplies small local versions.

## Configuration

Enable the provider with `memory.provider: memrl` in Hermes `config.yaml`.
Tuning comes from environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MEMRL_DELTA` | 0.72 | Phase A cosine threshold (MemRL `sim_threshold`) |
| `MEMRL_K1` | 5 | Retrieval keys kept by Phase A (`k_retrieve`) |
| `MEMRL_K2` | 5 | Memories injected (`topk`) |
| `MEMRL_LAM` | 0.5 | Phase B weight on Q (`weight_q`; `weight_sim` = 1 − λ) |
| `MEMRL_ALPHA` | 0.3 | EMA step size |
| `MEMRL_EPSILON` | 0.1 | ε-greedy exploration |
| `MEMRL_Q_INIT` | 0.0 | Q of a new experience (`q_init_pos`) |
| `MEMRL_Q_REFLECTION` | 0.0 | Q of a new reflection (`q_init_neg`) |
| `MEMRL_ADD_SIMILARITY` | 0.9 | Join a retrieved key at or above this similarity |
| `MEMRL_SIM_NORM_MEAN` / `MEMRL_SIM_NORM_STD` | 0.679 / 0.0567 | Fixed similarity z-score statistics |
| `MEMRL_SCRIPT_TEMPERATURE` | 0.0 | Temperature of the script call |
| `MEMRL_INTENT_TAG` | `question` | Keep only `<tag>…</tag>` as the intent; empty uses the whole prompt |
| `MEMRL_REWARD_SOURCE` | `agent` | `agent` or `external` (see Reward) |
| `MEMRL_TASK_ID` | none | Task this session belongs to, for `external` rewards |


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

ARIES starts the provider with `MEMRL_REWARD_SOURCE=external` and
`MEMRL_TASK_ID=<task execution ID>`. After each task's evaluation, it records
the verdict in `<run>/memrl/rewards.json`, which is staged into the next task
along with the store. The hand-off is sequential, so the profile must use
`execution.concurrency` 1.
