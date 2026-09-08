## SWE-Atlas QA

[SWE-Atlas](https://github.com/scaleapi/SWE-Atlas) (Scale AI) has three
tracks — Codebase QA, Test-Writing, and Refactoring — with different task and
grading shapes. ARIES implements only the QA track (`data/qa`, "Codebase
Q&A"); Test-Writing and Refactoring are out of scope.

`profiles/openclaw-sweatlasqa-smoke1-deepseek.json` runs the checked-in
SWE-Atlas QA profile against one real task
(`task-6905333b74f22949d97ba998`, a codebase-onboarding question about
Automattic's `wp-calypso`), with DeepSeek as both the harness model and the
judge. `profiles/openclaw-sweatlasqa-smoke1-sglang.json` runs the same task
against an external SGLang-served harness model instead
(`configs/sglang/qwen3.6-35b-local.yaml`), while keeping DeepSeek as the
judge — the runtime backend that serves the agent and the model that grades
its answer are independent choices; nothing requires them to match. The
`subset20` pair described below runs a fixed 20-task subset of `data/qa` at
`execution.concurrency: 3`.

### Task shape

Each task is a directory under the pinned checkout containing `task.toml`
(same `schema_version = "1.1"` shape as Terminal-Bench 2's task files — a
pre-built `docker_image`, resource limits, agent/verifier timeouts),
`instruction.md` (the codebase question, read and passed to the agent
verbatim — unlike Deep Research Bench, ARIES adds no prompt wrapper), and a
private `tests/` directory (`rubrics.json`, `system_prompt.txt`,
`user_prompt_template.txt`, optional `prompt.txt`) that ARIES reads directly
from the host checkout — none of it is ever uploaded into the sandbox.

The agent is expected to write its final answer, wrapped in
`<<FINAL_ANSWER>>` tags, to `/logs/agent/answer.txt` inside the sandbox —
`instruction.md` itself instructs this, so ARIES does not need to augment the
prompt the way Deep Research Bench does for its report path. `PrepareSandbox`
confirms this path starts absent before the harness gets bridge access, and
`Evaluate` downloads it once, after both isolation gates — the only sandbox
material grading ever touches.

### Grading

Unlike Terminal-Bench 2's deterministic pass/fail verifier, grading is by an
LLM judge against a per-task rubric (`tests/rubrics.json`), entirely
host-side — like Deep Research Bench, no code runs inside the sandbox during
evaluation. The vendored dataset's own verifier (`tests/test.sh` running
`tests/evaluate_answer.py` inside the sandbox) is not used at all; ARIES
instead ports its rubric-scoring logic directly into Go
(`pkg/benchmark/sweatlas/rubrics.go`), calling the judge over HTTP from the
host process. For each rubric, the downloaded answer and the rubric's title
(stripped of any numeric prefix like `"1.1: "`) are sent to the judge model,
whose YES/NO response (tolerating a few upstream response-format quirks) is
normalized and, for rubrics annotated "negative", flipped. The results are
aggregated exactly like the upstream script: `reward = 1` only if every
scored "must have" rubric scored 1, and `agg_score` is the mean over all
scored rubrics (any importance) — both are written to
`reward.txt`/`evaluation_results.json` in the run's output directory (not the
sandbox), and `evaluation.score` is always finite and in `[0, 1]` by
construction.

A `benchmark.judge` block is **required** (unlike Deep Research Bench, where
it is optional and falls back to the profile's own model) — judge-graded
rubric scoring is this benchmark's entire output, so there is no sensible
default and no way to disable grading; `judge.enabled` must not be set at
all for this type.

```json
"benchmark": {
  "type": "sweatlasqa",
  "root": ".cache/swe-atlas-qa",
  "tasks": ["task-6905333b74f22949d97ba998"],
  "judge": {
    "provider": "deepseek",
    "base_url": "https://api.deepseek.com",
    "api_key_env": "DEEPSEEK_API_KEY",
    "model": "deepseek-v4-flash"
  }
}
```

task.toml's `[verifier.env]` block declares the keys `EVAL_API_KEY`,
`EVAL_BASE_URL`, and `EVAL_MODEL`, but their *values* in the dataset are
shell-style template placeholders (e.g. `"${OPENAI_API_KEY}"`), meant to be
expanded by Scale's own reference harness ("Harbor") from its own process
environment — they are not literal secrets. Since grading no longer execs
anything inside the sandbox, ARIES never reads or synthesizes a verifier
environment from this block at all; it's decoded and ignored.

`judge.model` must be a string format matching whatever `judge.base_url`
endpoint expects — the sample task's own default
(`anthropic/claude-opus-4-5-20251101`) is an OpenRouter-style composite
string, which is not portable to every endpoint. The checked-in profile
above uses DeepSeek's own API with a plain DeepSeek model ID, which is
internally consistent for that endpoint; picking a different `judge.base_url`
means picking a `judge.model` string that endpoint actually accepts.

### Setup

`configs/versions.json`'s `sweatlasqa` block pins the checkout; run
`./bin/aries setup profiles/openclaw-sweatlasqa-smoke1-deepseek.json` (or
`profiles/openclaw-sweatlasqa-smoke1-sglang.json`, or the equivalent setup
entry point) before the first run.

### Agent memory (amem)

`profiles/openclaw-sweatlasqa-smoke1-amem-deepseek.json` runs the same smoke
task with the `openclaw-amem` memory plugin enabled. Two things have to line
up, and both come from the profile:

- `versions_file` must point at `configs/versions-amem.json`, which pins the
  OpenClaw image with the plugin pre-installed plus the Qdrant image the
  harness starts as a per-task sidecar. The default `configs/versions.json`
  pins neither, so amem cannot run against it.
- `harness.amem.enabled` turns the plugin on. Its `llm_base_url` / `llm_model`
  / `llm_api_key_env` are optional as a group and override the model amem uses
  for its *own* internal calls (note-metadata extraction, merge decisions);
  omitted, it reuses the harness's primary model and key.

Enabling amem also changes the task instruction. `instruction.md` is otherwise
passed to the agent verbatim, but when `harness.amem.enabled` is set,
`cmd/aries/wiring.go` sets `sweatlas.Options.AMEMBootstrap`, which appends a
required memory protocol: `memory_search` before exploring, `memory_add` after
each substantive finding, `memory_search` again before answering, and exactly
one `memory_consolidate` before the answer file is written. Deep Research
Bench established that this mandate is necessary — with the tools merely
available, the agent finished whole tasks without calling any of them — and
that the final `memory_consolidate` is what actually links stored notes into a
graph, since the plugin's own linking pass otherwise only fires on a nightly
timer no benchmark run reaches.

Each task occurrence gets its own Qdrant store, so memory does not carry
across questions within a run. At the end of each occurrence the store is
exported to `runs/<run>/<task>/amem-memory.json` and torn down; non-empty
`links` fields in that export are the signal that `memory_consolidate` was
actually called.

### The subset20 amem A/B

Two profiles run the same 20 tasks so amem can be measured against a control:

- `profiles/openclaw-sweatlasqa-subset20-amem-sglang.json` — plugin on.
- `profiles/openclaw-sweatlasqa-subset20-sglang.json` — plugin off.

Both point at `configs/versions-amem.json`, so both run the *same* OpenClaw
image; the control simply never gets `plugins.entries`, `plugins.slots.memory`
or the memory tool names in its sandbox allow-list, leaving OpenClaw's bundled
`memory-core` in the memory slot. The comparison is therefore "amem vs stock
OpenClaw memory", not "amem vs no memory at all". Task list, model, judge and
concurrency are identical between the two;
`TestSWEAtlasQASubset20ArmsDifferOnlyInAMEM` asserts that, since editing one
profile and forgetting the other would quietly turn the A/B into two unrelated
runs.

The 20 were drawn from the 124 QA tasks by stratifying on category
proportionally to the full set (Architecture 7, Root-cause 6, Onboarding 4,
Security 2, API 1) and then spreading across repositories and languages by
largest deficit, which lands within one task of the proportional share on all
three axes and covers all 11 repositories. The smoke task is deliberately
included as a known-good anchor.

| task | repo | lang | category |
| --- | --- | --- | --- |
| `task-6905333b74f22949d97ba998` | Automattic/wp-calypso | ts | Code Onboarding |
| `task-6905333b74f22949d97ba9ab` | simple-login/app | ts | Architecture & system design |
| `task-6905333b74f22949d97ba9ae` | simple-login/app | ts | Root-cause analysis |
| `task-6905333b74f22949d97ba9bc` | grafana/grafana | ts | Root-cause analysis |
| `task-6905333b74f22949d97ba9cc` | secdev/scapy | python | Architecture & system design |
| `task-6905333b74f22949d97ba9cd` | secdev/scapy | python | Root-cause analysis |
| `task-6905333b74f22949d97ba9d7` | paperless-ngx/paperless-ngx | python | API & library usage / integration |
| `task-6905333b74f22949d97ba9dd` | paperless-ngx/paperless-ngx | python | Architecture & system design |
| `task-6905333b74f22949d97ba9e0` | paperless-ngx/paperless-ngx | python | Root-cause analysis |
| `task-6905333b74f22949d97baa02` | kovidgoyal/kitty | c | Root-cause analysis |
| `task-6905333b74f22949d97baa04` | kovidgoyal/kitty | c | Architecture & system design |
| `task-6905333b74f22949d97baa06` | kovidgoyal/kitty | c | Architecture & system design |
| `task-6905333b74f22949d97baa07` | kovidgoyal/kitty | c | Architecture & system design |
| `task-6905333b74f22949d97baa0f` | trufflesecurity/trufflehog | go | Root-cause analysis |
| `task-6905333b74f22949d97baa14` | foxcpp/maddy | go | Architecture & system design |
| `task-6905333b74f22949d97baa1c` | minio/minio | go | Code Onboarding |
| `task-6905333b74f22949d97baa1d` | simple-login/app | ts | Security |
| `task-6905333b74f22949d97baa21` | foxcpp/maddy | go | Security |
| `task-6905333b74f22949d97baa25` | grafana/k6 | go | Code Onboarding |
| `task-6905333b74f22949d97baa2d` | drakkan/sftpgo | go | Code Onboarding |

Note that covering all 11 repositories means pulling all 11 task images, which
are large (wp-calypso alone is ~14 GB); budget disk before the first run. The
`execution.concurrency` of 3 oversubscribes CPU on a 32-core host, since each
task requests 16 CPUs, but these tasks are dominated by model latency rather
than local compute.

Both arms run the locally served `Qwen/Qwen3.6-35B-A3B-FP8` over SGLang in
`external` mode — the model the endpoint in these profiles actually serves.
Two things stay on DeepSeek deliberately:

- `benchmark.judge`, so the model under test never grades its own answers and
  grading stays comparable across harness backends.
- `harness.amem.llm_*`, which is the model amem uses for its *own* internal
  calls (note-metadata extraction, merge and link decisions). A locally served
  Qwen was found not to reliably emit the raw JSON amem parses there, silently
  defaulting note metadata to empty and every merge decision to "no" — see
  `HarnessAMEMConfig`'s doc comment. Pointing it at DeepSeek keeps the memory
  graph meaningful; leaving it on the primary model would make the amem arm
  look inert for reasons that have nothing to do with whether memory helps.

Both arms therefore need `SGLANG_API_KEY` *and* `DEEPSEEK_API_KEY` set.

### The pilot30 amem study: task-scope vs repo-scope

subset20 only ever answers "amem vs stock OpenClaw memory" (see above), and
every task occurrence in it gets an isolated store regardless of arm — it
cannot say whether memory carrying over *between* task occurrences does
anything, because nothing in that A/B ever shares a store across tasks. The
pilot30 study exists to close that gap, using a true no-memory control and a
second amem arm whose store is shared across every task occurrence hitting
the same repository.

**Hypotheses.**

- **H1 — task-scoped amem is no better than no memory.** Each task occurrence
  gets its own store, exported and torn down at the end of that occurrence
  (documented above); by construction, no note written during one task
  occurrence can ever be read back during another. If that's true, the
  mandatory bootstrap protocol (`memory_search` → explore → `memory_add` →
  `memory_consolidate` → `memory_search`, same as subset20) can only add
  turns and tokens on top of whatever the task itself needs, for no possible
  informational benefit — task-scoped agg_score should be statistically
  indistinguishable from control, and its cost should be higher.
- **H2 — repo-scoped amem beats both control and task-scope.** Scoping the
  store to `(repository, base_commit)` instead of the task occurrence lets a
  later task benefit from an earlier task's notes about the same codebase
  (architecture, conventions, prior findings) — something task-scope cannot
  do at all. If memory transfer across tasks is real, repo-scope's agg_score
  should exceed both other arms.
- **H3 — the benefit compounds with repo history.** If H2's mechanism is
  real, a task's benefit over control should grow with how many prior task
  occurrences against the same repository have already run (and written
  notes) earlier in the same execution — i.e., the repo-scope − control
  agg_score delta should trend upward against a task's position in its
  repository's actual execution order, not stay flat. (This is the one
  hypothesis the pilot30/shuffle1/shuffle2 replicate design in particular was
  built to test — see below.)

**Arms.** Three profiles per replicate, otherwise identical:

- `openclaw-sweatlasqa-<replicate>-sglang.json` — control, `harness.amem.enabled`
  unset/false.
- `openclaw-sweatlasqa-<replicate>-amem-task-sglang.json` — `harness.amem.enabled: true`,
  `scope` unset (defaults to per-task-occurrence, same as subset20's amem arm).
- `openclaw-sweatlasqa-<replicate>-amem-repo-sglang.json` — `harness.amem.enabled: true`,
  `harness.amem.scope: "repo"` — the shared-store path added in `amem_pool.go`
  (`amemRepoScopeKey`, keyed on `sha256(runID + "repo" + repository +
  base_commit)`; `CleanupSharedAMEMRepoScope` tears it down once at the end of
  the run instead of per task).

`TestSWEAtlasQAStudyArmsDifferOnlyInAMEM` (`pkg/config/config_test.go`) is the
three-way analogue of subset20's differ-only-in-AMEM test: it asserts control
has amem disabled, both memory arms have it enabled, `harness.amem` is
identical between the two memory arms once `scope` is normalized out, and
task list, OpenClaw image, model, judge, runtime, and concurrency are
identical across all three arms in a replicate. It also pins
`execution.concurrency: 1` for every arm — unlike subset20's 3 — specifically
so that (a) two same-repo tasks can never write the shared repo-scope store
concurrently, and (b) a task's position within its repository's execution
order is a single well-defined sequence rather than confounded by parallel
scheduling.

Both memory arms point `harness.amem.llm_*` (the model amem uses for its own
internal note-metadata and merge/consolidation calls) at the same local
`Qwen/Qwen3.6-35B-A3B-FP8` SGLang endpoint used as the primary harness model,
rather than DeepSeek as subset20 uses. This conflicts with subset20's own
documented finding that a locally served Qwen doesn't reliably emit the raw
JSON amem parses there, silently defaulting note metadata to empty and merge
decisions to "no" — it is a known deviation from subset20's setup, not a
validated choice for this study, and is worth revisiting before drawing firm
conclusions from pilot30's memory-arm results.

**Replicates.** The same fixed 30-task subset of `data/qa` is run three
times, once per task-execution order: `pilot30` (baseline order),
`pilot30-shuffle1`, `pilot30-shuffle2` — each a distinct full derangement of
the 30 tasks' order with respect to their per-repository position, so that a
task's position within its repo is decorrelated from task identity across
replicates and the position-stratified analysis (H3) can pool three
quasi-independent samples instead of relying on one. A 4-task `smoke4` trio
(same three arms, same invariants) exists to sanity-check the setup cheaply
before committing to a full 30-task run.

**Metrics and analysis.** All analysis is done post-hoc from files already on
disk (`scripts/summarize_sweatlas_arms.py`, extended ad hoc for
position-within-repo and turn/tool-call breakdowns) — no additional
instrumentation is required at run time:

- Primary outcome: `agg_score` from each task's `evaluation/evaluation_results.json`.
- Cost outcomes: input/output tokens and wall-clock runtime from
  `harness-turn-01/telemetry/sessions.json`; assistant-turn count and the
  `exec` vs. `memory_search`/`memory_add`/`memory_consolidate` tool-call
  split from the session transcript
  (`harness-turn-01/telemetry/<session>.jsonl`).
- A task's **position within its repository** is derived from its own
  execution order — the numeric suffix on its run directory
  (`task-<id>-NNN`) — filtered to one repository and re-ranked, giving "the
  Nth task run against this repo so far." This suffix is the actual harness
  execution sequence and was found to disagree with alphabetical task-ID
  order for a handful of tasks per replicate, so alphabetical order is not a
  safe substitute.
- Because all three arms in a replicate run the identical task list (enforced
  by the config test above), the core comparison is a **paired per-task
  delta** (same task, same replicate, arm A score − arm B score), both pooled
  and stratified by position — this cancels out task-difficulty variance that
  a same-position, different-task comparison would not.
- A task can fail to produce `evaluation_results.json` (the rubric judge
  never completed) while still writing a `reward.txt`; every such case
  observed had `reward.txt == 0`, so these are counted as `agg_score = 0`
  rather than dropped from the mean — dropping them was found to bias the
  comparison, since the memory arms fail outright considerably more often
  than control.

**Out of scope for this study.** Whether the effect (if any) replicates on
the full 124-task set, on a different harness model or judge, and whether
capping or pruning the repo-scope store changes its cost profile are open
follow-up questions, not something pilot30 itself tests.
