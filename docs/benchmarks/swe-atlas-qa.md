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
(`configs/sglang/qwen3-8b-local.yaml`), while keeping DeepSeek as the judge —
the runtime backend that serves the agent and the model that grades its
answer are independent choices; nothing requires them to match.
`profiles/openclaw-sweatlasqa-subset20-deepseek.json` runs a fixed 20-task
subset of `data/qa` with DeepSeek as both the harness model and the judge, at
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

- `profiles/openclaw-sweatlasqa-subset20-amem-deepseek.json` — plugin on.
- `profiles/openclaw-sweatlasqa-subset20-deepseek.json` — plugin off.

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
