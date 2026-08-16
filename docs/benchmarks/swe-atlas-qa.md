## SWE-Atlas QA

[SWE-Atlas](https://github.com/scaleapi/SWE-Atlas) (Scale AI) has three
tracks — Codebase QA, Test-Writing, and Refactoring — with different task and
grading shapes. ARIES implements only the QA track (`data/qa`, "Codebase
Q&A"); Test-Writing and Refactoring are out of scope.

`profiles/openclaw-sweatlasqa-smoke1-deepseek.json` runs the checked-in
SWE-Atlas QA profile against one real task
(`task-6905333b74f22949d97ba998`, a codebase-onboarding question about
Automattic's `wp-calypso`).

### Task shape

Each task is a directory under the pinned checkout containing `task.toml`
(same `schema_version = "1.1"` shape as Terminal-Bench 2's task files — a
pre-built `docker_image`, resource limits, agent/verifier timeouts),
`instruction.md` (the codebase question, read and passed to the agent
verbatim — unlike Deep Research Bench, ARIES adds no prompt wrapper), and a
private `tests/` verifier tree injected into the sandbox only after both
isolation gates (harness stopped, bridge revoked), exactly like Terminal-Bench
2's `tests/test.sh`.

The agent is expected to write its final answer, wrapped in
`<<FINAL_ANSWER>>` tags, to `/logs/agent/answer.txt` inside the sandbox —
`instruction.md` itself instructs this, so ARIES does not need to augment the
prompt the way Deep Research Bench does for its report path.

### Grading

Unlike Terminal-Bench 2's deterministic pass/fail verifier, grading is by an
LLM judge against a per-task rubric (`tests/rubrics.json`) — closer to Deep
Research Bench's judge-graded model, but the verifier script itself still
runs inside the sandbox rather than host-side. The injected `tests/test.sh`
runs `tests/evaluate_answer.py`, which reads the agent's answer, scores each
rubric criterion with a judge model call, and writes
`/logs/verifier/reward.txt` (`1` only if every "must have" rubric scored 1)
and `/logs/verifier/evaluation_results.json` (a richer breakdown including
`agg_score`, the fraction of all scored rubrics that passed).
`evaluation.reward` comes from `reward.txt`; `evaluation.score` comes from
`agg_score`, validated to be finite and in `[0, 1]` (rejecting, not clamping,
anything outside that range).

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

**Deviation from Terminal-Bench 2's verifier-injection contract**: task.toml's
`[verifier.env]` block declares the keys `EVAL_API_KEY`, `EVAL_BASE_URL`, and
`EVAL_MODEL`, but their *values* in the dataset are shell-style template
placeholders (e.g. `"${OPENAI_API_KEY}"`), meant to be expanded by Scale's own
reference harness ("Harbor") from its own process environment — they are not
literal secrets. ARIES validates that task.toml declares exactly this key
set (failing loudly if a future dataset revision changes the contract), then
discards the literal values entirely and synthesizes the real verifier
environment fresh from the profile's `judge` config at evaluation time. This
avoids adding a generic `${VAR}`-style templating engine to the codebase for
a one-off need.

`judge.model` must be a string format matching whatever `judge.base_url`
endpoint expects — the sample task's own default
(`anthropic/claude-opus-4-5-20251101`) is an OpenRouter-style composite
string, which is not portable to every endpoint. The checked-in profile
above uses DeepSeek's own API with a plain DeepSeek model ID, which is
internally consistent for that endpoint; picking a different `judge.base_url`
means picking a `judge.model` string that endpoint actually accepts.

### Setup

`configs/versions.json`'s `sweatlasqa` block pins the checkout; run
`./bin/aries setup profiles/openclaw-sweatlasqa-smoke1-deepseek.json` (or the
equivalent setup entry point) before the first run.
