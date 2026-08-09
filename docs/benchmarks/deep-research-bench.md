## Deep Research Bench

`profiles/openclaw-drb-smoke1-deepseek.json` runs the checked-in Deep Research
Bench profile instead of Terminal-Bench 2 (`profiles/openclaw-drb-smoke3-deepseek.json`
and `profiles/hermes-drb-smoke1-deepseek.json` select a larger task subset and
the Hermes harness respectively). Unlike Terminal-Bench 2, its tasks have no
Docker storage quota (`storage_mb` is unset), so it does not require an
XFS-with-`pquota` container storage backend.

Deep Research Bench tasks are open-ended web research: the profile's
`benchmark.environment.image` must have outbound network access and any
research tooling (browser/search CLI) the agent needs preinstalled — ARIES
does not build or provide that image. Grading is by an LLM judge rather than a
deterministic verifier script, so the profile requires a top-level `judge`
block naming a separate model. The checked-in profile points the judge at the
same DeepSeek endpoint as the harness model:

```json
"judge": {
  "provider": "deepseek",
  "base_url": "https://api.deepseek.com",
  "api_key_env": "DEEPSEEK_API_KEY",
  "model": "deepseek-v4-flash"
}
```

The judge is a separate model call from the harness's own, but since both
point at DeepSeek here, one credential covers both:

```sh
export DEEPSEEK_API_KEY=...
./bin/aries profiles/openclaw-drb-smoke1-deepseek.json
```

`judge.api_key_env` does not have to match `model.api_key_env` — point the
judge at a different provider (e.g. OpenAI) by changing its `provider`,
`base_url`, `api_key_env`, and `model` fields and exporting that provider's
key instead.

The agent is instructed to write its final report to a fixed in-container path
before finishing; ARIES downloads it after the harness stops and the bridge is
revoked, then grades it against the pinned dataset's reference report using
the RACE algorithm across four dimensions (comprehensiveness, insight,
instruction following, readability). Judge artifacts land in
`runs/<run>/<task_id>/evaluation/{report.md,judge_prompt.txt,judge_response.json}`;
`run-result.json`'s `evaluation.score` is RACE's overall ratio
(`target/(target+reference)`) scaled to `[0,1]`, and `evaluation.reward` is `1`
only above the configured pass threshold (default 50). Every evaluated task
makes a paid LLM-judge API call in addition to the harness model call — using
a cheaper judge model is recommended for large task counts.

### FACT citation checking (optional)

A top-level `fact` block additionally grades citation trustworthiness with the
FACT metric: it extracts claim/citation pairs from the report, deduplicates
them, and validates each cited URL's content (fetched through the Jina AI
Reader API) against its claim, using its own (typically cheaper) judge model.
`fact.jina_api_key_env` names the host environment variable holding the Jina
key; FACT has no partial-enable state — a profile must set both the model
fields and `jina_api_key_env`, or omit the whole block (or leave it unset) to
skip FACT entirely, at zero extra cost. FACT is purely additive: its result
never affects `evaluation.score`, `evaluation.reward`, or task status, which
come from RACE alone. Its artifacts, when configured, land alongside the RACE
ones as `fact_report.json` (success) or `fact_error.txt` (failure) — a failed
FACT run does not fail the task. See the checked-in DRB profiles for a working
example.

### Web search and fetch (optional)

Deep Research Bench tasks need the agent to search and read live web pages
from inside the sandbox. Both harnesses support this through
`harness.web_search.enabled: true`, using the DRB task sandbox's built-in
SearXNG instance as the search backend:

- **OpenClaw** needs no further configuration; its `web_fetch` tool needs no
  separate extraction backend.
- **Hermes** additionally accepts `harness.web_search.extract_api_key_env`,
  naming the host environment variable holding a Tavily API key. Without it,
  Hermes's `web_search` tool still works, but `web_extract` (reading a page's
  content) has no backend and fails. Export the key before the run, e.g.
  `export TAVILY_API_KEY=...`, matching the name given in the profile.

The task prompt itself nudges the agent to call `web_fetch`/`web_extract`
rather than reimplement page retrieval with `curl`/`wget`/a custom parser over
the terminal tool, since models don't reliably prefer the dedicated tool on
their own even when it's in their function-calling schema.

### Disabling OpenClaw subagents

`harness.subagents.enabled: false` (OpenClaw only) turns off OpenClaw's
built-in ability to spawn nested agent sessions. The checked-in DRB profiles
set this explicitly, since subagent spawning is not useful for this
benchmark's single-report task shape and adds uncontrolled cost.