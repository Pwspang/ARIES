#!/bin/sh
# Builds ARIES and runs the ctxlimit calibration/smoke set (4 tasks/arm, 16
# runs total): the same 3 arms as run-amem-study-smoke.sh (control, amem-task,
# amem-repo) plus amem-global (harness.amem.scope: "global" — one shared
# memory store across both of the 4 tasks' repositories, to check the
# cross-repo transfer plumbing cheaply before the full pilot30-ctxlimit run),
# with model.context_window_tokens set so OpenClaw's own auto-compaction
# should trigger during each task. Inspect each run's
# harness-turn-01/telemetry/sessions.json (contextBudgetStatus.route) and
# telemetry/<session>.jsonl (look for "auto-threshold" compaction
# checkpoints) before committing to the full pilot30-ctxlimit run — if no
# task shows a compaction, context_window_tokens in these profiles (and the
# pilot30-ctxlimit-* profiles) is too loose and must be lowered first.
# Requires DEEPSEEK_API_KEY and SGLANG_API_KEY set (e.g. via .env) and a
# reachable sglang server per the profiles' model.base_url.
set -eu
cd "$(dirname "$0")/.."
go build -o bin/aries ./cmd/aries
./bin/aries profiles/openclaw-sweatlasqa-smoke4-ctxlimit-sglang.json &
./bin/aries profiles/openclaw-sweatlasqa-smoke4-ctxlimit-amem-task-sglang.json &
./bin/aries profiles/openclaw-sweatlasqa-smoke4-ctxlimit-amem-repo-sglang.json &
./bin/aries profiles/openclaw-sweatlasqa-smoke4-ctxlimit-amem-global-sglang.json &
wait
