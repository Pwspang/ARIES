#!/bin/sh
# Builds ARIES and runs the ctxlimit study's pilot (30 tasks/arm, 90 runs
# total) — the same task list, model, and judge as
# run-amem-study-pilot.sh's pilot30 trio, with model.context_window_tokens
# set so OpenClaw's own auto-compaction should trigger routinely during each
# task (pilot30's own unconstrained runs never did — see
# docs/benchmarks/swe-atlas-qa.md). Only run this after
# run-amem-study-smoke-ctxlimit.sh has confirmed compaction actually fires.
# Requires DEEPSEEK_API_KEY and SGLANG_API_KEY set (e.g. via .env) and a
# reachable sglang server per the profiles' model.base_url.
set -eu
cd "$(dirname "$0")/.."
go build -o bin/aries ./cmd/aries
./bin/aries profiles/openclaw-sweatlasqa-pilot30-ctxlimit-sglang.json &
./bin/aries profiles/openclaw-sweatlasqa-pilot30-ctxlimit-amem-task-sglang.json &
./bin/aries profiles/openclaw-sweatlasqa-pilot30-ctxlimit-amem-repo-sglang.json &
wait
