#!/bin/sh
# Builds ARIES and runs the ctxlimit study's pilot (30 tasks/arm, 120 runs
# total) — the same task list, model, and judge as
# run-amem-study-pilot.sh's pilot30 trio, plus a fourth amem-global arm
# (harness.amem.scope: "global" — one shared memory store across the whole
# run, spanning both of pilot30's repositories, to test whether memory
# transfers across repositories rather than just within one — see
# docs/benchmarks/swe-atlas-qa.md), with model.context_window_tokens set so
# OpenClaw's own auto-compaction should trigger routinely during each task
# (pilot30's own unconstrained runs never did — see
# docs/benchmarks/swe-atlas-qa.md). Only run this after
# run-amem-study-smoke-ctxlimit.sh has confirmed compaction actually fires.
# Requires DEEPSEEK_API_KEY and SGLANG_API_KEY set (e.g. via .env) and a
# reachable sglang server per the profiles' model.base_url.
set -eu
cd "$(dirname "$0")/.."
go build -o bin/aries ./cmd/aries
./bin/aries profiles/openclaw-sweatlasqa-pilot30-ctxlimit-amem-global-sglang.json &
wait
