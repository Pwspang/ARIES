#!/bin/sh
# Builds ARIES and runs the ctxlimit study's shuffle1 replicate (30 tasks/arm,
# 60 runs total): same 30 tasks and 96k context_window_tokens as
# run-amem-study-pilot-ctxlimit.sh, but in shuffle1's distinct per-repository
# task order (same order already used by the non-ctxlimit pilot30-shuffle1
# pair) so a task's position within its repository's execution order is
# decorrelated from task identity — needed to tell "memory's effect changes
# with position in the repo's history" apart from "these particular
# late-order tasks are just harder," per the position-within-repo finding in
# the ctxlimit pilot30 analysis.
# Requires DEEPSEEK_API_KEY and SGLANG_API_KEY set (e.g. via .env) and a
# reachable sglang server per the profiles' model.base_url.
set -eu
cd "$(dirname "$0")/.."
go build -o bin/aries ./cmd/aries
./bin/aries profiles/openclaw-sweatlasqa-pilot30-ctxlimit-shuffle1-sglang.json &
./bin/aries profiles/openclaw-sweatlasqa-pilot30-ctxlimit-shuffle1-amem-repo-sglang.json &
wait
