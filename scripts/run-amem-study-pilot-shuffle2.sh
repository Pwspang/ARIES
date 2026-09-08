#!/bin/sh
# Runs replicate 2 of the amem repo-scope study's continual-learning check:
# same 30 tasks as pilot30 (simple-login/app + paperless-ngx/paperless-ngx,
# 15 each), same 3 arms, but with each repo's 15 tasks in a different fixed,
# full derangement of the original pilot30 order (no task keeps its
# original slot, and this derangement is distinct from shuffle1's) — see
# run-amem-study-pilot-shuffle1.sh for why this decorrelates task identity
# from position-within-repo. Aggregating shuffle1 + shuffle2 (+ the
# original pilot30) gives three independent position assignments per task
# to average over instead of trusting any single task order.
# Requires DEEPSEEK_API_KEY and SGLANG_API_KEY set (e.g. via .env) and a
# reachable sglang server per the profiles' model.base_url.
set -eu
cd "$(dirname "$0")/.."
go build -o bin/aries ./cmd/aries
./bin/aries profiles/openclaw-sweatlasqa-pilot30-shuffle2-sglang.json &
./bin/aries profiles/openclaw-sweatlasqa-pilot30-shuffle2-amem-task-sglang.json &
./bin/aries profiles/openclaw-sweatlasqa-pilot30-shuffle2-amem-repo-sglang.json &
wait
