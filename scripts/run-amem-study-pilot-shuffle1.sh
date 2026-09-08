#!/bin/sh
# Runs replicate 1 of the amem repo-scope study's continual-learning check:
# same 30 tasks as pilot30 (simple-login/app + paperless-ngx/paperless-ngx,
# 15 each), same 3 arms, but with each repo's 15 tasks in a fixed, full
# derangement of the original pilot30 order (no task keeps its original
# slot). This decorrelates task identity from position-within-repo, so a
# task's score can be compared across replicates at different positions
# instead of only at the one position pilot30 fixed it to. See the study
# plan for the analysis this replicate is for.
# Requires DEEPSEEK_API_KEY and SGLANG_API_KEY set (e.g. via .env) and a
# reachable sglang server per the profiles' model.base_url.
set -eu
cd "$(dirname "$0")/.."
go build -o bin/aries ./cmd/aries
./bin/aries profiles/openclaw-sweatlasqa-pilot30-shuffle1-sglang.json &
./bin/aries profiles/openclaw-sweatlasqa-pilot30-shuffle1-amem-task-sglang.json &
./bin/aries profiles/openclaw-sweatlasqa-pilot30-shuffle1-amem-repo-sglang.json &
wait
