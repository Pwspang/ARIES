#!/bin/sh
# Builds ARIES and runs the 3-arm amem repo-scope study's pilot (30
# tasks/arm across simple-login/app + paperless-ngx/paperless-ngx, 90 runs
# total). Only run this after the smoke test (run-amem-study-smoke.sh) has
# passed all its checks — see the study plan.
# Requires DEEPSEEK_API_KEY and SGLANG_API_KEY set (e.g. via .env) and a
# reachable sglang server per the profiles' model.base_url.
set -eu
cd "$(dirname "$0")/.."
go build -o bin/aries ./cmd/aries
./bin/aries profiles/openclaw-sweatlasqa-pilot30-sglang.json &
./bin/aries profiles/openclaw-sweatlasqa-pilot30-amem-task-sglang.json &
./bin/aries profiles/openclaw-sweatlasqa-pilot30-amem-repo-sglang.json &
wait
