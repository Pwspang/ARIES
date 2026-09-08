#!/bin/sh
# Builds ARIES and runs the 3-arm amem repo-scope study's smoke test (4
# tasks/arm, 12 runs total) — validates the whole pipeline before spending
# the pilot's budget. See the study plan for what to check in the output.
# Requires DEEPSEEK_API_KEY and SGLANG_API_KEY set (e.g. via .env) and a
# reachable sglang server per the profiles' model.base_url.
set -eu
cd "$(dirname "$0")/.."
go build -o bin/aries ./cmd/aries
./bin/aries profiles/openclaw-sweatlasqa-smoke4-sglang.json &
./bin/aries profiles/openclaw-sweatlasqa-smoke4-amem-task-sglang.json &
./bin/aries profiles/openclaw-sweatlasqa-smoke4-amem-repo-sglang.json &
wait
