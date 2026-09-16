#!/bin/sh
# Builds ARIES and runs the ctxlimit study's shuffle2 replicate (30 tasks/arm,
# 60 runs total): same 30 tasks and 96k context_window_tokens as
# run-amem-study-pilot-ctxlimit.sh, but in shuffle2's distinct per-repository
# task order (same order already used by the non-ctxlimit pilot30-shuffle2
# pair) -- a second, independent derangement from shuffle1's, so the
# position-within-repo analysis can pool two quasi-independent samples
# instead of relying on shuffle1 alone.
# Requires DEEPSEEK_API_KEY and SGLANG_API_KEY set (e.g. via .env) and a
# reachable sglang server per the profiles' model.base_url.
set -eu
cd "$(dirname "$0")/.."
go build -o bin/aries ./cmd/aries
./bin/aries profiles/openclaw-sweatlasqa-pilot30-ctxlimit-shuffle2-sglang.json &
./bin/aries profiles/openclaw-sweatlasqa-pilot30-ctxlimit-shuffle2-amem-repo-sglang.json &
wait
