#!/bin/sh
# Builds ARIES and runs the amem-enabled Deep Research Bench subset-10 profile.
# Requires DEEPSEEK_API_KEY, JINA_API_KEY, SGLANG_API_KEY, TAVILY_API_KEY set
# (e.g. via .env) and a reachable sglang server per
# profiles/openclaw-drb-subset10-amem-sglang.json's model.base_url.
set -eu
cd "$(dirname "$0")/.."
go build -o bin/aries ./cmd/aries
./bin/aries profiles/openclaw-drb-subset10-amem-sglang.json
