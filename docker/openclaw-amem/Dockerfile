# Pre-bakes the amem memory plugin (https://amem.owo.lc, npm package
# "openclaw-amem", from github.com/amemhq/amem) into the upstream OpenClaw
# image, so per-task OpenClaw containers (recreated with no persistent volume
# on every task attempt — see pkg/harness/openclaw/harness.go) never have to
# hit the npm registry or re-download the plugin's embedding model at task
# time. Two things are pre-baked for that reason:
#
#  1. The plugin itself, via OpenClaw's own plugin manager (`openclaw plugins
#     install`), not `npm install -g` — this is a real OpenClaw CLI plugin,
#     not an MCP stdio server, so it must be installed the way OpenClaw's
#     plugin loader expects (see pkg/harness/openclaw/config.go's
#     "openclaw-amem" plugins.entries wiring).
#  2. Its ONNX embedding model (Xenova/bge-m3, ~2.27 GB), which the plugin
#     otherwise downloads lazily on first activation — verified by manually
#     booting the gateway with the plugin enabled: it lands under
#     @huggingface/transformers's own default cache dir inside the plugin's
#     node_modules, so simply triggering one real activation during the
#     image build (network egress available at build time, same as the npm
#     install above) bakes it permanently into this layer with no extra
#     runtime config needed.
#
# socat is installed so the launcher (see gatewayLauncherScript in harness.go)
# can front the plugin's Qdrant client with a loopback proxy: the plugin's
# Qdrant base URL is hardcoded to http://localhost:6333 in its own source
# (confirmed by inspecting its bundled dist/index.js — there is no env var or
# config field that overrides it), so the run-scoped shared Qdrant container
# (see amem_qdrant.go) is only reachable by forwarding 127.0.0.1:6333 to its
# real Docker network alias.
#
# @openclaw/tavily-plugin is ALSO pre-baked here, for a reason unrelated to
# amem itself: neither tavily nor any other web-search plugin ships with the
# base OpenClaw image (`openclaw plugins list` on it shows zero search
# plugins — confirmed empirically) — OpenClaw normally fetches a
# plugins.entries-configured-but-missing plugin from the npm registry
# automatically on gateway boot ("Doctor changes: Installed missing
# configured plugin"), and that auto-install DOES work when the container
# has ordinary internet egress (verified against this same base image on a
# normal bridge network). It silently fails, leaving only the
# "plugin is unavailable" warning with no install attempt logged, when the
# container's network can't reach the npm registry — which is what a real
# harness run against this amem-profile's task-private Docker network showed
# (gateway.log: warning present, no "Doctor changes" install line, web_search
# then failing with "no provider is available" for the rest of the task).
# Since every profile pinning this image (configs/versions-amem.json) uses
# tavily as its web_search provider today, pre-baking it here avoids relying
# on that egress. A profile switching to a different provider (firecrawl,
# searxng) under this same image would need that plugin pre-baked here too.
FROM ghcr.io/openclaw/openclaw:2026.7.1
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends socat \
    && rm -rf /var/lib/apt/lists/*
USER node
RUN openclaw plugins install openclaw-amem@2.1.1 --pin \
    && openclaw plugins install @openclaw/tavily-plugin --pin

# amem's own note-metadata-extraction LLM call (llmConstructNote) is hardcoded
# to a 400-token response budget with no config override anywhere in the
# plugin's manifest — confirmed against real runs on two different backend
# models (sglang-served Qwen, then DeepSeek's deepseek-v4-flash): both
# produced JSON responses (keywords/tags/context/category/topics) that don't
# reliably fit in 400 tokens, so the response gets cut off mid-JSON
# ("Unterminated string in JSON") or lands as valid-but-empty JSON, leaving
# every stored note's keywords/tags/context permanently blank regardless of
# which LLM amem is pointed at. This isn't a prompt problem — amem's prompt
# already asks for compact JSON — it's the max_tokens ceiling on the
# response itself. A first pass raised this to 1200, which helped but wasn't
# enough: a real amem-sglang smoke run (2026-08-24, task 61-001) still hit
# the truncated/empty-JSON fallback on 2 of 10 notes at 1200 tokens, both on
# notes with dense numeric market data. Raised further to 2000. Patching the
# one call site that uses it (verified unique in the bundle) raises that
# ceiling; the build fails loudly if a future amem version changes this
# literal, so it can't silently no-op.
RUN set -eu; \
    amem_index="$HOME/.openclaw/npm/projects/openclaw-amem/node_modules/openclaw-amem/dist/index.js"; \
    matches=$(grep -o 'llmCall(prompt, 400)' "$amem_index" | wc -l); \
    if [ "$matches" != "1" ]; then \
      echo "expected exactly one llmCall(prompt, 400) occurrence in $amem_index, found $matches - amem version drift, update this patch" >&2; \
      exit 1; \
    fi; \
    sed -i 's/llmCall(prompt, 400)/llmCall(prompt, 2000)/' "$amem_index"; \
    grep -q 'llmCall(prompt, 2000)' "$amem_index"

# Trigger one real plugin activation to pre-download its embedding model.
# Qdrant reachability doesn't matter here — the plugin logs a non-fatal
# "ensureCollection failed" and continues; only the embedding-model download
# is being baked in. The dummy config never gets written back to the image's
# ~/.openclaw/openclaw.json (it lives at a throwaway path), so it has no
# effect on the real config ARIES renders per task.
RUN set -eu; \
    export ANTHROPIC_API_KEY=bake-time-placeholder; \
    export AMEM_LLM_PROVIDER=anthropic; \
    export AMEM_LLM_API_KEY=bake-time-placeholder; \
    export OPENCLAW_CONFIG_PATH=/tmp/bake-openclaw.json; \
    printf '%s' '{"gateway":{"mode":"local","auth":{"mode":"token","token":"bake"}},"models":{"mode":"merge","providers":{"anthropic":{"baseUrl":"https://api.anthropic.com","apiKey":"bake-time-placeholder","api":"anthropic-messages","models":[{"id":"claude-sonnet-4-6","name":"claude-sonnet-4-6"}]}}},"agents":{"defaults":{"model":{"primary":"anthropic/claude-sonnet-4-6"},"sandbox":{"mode":"off"}}},"tools":{"deny":[]},"plugins":{"allow":["openclaw-amem"],"slots":{"memory":"openclaw-amem"},"entries":{"openclaw-amem":{"enabled":true,"hooks":{"allowConversationAccess":true},"config":{"agentId":"bake"}}}}}' > /tmp/bake-openclaw.json; \
    (openclaw gateway run --port 18790 --auth token --bind loopback > /tmp/bake-gateway.log 2>&1 &) ; \
    cache_dir="$HOME/.openclaw/npm/projects/openclaw-amem/node_modules/@huggingface/transformers/.cache/Xenova/bge-m3/onnx"; \
    for i in $(seq 1 120); do \
      if [ -f "$cache_dir/model.onnx_data" ] && [ -z "$(find "$cache_dir" -name '*.tmp.*' 2>/dev/null)" ]; then break; fi; \
      sleep 5; \
    done; \
    ls -la "$cache_dir" || (echo "amem model pre-bake failed" && cat /tmp/bake-gateway.log && exit 1); \
    test -s "$cache_dir/model.onnx_data"; \
    pkill -f 'openclaw gateway run' || true; \
    sleep 1; \
    rm -f /tmp/bake-openclaw.json /tmp/bake-gateway.log
