# Pre-bakes the amem memory plugin (https://amem.owo.lc, forked from the npm
# package "openclaw-amem" / github.com/amemhq/amem — see
# third_party/amem-fork/ at the repo root for the vendored fork source and
# what's patched) into the upstream OpenClaw image, so per-task OpenClaw
# containers (recreated with no persistent volume on every task attempt —
# see pkg/harness/openclaw/harness.go) never have to hit the npm registry or
# re-download the plugin's models at task time. Three things are pre-baked
# for that reason:
#
#  1. The plugin itself, built from third_party/amem-fork/ in the first
#     stage below and installed via OpenClaw's own plugin manager (`openclaw
#     plugins install <local tarball>`), not `npm install -g` — this is a
#     real OpenClaw CLI plugin, not an MCP stdio server, so it must be
#     installed the way OpenClaw's plugin loader expects (see
#     pkg/harness/openclaw/config.go's "openclaw-amem" plugins.entries
#     wiring).
#  2. Its ONNX embedding model (Xenova/bge-m3, ~2.27 GB), which the plugin
#     otherwise downloads lazily on first activation — verified by manually
#     booting the gateway with the plugin enabled: it lands under
#     @huggingface/transformers's own default cache dir inside the plugin's
#     node_modules, so simply triggering one real activation during the
#     image build (network egress available at build time, same as the npm
#     install above) bakes it permanently into this layer with no extra
#     runtime config needed.
#  3. Its ONNX cross-encoder reranker model (Xenova/ms-marco-MiniLM-L-6-v2,
#     added in this fork — see
#     third_party/amem-fork/packages/amem-core/src/reranker.ts), pre-warmed
#     the same way as the embedding model (see index.ts's loadRerankModel()
#     call alongside its existing ensureCollection() pre-warm).
#
# The fork lives outside this directory (third_party/amem-fork/ at the repo
# root, not docker/openclaw-amem/), so this image must be built with the repo
# root as build context, e.g. from the repo root:
#   docker build -f docker/openclaw-amem/Dockerfile -t <tag> .
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
# ── Builder stage: build our fork of openclaw-amem from vendored source ─────
# third_party/amem-fork/ (repo root) is a pinned vendor copy of
# github.com/amemhq/amem (MIT licensed) at tag openclaw-amem@2.1.1, commit
# e0cb26f, patched for goal-aware embedding and a post-BFS cross-encoder
# reranker (see third_party/amem-fork/packages/amem-core/src/memory.ts and
# reranker.ts) and for the note-metadata-extraction token budget (see
# llm.ts — this used to be a Dockerfile sed patch against the built bundle;
# it is now a direct source change since the source is vendored here).
# openclaw-amem's tsup config aliases `@amemhq/core` straight to amem-core's
# src/, inlining the engine into dist/index.js, so building the plugin alone
# (no native deps involved — @huggingface/transformers, onnxruntime-node stay
# external, resolved at `openclaw plugins install` time in the final stage)
# is a pure-JS esbuild step, portable across the builder's platform.
FROM node:24 AS amem-build
WORKDIR /amem-src
COPY third_party/amem-fork/ ./
RUN corepack enable && corepack prepare pnpm@11.15.1 --activate
RUN pnpm install --frozen-lockfile
# test:unit (not the broader "test" script) on purpose — this repo's
# test/integration suites expect a real Qdrant/network and aren't relevant to
# a from-source build check here.
RUN pnpm --filter @amemhq/core build \
    && pnpm --filter @amemhq/core typecheck \
    && pnpm --filter @amemhq/core test:unit \
    && pnpm --filter openclaw-amem build \
    && pnpm --filter openclaw-amem typecheck \
    && pnpm --filter openclaw-amem test:unit
RUN mkdir -p /tmp/amem-pack && cd packages/openclaw-amem && npm pack --pack-destination /tmp/amem-pack

FROM ghcr.io/openclaw/openclaw:2026.7.1
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends socat \
    && rm -rf /var/lib/apt/lists/*
USER node
COPY --from=amem-build --chown=node:node /tmp/amem-pack/*.tgz /tmp/openclaw-amem.tgz
# npm-pack: (not a bare archive path) so this installs into the same
# ~/.openclaw/npm/projects/<id>/ layout a registry install uses — the layout
# the model pre-bake step below assumes — and so OpenClaw actually resolves
# the plugin's own dependencies (@huggingface/transformers, onnxruntime-node,
# @node-rs/jieba) instead of skipping dependency repair, which is what a bare
# local-archive install does per OpenClaw's own plugin docs.
RUN openclaw plugins install npm-pack:/tmp/openclaw-amem.tgz --pin \
    && openclaw plugins install @openclaw/tavily-plugin --pin \
    && rm -f /tmp/openclaw-amem.tgz

# Trigger one real plugin activation to pre-download its embedding model and
# the cross-encoder reranker model added in this fork.
# Qdrant reachability doesn't matter here — the plugin logs a non-fatal
# "ensureCollection failed" and continues; only the model downloads are being
# baked in. The dummy config never gets written back to the image's
# ~/.openclaw/openclaw.json (it lives at a throwaway path), so it has no
# effect on the real config ARIES renders per task.
RUN set -eu; \
    export ANTHROPIC_API_KEY=bake-time-placeholder; \
    export AMEM_LLM_PROVIDER=anthropic; \
    export AMEM_LLM_API_KEY=bake-time-placeholder; \
    export OPENCLAW_CONFIG_PATH=/tmp/bake-openclaw.json; \
    printf '%s' '{"gateway":{"mode":"local","auth":{"mode":"token","token":"bake"}},"models":{"mode":"merge","providers":{"anthropic":{"baseUrl":"https://api.anthropic.com","apiKey":"bake-time-placeholder","api":"anthropic-messages","models":[{"id":"claude-sonnet-4-6","name":"claude-sonnet-4-6"}]}}},"agents":{"defaults":{"model":{"primary":"anthropic/claude-sonnet-4-6"},"sandbox":{"mode":"off"}}},"tools":{"deny":[]},"plugins":{"allow":["openclaw-amem"],"slots":{"memory":"openclaw-amem"},"entries":{"openclaw-amem":{"enabled":true,"hooks":{"allowConversationAccess":true},"config":{"agentId":"bake"}}}}}' > /tmp/bake-openclaw.json; \
    (openclaw gateway run --port 18790 --auth token --bind loopback > /tmp/bake-gateway.log 2>&1 &) ; \
    embed_cache_dir="$HOME/.openclaw/npm/projects/openclaw-amem/node_modules/@huggingface/transformers/.cache/Xenova/bge-m3/onnx"; \
    rerank_cache_dir="$HOME/.openclaw/npm/projects/openclaw-amem/node_modules/@huggingface/transformers/.cache/Xenova/ms-marco-MiniLM-L-6-v2/onnx"; \
    for i in $(seq 1 120); do \
      if [ -f "$embed_cache_dir/model.onnx_data" ] && [ -z "$(find "$embed_cache_dir" -name '*.tmp.*' 2>/dev/null)" ]; then break; fi; \
      sleep 5; \
    done; \
    ls -la "$embed_cache_dir" || (echo "amem embedding model pre-bake failed" && cat /tmp/bake-gateway.log && exit 1); \
    test -s "$embed_cache_dir/model.onnx_data"; \
    for i in $(seq 1 60); do \
      if [ -n "$(find "$rerank_cache_dir" -name '*.onnx' -size +1k 2>/dev/null)" ] && [ -z "$(find "$rerank_cache_dir" -name '*.tmp.*' 2>/dev/null)" ]; then break; fi; \
      sleep 5; \
    done; \
    ls -la "$rerank_cache_dir" || (echo "amem reranker model pre-bake failed" && cat /tmp/bake-gateway.log && exit 1); \
    test -n "$(find "$rerank_cache_dir" -name '*.onnx' -size +1k 2>/dev/null)"; \
    pkill -f 'openclaw gateway run' || true; \
    sleep 1; \
    rm -f /tmp/bake-openclaw.json /tmp/bake-gateway.log
