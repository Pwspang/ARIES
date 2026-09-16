# Pre-bakes mem0's first-party OpenClaw memory plugin (https://mem0.ai,
# github.com/mem0ai/mem0, integrations/openclaw, npm "@mem0/openclaw-mem0")
# into the upstream OpenClaw image, so per-task OpenClaw containers
# (recreated with no persistent volume on every task attempt — see
# pkg/harness/openclaw/harness.go) never have to hit the npm registry at
# task time. The plugin is installed via OpenClaw's own plugin manager
# (`openclaw plugins install`), not `npm install -g` — this is a real
# OpenClaw CLI plugin, not an MCP stdio server, so it must be installed the
# way OpenClaw's plugin loader expects (see pkg/harness/openclaw/config.go's
# "openclaw-mem0" plugins.entries wiring).
#
# Unlike docker/openclaw-amem, no sidecar/proxy tooling (socat) and no
# embedding-model pre-download are needed here: this profile only wires
# mem0's "open-source" (self-hosted) mode with its default local vector
# store, not Qdrant/PGVector — see mem0PluginConfig's doc comment in
# pkg/harness/openclaw/config.go for why no sidecar container is required.
#
# The plugin's own package.json ("openclaw.compat") declares
# pluginApi/minGatewayVersion >= 2026.4.24, comfortably below this image's
# pinned OpenClaw base — matching the same base image pin
# docker/openclaw-amem and docker/openclaw-lossless-claw use, so all three
# memory-backend experiments run on an identical OpenClaw base for a fair
# comparison.
#
# @openclaw/tavily-plugin is ALSO pre-baked here, for a reason unrelated to
# mem0 itself: neither tavily nor any other web-search plugin ships with the
# base OpenClaw image, and this harness's isolated per-task Docker network
# makes a runtime `npm install` (OpenClaw's normal auto-install fallback for
# a configured-but-missing plugin) unreliable — see
# docker/openclaw-amem/Dockerfile's comment for the full failure mode this
# was discovered from. Every profile pinning this image
# (configs/versions-mem0.json) uses tavily as its web_search provider today;
# a profile switching to a different provider (firecrawl, searxng) under
# this same image would need that plugin pre-baked here too.
FROM ghcr.io/openclaw/openclaw:2026.7.1
USER node
RUN openclaw plugins install @mem0/openclaw-mem0@1.1.0 --pin \
    && openclaw plugins install @openclaw/tavily-plugin --pin
