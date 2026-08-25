# Pre-bakes the lossless-claw context-management plugin
# (github.com/Martian-Engineering/lossless-claw, npm package
# "@martian-engineering/lossless-claw") into the upstream OpenClaw image, so
# per-task OpenClaw containers (recreated with no persistent volume on every
# task attempt — see pkg/harness/openclaw/harness.go) never have to hit the
# npm registry at task time. The plugin is installed via OpenClaw's own plugin
# manager (`openclaw plugins install`), not `npm install -g` — this is a real
# OpenClaw CLI plugin, not an MCP stdio server, so it must be installed the
# way OpenClaw's plugin loader expects (see pkg/harness/openclaw/config.go's
# "lossless-claw" plugins.entries wiring).
#
# Unlike docker/openclaw-amem, no sidecar/proxy tooling (socat) and no
# embedding-model pre-download are needed: lossless-claw persists to a local
# SQLite file inside the container's own state directory, with no vector DB
# or ONNX model involved.
#
# The plugin's own docs (docs/configuration.md, confirmed against the
# 0.15.4 package) state it supports OpenClaw 2026.5.28 through 2026.7.1 on
# its 0.x release line — matching the same base image pin
# docker/openclaw-amem uses, so both context-engine experiments run on an
# identical OpenClaw base for a fair comparison.
#
# @openclaw/tavily-plugin is ALSO pre-baked here, for a reason unrelated to
# lossless-claw itself: neither tavily nor any other web-search plugin ships
# with the base OpenClaw image, and this harness's isolated per-task Docker
# network makes a runtime `npm install` (OpenClaw's normal auto-install
# fallback for a configured-but-missing plugin) unreliable — see
# docker/openclaw-amem/Dockerfile's comment for the full failure mode this
# was discovered from. Every profile pinning this image
# (configs/versions-lossless-claw.json) uses tavily as its web_search
# provider today; a profile switching to a different provider (firecrawl,
# searxng) under this same image would need that plugin pre-baked here too.
FROM ghcr.io/openclaw/openclaw:2026.7.1
USER node
RUN openclaw plugins install @martian-engineering/lossless-claw@0.15.4 --pin \
    && openclaw plugins install @openclaw/tavily-plugin --pin
