# SWE-Atlas QA with a CodeRAG tool

This experiment measures how much of an agent's [SWE-Atlas QA](swe-atlas-qa.md)
tool use is code retrieval when a code RAG is available. The CodeRAG is
[claude-context](https://github.com/zilliztech/claude-context). ARIES code is
unchanged. Each run's profile adds the CodeRAG as a URL server under
`harness.mcp_servers` (see [MCP configuration](../implementation/harnesses.md#model-context-protocol-mcp)).
The tooling is in [`scripts/sweatlas-coderag/`](../../scripts/sweatlas-coderag/).

Giving the agent a tool, and telling it to use the tool, changes the
experimental method, so record the CodeRAG runs as such. Run each repository's
baseline profile too, so the two can be compared.

Neither OpenClaw nor Hermes puts an MCP server's `instructions` into the prompt.
Both show the model only each tool's description; OpenClaw also caps each
description at 1,200 characters. In smoke runs with claude-context's default
description, agents never called the tool. So by default the `coderag`
profiles set [`benchmark.instruction_append`](swe-atlas-qa.md#harness-prompt-append)
to require the tool for locating code:

> A semantic code search tool, search_code from the coderag MCP server, indexes
> the repository at /app. You MUST use it (path "/app" plus a natural-language
> query) to locate code instead of grep, find or ls. Use shell commands only to
> read specific lines from files the tool returned, or to run code.

Set `CODERAG_INSTRUCTION_APPEND` to change this text, or to an empty value to
omit it. Baseline profiles never set it. The judge never sees the appended
text.

## Design

```
offline   coderag.sh extract ─► .cache/coderag/repos/<slug>          (/app copied out of the pinned image)
          coderag.sh build   ─► Milvus  hybrid_code_chunks_swe_atlas_<slug>_<md5("/app")[:8]>
serve     coderag-serve-<slug>: supergateway (SSE) ─ claude-context-mcp (stdio), Milvus user coderag_serve (read-only)
run       ARIES OpenClaw harness ─► http://<host>:82NN/sse ─► tool coderag__search_code
measure   retrieval_share.py runs/<run-id> …                        (harness telemetry, not bridge logs)
```

- **One server per repository, on the host.** The 124 QA tasks use 11
  repositories. Each task's repository sits at `/app` in its own image. A
  stdio MCP server would run in the harness container, which has no copy of
  `/app`. So each repository gets a server on the host instead. Each
  per-repository profile points at that repository's server.
- **The index is built offline.** `coderag.sh build` indexes a repository
  before any run, using claude-context's own construction code: the MCP
  server's config and embedding factory, plus the AST splitter that
  `index_codebase` uses by default. The build is accepted only if all three hold:
  - indexing reports `completed`;
  - Milvus holds exactly the reported number of chunks (the library drops
    failed non-embedding batches with only a log line, and this check catches
    that);
  - a sample search returns results.

  Each accepted build writes a manifest with the embedding model, the package
  versions, and the image digest. Agent time never includes indexing.
- **Serving is read-only.** Milvus runs with authorization enabled. The
  servers connect as `coderag_serve`, which may only list, load, query and
  search. claude-context always lists `index_codebase` and `clear_index`, and
  the agent can call them, but Milvus refuses the writes. Background sync and
  the trigger watcher are both turned off.
- **Paths match the sandbox.** The build and every server mount the
  repository read-only at `/app`. The collection name hashes that path, and
  `search_code` checks that the path exists on the server's filesystem. So the
  agent can pass the `/app` it sees, and result paths match the sandbox. Each
  repository gets its own collection through `CODE_CHUNKS_COLLECTION_NAME_OVERRIDE`.
- **Parallel runs are safe.** supergateway's SSE mode starts one
  claude-context process per MCP session, so `execution.concurrency` above 1
  is fine.

**Query instruction.** Instruction-tuned embedding models such as Qwen3-Embedding expect each
query in the form `Instruct: <task>\nQuery:<text>`. claude-context doesn't add this.
`CODERAG_QUERY_INSTRUCTION` in `embedding.env` sets the instruction, and
[`image/serve-mcp.mjs`](../../scripts/sweatlas-coderag/image/serve-mcp.mjs) prefixes it to the
search query only. Documents are indexed without it, so changing it needs only
`coderag.sh serve`, not a rebuild. BM25 still sees the raw query.

Build and serve must use the same embedding provider and model, because each
query is embedded when it is made. The servers are reached the same way as
[external SGLang](../configuration.md#external-sglang): the address must be
reachable from both the ARIES host and the harness containers.
`localhost` won't work. The servers bind `0.0.0.0:8201-8211` (8101+ collided with a model endpoint on 8102), while Milvus binds
only to host loopback.

## Steps

Run these from the repository root. State goes in `.cache/coderag`, which is
gitignored; set `CODERAG_STATE` to use a different directory. That state
includes the extracted repositories, the Milvus data, generated passwords,
manifests, and generated profiles.

1. Run `./bin/aries setup` on any SWE-Atlas QA profile so that
   `.cache/swe-atlas-qa` exists.
2. Run `scripts/sweatlas-coderag/coderag.sh repos`, then
   `scripts/sweatlas-coderag/coderag.sh extract`.
3. Copy `scripts/sweatlas-coderag/embedding.env.example` to
   `.cache/coderag/embedding.env` and point it at an OpenAI-compatible
   embedding endpoint.
4. Run `scripts/sweatlas-coderag/coderag.sh up`. This creates the Docker
   network `coderag`, starts Milvus `v2.6.4`, builds the
   `aries-coderag:0.1.15` image, and sets up the two Milvus users. The
   passwords go in `.cache/coderag/secrets.env` with mode 0600.
5. Run `scripts/sweatlas-coderag/coderag.sh build`. It writes
   `.cache/coderag/build/manifest-<slug>.json` for each accepted build.
6. Run `scripts/sweatlas-coderag/coderag.sh serve`, then
   `scripts/sweatlas-coderag/coderag.sh check`. The check runs from a
   container on Docker's default bridge network, the same path a harness
   container takes. It lists the tools, runs `search_code` on `/app`, confirms
   that `clear_index` is refused, and confirms the row count did not change.
7. Run `scripts/sweatlas-coderag/coderag.sh profiles <base-profile>.json`. This
   writes a `coderag` profile (MCP server plus the appended instruction) and a
   `baseline` profile (neither) for each repository into
   `.cache/coderag/profiles/`. Use your own SWE-Atlas QA profile as the base,
   with its model and judge. Then run `./bin/aries setup` and `./bin/aries` on
   each profile.
8. Run `python3 scripts/sweatlas-coderag/retrieval_share.py runs/<run-id> …`.

`coderag.sh ports` lists each repository's port and URL. `coderag.sh down`
stops the servers and Milvus and keeps the index. To expose only
`search_code`, set `CODERAG_TOOLS=search_code`. That changes what the agent
sees, so report it with the results.

## Metric

`retrieval_share.py` reads the OpenClaw session exports under each task's
`harness/telemetry/`. It also reads the older `harness-turn-NN/telemetry/`
layout. In those exports, every `toolCall` part carries the tool's name, and
OpenClaw names MCP tools `<server>__<tool>`. The script has to read the
harness telemetry because MCP calls never pass through the ToolBridge, so
`bridge/tool-calls.jsonl` never shows them.

The **retrieval share** is the number of `search_code` calls divided by the
number of all tool calls. The script reports it per task, then per
repository and overall. Each aggregate gives a micro average
(Σ retrieval / Σ calls) and a macro average (the mean of the per-task shares),
next to the evaluation score. The script counts other claude-context calls
separately, and flags `index_codebase` and `clear_index` as attempts to change
the index. `--json` prints every row and aggregate.
