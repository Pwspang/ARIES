// Offline claude-context index build for one repository mounted at /app.
//
// Builds the collection exactly the way the read-only MCP server will look it
// up: the same env-driven config (createMcpConfig), the same embedding factory,
// the same collection-name override and hybrid mode, and the same default AST
// splitter that index_codebase uses. Exits non-zero unless the build is
// complete and Milvus holds exactly the reported number of chunks.
//
// Env (same names the MCP server reads): EMBEDDING_PROVIDER, EMBEDDING_MODEL,
// OPENAI_API_KEY, OPENAI_BASE_URL, MILVUS_ADDRESS, MILVUS_TOKEN (read/write
// user), CODE_CHUNKS_COLLECTION_NAME_OVERRIDE, HYBRID_MODE.
// Args: [codebasePath=/app] [manifestPath]
import fs from "node:fs";
import { Context, MilvusVectorDatabase } from "@zilliz/claude-context-core";
import { MilvusClient } from "@zilliz/milvus2-sdk-node";
import { createMcpConfig } from "@zilliz/claude-context-mcp/dist/config.js";
import { createEmbeddingInstance } from "@zilliz/claude-context-mcp/dist/embedding.js";
import { createRequestSplitter } from "@zilliz/claude-context-mcp/dist/splitter.js";

const codebasePath = process.argv[2] || "/app";
const manifestPath = process.argv[3];
const started = new Date();

const config = createMcpConfig();
if (!config.collectionNameOverride) {
  throw new Error("CODE_CHUNKS_COLLECTION_NAME_OVERRIDE is required: every repo is indexed at /app");
}
const embedding = createEmbeddingInstance(config);
const dimension = typeof embedding.detectDimension === "function"
  ? await embedding.detectDimension()
  : embedding.getDimension();
const vectorDatabase = new MilvusVectorDatabase({
  address: config.milvusAddress,
  ...(config.milvusToken && { token: config.milvusToken }),
});
const context = new Context({ embedding, vectorDatabase, collectionNameOverride: config.collectionNameOverride });
const collection = context.getCollectionName(codebasePath);
console.log(`[build] ${codebasePath} -> ${collection} (${config.embeddingProvider}/${config.embeddingModel}, dim ${dimension})`);

let lastPct = -1;
const stats = await context.indexCodebase(codebasePath, (p) => {
  const pct = Math.floor(p.percentage ?? 0);
  if (pct !== lastPct && pct % 5 === 0) {
    lastPct = pct;
    console.log(`[build] ${pct}% ${p.phase ?? ""}`);
  }
}, /* forceReindex */ true, [], [], createRequestSplitter("ast"));

const client = new MilvusClient({ address: config.milvusAddress, token: config.milvusToken });
await client.flushSync({ collection_names: [collection] });
await client.loadCollectionSync({ collection_name: collection });
const countResult = await client.query({
  collection_name: collection,
  output_fields: ["count(*)"],
  consistency_level: "Strong",
});
const rows = Number(countResult.data?.[0]?.["count(*)"] ?? -1);

const sample = await context.semanticSearch(codebasePath, "main entry point of the application", 3);

const pkg = (name) => JSON.parse(fs.readFileSync(new URL(`./node_modules/${name}/package.json`, import.meta.url))).version;
const manifest = {
  collection,
  codebase_path: codebasePath,
  status: stats.status,
  indexed_files: stats.indexedFiles,
  total_chunks: stats.totalChunks,
  milvus_rows: rows,
  embedding: { provider: config.embeddingProvider, model: config.embeddingModel, base_url: config.openaiBaseUrl || null, dimension },
  hybrid_mode: process.env.HYBRID_MODE ?? "default(true)",
  splitter: "ast",
  ignore_and_extensions: "claude-context defaults + repo .gitignore/.*ignore",
  versions: {
    "@zilliz/claude-context-core": pkg("@zilliz/claude-context-core"),
    "@zilliz/claude-context-mcp": pkg("@zilliz/claude-context-mcp"),
    "@zilliz/milvus2-sdk-node": pkg("@zilliz/milvus2-sdk-node"),
  },
  sample_hits: sample.map((r) => `${r.relativePath}:${r.startLine}-${r.endLine}`),
  started: started.toISOString(),
  finished: new Date().toISOString(),
};
manifest.accepted = manifest.status === "completed" && rows === stats.totalChunks && stats.totalChunks > 0 && sample.length > 0;
console.log(JSON.stringify(manifest, null, 2));
if (manifestPath) fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
process.exit(manifest.accepted ? 0 : 1);
