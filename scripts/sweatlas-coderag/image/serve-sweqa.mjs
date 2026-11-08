// SWE-QA-Agent baseline: a read-only, three-tool code-retrieval MCP server.
//
// Mirrors the action space of SWE-QA-Agent (Peng et al., "SWE-QA: Can Language
// Models Answer Repository-level Code Questions?", arXiv 2509.14635v1; ACL 2026
// Findings), which answers repository questions with exactly three retrieval
// actions: GetRepoStructure (a `tree`-style listing), SearchContent (RAG over a
// code-embedding index) and ReadFile. The authors did not release the agent,
// so the tools here follow the paper's descriptions:
//   get_repo_structure  hierarchical file/directory listing of /app
//   search_content      semantic + BM25 hybrid search over the offline
//                       claude-context index (same collection as search_code)
//   read_file           file contents with line numbers, optional line range
//
// Deviations, recorded for the paper: SearchContent uses Qwen3-Embedding-4B
// (best MRR on Agent Retrieval Bench, arXiv 2607.24882) with claude-context's
// AST chunks and hybrid BM25 instead of voyage-code-3; the structure listing is
// depth-limited and skips VCS/dependency directories so it fits in context.
//
// Env: the claude-context variables used by build-index.mjs (EMBEDDING_*,
// OPENAI_*, MILVUS_ADDRESS, MILVUS_TOKEN = read-only user,
// CODE_CHUNKS_COLLECTION_NAME_OVERRIDE), CODERAG_QUERY_INSTRUCTION, and
// SWEQA_ROOT (default /app).
import fs from "node:fs";
import path from "node:path";
import { z } from "zod";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { Context, MilvusVectorDatabase } from "@zilliz/claude-context-core";
import { createMcpConfig } from "@zilliz/claude-context-mcp/dist/config.js";
import { createEmbeddingInstance } from "@zilliz/claude-context-mcp/dist/embedding.js";
import { applyQueryInstruction } from "./query-instruction.mjs";

const ROOT = fs.realpathSync(process.env.SWEQA_ROOT || "/app");
// Directories `tree` would list but that are VCS metadata or installed
// dependencies, not repository code. Dot-entries are hidden as `tree` does.
const SKIP_DIRS = new Set(["node_modules", "__pycache__", "site-packages", "bower_components"]);
const TREE_MAX_DEPTH = 3, TREE_MAX_ENTRIES = 1500;
const READ_MAX_LINES = 2000, READ_MAX_CHARS = 200_000;
const SEARCH_DEFAULT_K = 10, SEARCH_MAX_K = 30, SNIPPET_MAX_CHARS = 5000;

// Logs go to stderr: stdout carries the MCP stdio protocol.
const log = (...a) => console.error("[sweqa]", ...a);

applyQueryInstruction();
const config = createMcpConfig();
if (!config.collectionNameOverride) throw new Error("CODE_CHUNKS_COLLECTION_NAME_OVERRIDE is required");
const context = new Context({
  embedding: createEmbeddingInstance(config),
  vectorDatabase: new MilvusVectorDatabase({
    address: config.milvusAddress,
    ...(config.milvusToken && { token: config.milvusToken }),
  }),
  collectionNameOverride: config.collectionNameOverride,
});

const ok = (text) => ({ content: [{ type: "text", text }] });
const fail = (text) => ({ content: [{ type: "text", text: `Error: ${text}` }], isError: true });

// Resolves an agent-supplied path (absolute under the root, or relative to it)
// and refuses anything that escapes the root, including through symlinks.
function resolveInRoot(p) {
  const candidate = path.isAbsolute(p) ? p : path.join(ROOT, p);
  let real;
  try { real = fs.realpathSync(candidate); } catch { return { error: `no such file or directory: ${p}` }; }
  if (real !== ROOT && !real.startsWith(ROOT + path.sep)) return { error: `path is outside ${ROOT}: ${p}` };
  return { real };
}
const display = (real) => (real === ROOT ? ROOT : path.join(ROOT, path.relative(ROOT, real)));

function listDir(dir) {
  return fs.readdirSync(dir, { withFileTypes: true })
    .filter((e) => !e.name.startsWith(".") && !(e.isDirectory() && SKIP_DIRS.has(e.name)))
    .sort((a, b) => (b.isDirectory() - a.isDirectory()) || a.name.localeCompare(b.name));
}

function tree(start, maxDepth) {
  const lines = [`${display(start)}/`];
  let shown = 0, truncated = false, dirs = 0, files = 0;
  const walk = (dir, depth, indent) => {
    let entries;
    try { entries = listDir(dir); } catch { return; }
    for (const e of entries) {
      if (shown >= TREE_MAX_ENTRIES) { truncated = true; return; }
      const full = path.join(dir, e.name);
      if (e.isDirectory()) {
        dirs++; shown++;
        if (depth < maxDepth) {
          lines.push(`${indent}${e.name}/`);
          walk(full, depth + 1, indent + "  ");
        } else {
          let n = 0; try { n = listDir(full).length; } catch { /* unreadable */ }
          lines.push(`${indent}${e.name}/ (${n} entries)`);
        }
      } else {
        files++; shown++;
        lines.push(`${indent}${e.name}`);
      }
    }
  };
  walk(start, 1, "  ");
  lines.push("", `${dirs} directories, ${files} files shown (depth ${maxDepth}).`);
  if (truncated) lines.push(`Listing truncated at ${TREE_MAX_ENTRIES} entries; pass a subdirectory as path to see more.`);
  return lines.join("\n");
}

// claude-context's AST chunks can report line numbers a few lines off (seen:
// startLine 242 for a chunk whose first line is line 240). Re-anchor each hit
// on the file by finding the snippet's first and last non-blank lines nearest
// the reported positions, so read_file ranges built from a hit are correct.
function anchor(hit) {
  let lines;
  try { lines = fs.readFileSync(path.join(ROOT, hit.relativePath), "utf8").split("\n"); } catch { return hit; }
  const body = hit.content.split("\n").map((l) => l.trimEnd()).filter((l) => l.trim() !== "");
  if (body.length === 0) return hit;
  const nearest = (needle, around) => {
    for (let d = 0; d <= 200; d++) {
      for (const i of [around - 1 - d, around - 1 + d]) {
        if (i >= 0 && i < lines.length && lines[i].trimEnd() === needle) return i + 1;
      }
    }
    return null;
  };
  const start = nearest(body[0], hit.startLine);
  const end = nearest(body[body.length - 1], hit.endLine);
  return {
    ...hit,
    startLine: start ?? hit.startLine,
    endLine: end && end >= (start ?? hit.startLine) ? end : hit.endLine,
  };
}

const server = new McpServer({ name: "sweqa", version: "1.0.0" });

server.tool(
  "get_repo_structure",
  `Show the hierarchical file and directory structure of the repository at ${ROOT} (like the tree command). ` +
    `Directories at the depth limit show their entry count; pass a subdirectory as path to expand it.`,
  {
    path: z.string().optional().describe(`Directory to list, absolute or relative to ${ROOT} (default: ${ROOT})`),
    max_depth: z.number().int().min(1).max(10).optional().describe(`How many levels to expand (default ${TREE_MAX_DEPTH})`),
  },
  async ({ path: p = ROOT, max_depth = TREE_MAX_DEPTH }) => {
    const r = resolveInRoot(p);
    if (r.error) return fail(r.error);
    if (!fs.statSync(r.real).isDirectory()) return fail(`not a directory: ${p}`);
    return ok(tree(r.real, max_depth));
  },
);

server.tool(
  "search_content",
  `Search the repository at ${ROOT} with a natural-language or code query. Retrieval-augmented search over a ` +
    `code-embedding index (semantic + keyword); returns the most relevant code snippets with file paths and line ranges.`,
  {
    query: z.string().min(1).describe("What to look for, in natural language or code terms"),
    limit: z.number().int().min(1).max(SEARCH_MAX_K).optional().describe(`Number of snippets (default ${SEARCH_DEFAULT_K})`),
  },
  async ({ query, limit = SEARCH_DEFAULT_K }) => {
    try {
      const hits = (await context.semanticSearch(ROOT, query, limit, 0.3)).map(anchor);
      if (hits.length === 0) return ok(`No results for: ${query}`);
      const body = hits.map((h, i) => {
        const snippet = h.content.length > SNIPPET_MAX_CHARS ? h.content.slice(0, SNIPPET_MAX_CHARS) + "\n... (truncated)" : h.content;
        return `${i + 1}. ${path.join(ROOT, h.relativePath)}:${h.startLine}-${h.endLine} (${h.language})\n\`\`\`${h.language}\n${snippet}\n\`\`\``;
      }).join("\n\n");
      return ok(`${hits.length} results for: ${query}\n\n${body}`);
    } catch (e) {
      log("search failed:", e);
      return fail(`search failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  },
);

server.tool(
  "read_file",
  `Read a file from the repository at ${ROOT}, with line numbers. Give start_line/end_line to read a range ` +
    `(at most ${READ_MAX_LINES} lines per call).`,
  {
    path: z.string().min(1).describe(`File path, absolute or relative to ${ROOT}`),
    start_line: z.number().int().min(1).optional().describe("First line to return (1-based, default 1)"),
    end_line: z.number().int().min(1).optional().describe("Last line to return (inclusive)"),
  },
  async ({ path: p, start_line = 1, end_line }) => {
    const r = resolveInRoot(p);
    if (r.error) return fail(r.error);
    if (!fs.statSync(r.real).isFile()) return fail(`not a regular file: ${p}`);
    const buf = fs.readFileSync(r.real);
    if (buf.subarray(0, 8192).includes(0)) return fail(`binary file: ${p}`);
    const lines = buf.toString("utf8").split("\n");
    if (lines.length > 1 && lines[lines.length - 1] === "") lines.pop();
    const total = lines.length;
    if (start_line > total) return fail(`start_line ${start_line} is past the end of the file (${total} lines)`);
    let last = Math.min(end_line ?? total, total, start_line + READ_MAX_LINES - 1);
    if (last < start_line) return fail("end_line must not be before start_line");
    const width = String(last).length;
    const out = [];
    let chars = 0;
    for (let n = start_line; n <= last; n++) {
      const line = `${String(n).padStart(width)}\t${lines[n - 1]}`;
      if (chars + line.length > READ_MAX_CHARS) {
        if (out.length === 0) { out.push(line.slice(0, READ_MAX_CHARS) + " ... (line truncated)"); last = n; }
        else last = n - 1;
        break;
      }
      out.push(line); chars += line.length + 1;
    }
    const more = last < total ? `; call again with start_line=${last + 1} for more` : "";
    return ok(`${display(r.real)} (lines ${start_line}-${last} of ${total}${more})\n${out.join("\n")}`);
  },
);

await server.connect(new StdioServerTransport());
log(`serving ${ROOT} from collection ${context.getCollectionName(ROOT)}`);
