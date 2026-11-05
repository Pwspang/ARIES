// Talk to a served claude-context MCP endpoint the way the harness will.
// Usage: node mcp-check.mjs SSE_URL "query" [--try-clear]
// Prints the tool list and search_code hits for /app; with --try-clear also
// calls clear_index (which the read-only Milvus user must refuse).
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";

const [url, query = "main entry point", flag] = process.argv.slice(2);
if (!url) throw new Error("usage: mcp-check.mjs SSE_URL [query] [--try-clear]");
const client = new Client({ name: "aries-coderag-check", version: "1.0.0" });
await client.connect(new SSEClientTransport(new URL(url)));
const tools = await client.listTools();
console.log("tools:", tools.tools.map((t) => t.name).join(", "));

const text = (r) => (r.content || []).map((c) => c.text ?? "").join("\n");
const search = await client.callTool({ name: "search_code", arguments: { path: "/app", query, limit: 3 } });
console.log(`search_code isError=${!!search.isError}\n${text(search).slice(0, 1500)}`);

let clearRefused = null;
if (flag === "--try-clear") {
  const clear = await client.callTool({ name: "clear_index", arguments: { path: "/app" } });
  const body = text(clear);
  console.log(`clear_index isError=${!!clear.isError}\n${body.slice(0, 600)}`);
  clearRefused = !!clear.isError || /denied|permission|privilege|not authorized|fail/i.test(body);
}
await client.close();
const ok = !search.isError && /\/app|Found|```/.test(text(search)) && clearRefused !== false;
process.exit(ok ? 0 : 1);
