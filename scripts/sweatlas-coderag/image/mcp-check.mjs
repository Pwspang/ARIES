// Talk to a served MCP endpoint the way the harness will.
// Usage: node mcp-check.mjs SSE_URL "query" [--try-clear]
//
// claude-context endpoint (search_code): prints search_code hits for /app;
// with --try-clear also calls clear_index, which the read-only Milvus user
// must refuse.
// SWE-QA baseline endpoint (search_content): checks get_repo_structure,
// search_content, read_file on the top hit, and that read_file refuses a path
// outside /app.
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";

const [url, query = "main entry point", flag] = process.argv.slice(2);
if (!url) throw new Error("usage: mcp-check.mjs SSE_URL [query] [--try-clear]");
const client = new Client({ name: "aries-coderag-check", version: "1.0.0" });
await client.connect(new SSEClientTransport(new URL(url)));
const names = (await client.listTools()).tools.map((t) => t.name);
console.log("tools:", names.join(", "));

const text = (r) => (r.content || []).map((c) => c.text ?? "").join("\n");
const call = async (name, args, show = 1200) => {
  const r = await client.callTool({ name, arguments: args });
  console.log(`${name} ${JSON.stringify(args)} isError=${!!r.isError}\n${text(r).slice(0, show)}\n`);
  return r;
};

let ok;
if (names.includes("search_content")) {
  const structure = await call("get_repo_structure", { max_depth: 1 }, 600);
  const search = await call("search_content", { query, limit: 3 });
  const top = /^1\. (\/app\/\S+?):(\d+)-(\d+)/m.exec(text(search));
  const read = top ? await call("read_file", { path: top[1], start_line: Number(top[2]), end_line: Number(top[2]) + 5 }, 600) : null;
  const escape = await call("read_file", { path: "/etc/passwd" }, 200);
  ok = !structure.isError && /^\/app\//.test(text(structure)) &&
    !search.isError && !!top && read && !read.isError && !!escape.isError;
} else {
  const search = await call("search_code", { path: "/app", query, limit: 3 }, 1500);
  let clearRefused = null;
  if (flag === "--try-clear") {
    const clear = await call("clear_index", { path: "/app" }, 600);
    clearRefused = !!clear.isError || /denied|permission|privilege|not authorized|fail/i.test(text(clear));
  }
  ok = !search.isError && /\/app|Found|```/.test(text(search)) && clearRefused !== false;
}
await client.close();
process.exit(ok ? 0 : 1);
