// claude-context MCP server entrypoint with an optional query instruction
// (see query-instruction.mjs).
import { applyQueryInstruction } from "./query-instruction.mjs";

applyQueryInstruction();
await import("@zilliz/claude-context-mcp/dist/index.js");
