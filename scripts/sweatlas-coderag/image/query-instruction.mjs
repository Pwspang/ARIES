// Optional query instruction for instruction-tuned embedding models.
//
// Instruction-tuned embedding models such as Qwen3-Embedding expect queries
// as "Instruct: <task>\nQuery:<text>" while documents stay plain.
// claude-context embeds documents with embedBatch() and search queries with
// embed() (Context.semanticSearch), so prefixing embed() alone changes only
// the query side; the offline index is unaffected. BM25 still receives the
// raw query text.
//
// Env: CODERAG_QUERY_INSTRUCTION (unset or empty: unchanged behavior).
import { OpenAIEmbedding } from "@zilliz/claude-context-core";

export function applyQueryInstruction() {
  const instruction = process.env.CODERAG_QUERY_INSTRUCTION;
  if (!instruction) return;
  const embed = OpenAIEmbedding.prototype.embed;
  OpenAIEmbedding.prototype.embed = function (text) {
    return embed.call(this, `Instruct: ${instruction}\nQuery:${text}`);
  };
  // stdout carries the MCP stdio protocol; log to stderr only.
  console.error(`[coderag] query instruction: ${instruction}`);
}
