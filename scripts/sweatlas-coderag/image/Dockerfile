# claude-context offline index builder + read-only MCP servers for SWE-Atlas QA CodeRAG
# (claude-context search_code, and the three-tool SWE-QA-Agent baseline).
FROM node:22.20.0-bookworm
WORKDIR /opt/coderag
COPY package.json package-lock.json ./
# faiss-node (an unused optional backend of claude-context-core) needs cmake-js to
# build; skip install scripts and rebuild only the tree-sitter grammars the AST
# splitter loads.
RUN npm ci --omit=dev --ignore-scripts --no-audit --no-fund \
 && npm rebuild tree-sitter tree-sitter-go tree-sitter-cpp tree-sitter-java tree-sitter-rust \
      tree-sitter-scala tree-sitter-python tree-sitter-c-sharp tree-sitter-javascript tree-sitter-typescript
COPY build-index.mjs setup-milvus.mjs mcp-check.mjs query-instruction.mjs serve-mcp.mjs serve-sweqa.mjs ./
ENV PATH=/opt/coderag/node_modules/.bin:$PATH \
    NODE_ENV=production
# Default: serve claude-context (stdio) over SSE on :8000 (GET /sse, POST /message).
CMD ["supergateway", "--stdio", "node /opt/coderag/serve-mcp.mjs", "--port", "8000"]
