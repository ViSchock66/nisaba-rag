# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-01-01

### Added

- Incremental indexing pipeline: content-hash tracking in SQLite, so unchanged
  files are never re-embedded.
- Character-based chunking with configurable size and overlap.
- Dense retrieval over ChromaDB (cosine space).
- Sparse retrieval via a self-contained Okapi BM25 implementation, with accent
  normalization and Spanish/English stopwords.
- Hybrid retrieval fusing dense and sparse rankings with Reciprocal Rank Fusion.
- Cross-encoder reranker on ONNX Runtime, scoring pairs individually and
  ranking on the logit margin. Shipped as a core dependency and enabled by
  default; only the model file is a separate download.
- Batched embeddings via Ollama's `/api/embed`, with a transparent fallback to
  the legacy per-text `/api/embeddings` endpoint. Measured ~2x faster indexing
  end to end.
- MCP stdio server exposing indexing, search, status and source management tools.
- Command line interface covering the same operations.
- Layered configuration: environment variables over `config.json` over defaults,
  with paths resolved against `NISABA_HOME` for portability.
- Collection-name validation with an actionable error, since the vector store
  rejects malformed names opaquely.
- `nisaba-rag doctor` / `check_backend()`: diagnoses the embedding backend and
  the reranker, distinguishing an unreachable server from a model that was
  never pulled from a dimension mismatch, and printing the exact fix.
- A missing embedding model raises a dedicated error carrying the `ollama pull`
  command, instead of being folded into the generic failure message.
- CLI output is reconfigured to UTF-8, so a document containing a BOM, an emoji
  or CJK text no longer crashes the command on a legacy Windows code page.
- Offline test suite: no embedding server, GPU or network required.
- Separate reranker test module that loads a real ONNX model and skips cleanly
  when none is configured.
- `tools/verify.py` and `tools/verify_reranker.py`: standalone install checks
  that need no test runner, covering the engine and the real inference path.
- `tools/mcp_client_probe.py`: a minimal MCP client that launches the server
  over stdio the way a harness does, performs the handshake, and drives indexing
  and both search modes through the protocol. It separates a server problem
  from a client-registration problem.
- Recall@k / MRR evaluation harness with a golden-set format.
- GPU execution providers available as opt-in extras (`gpu-cuda`, `gpu-dml`).

### Documentation

- README: what is indexed and where it is stored, including the fact that
  Nisaba watches no folder and does not read conversations.
- README: how a harness connects over stdio, why `cwd` matters, and what
  "harness-agnostic" implies in practice.
- README: how to verify an install using the three standalone scripts.
- README: troubleshooting entries for a missing model and a dimension mismatch.

### Fixed

- `search_documents(rerank=True)` no longer fails the whole search when no
  reranker model is installed; it degrades to the unranked ordering, which is
  what lets a fresh clone work out of the box.
- The legacy `embed.ollama_base_url` config key is honoured again. The check
  previously ran after merging with the defaults, where `base_url` is always
  present, so an older config file had its URL silently ignored.
- `--json` is accepted both before and after the subcommand
  (`nisaba-rag --json sources` and `nisaba-rag sources --json`).
- `load_config()` deep-copies its result, so a caller that mutates a nested
  value can no longer corrupt the module-level defaults.
- A test module contained a syntax error that made the whole file fail to
  collect. The suite is now actually executed and passing.

### Notes

- Requires `mcp>=2.0.0`. The MCP Python SDK renamed `FastMCP` to `MCPServer`
  in v2 and removed `mcp.server.fastmcp`, so v1 is excluded rather than
  merely discouraged.
- A truncated batch response falls back to per-text requests instead of
  pairing vectors with the wrong chunks.

[Unreleased]: https://github.com/ViSchock66/nisaba-rag/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ViSchock66/nisaba-rag/releases/tag/v0.1.0
