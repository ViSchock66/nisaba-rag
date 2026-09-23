# Security policy

## Reporting a vulnerability

Please **do not** open a public issue for a security problem. Report it
privately through GitHub's "Report a vulnerability" advisory form on the
repository, including a description, reproduction steps and the impact you
believe it has.

Expect an acknowledgement within a few days. This is a volunteer project, so
please be patient with the timeline.

## Scope

Things worth reporting:

- A path traversal or symlink escape that lets indexing read outside the
  configured root.
- Command injection through a file path, query or config value.
- A way for a crafted document to execute code during indexing or search.
- Credential leakage into logs, the vector store or the SQLite database.
- Denial of service through a malicious document (unbounded memory or CPU).

## Not in scope

- **Prompt injection in retrieved text.** Retrieved chunks are data handed to a
  language model; defending against a model being manipulated by its own
  context is the caller's responsibility, not the retriever's. Treat retrieved
  content as untrusted input in whatever consumes it.
- **Local-only exposure.** The MCP server speaks stdio and is launched by the
  client. It does not open a network port; there is no authentication model
  because there is no remote surface.
- **Content of the embedding provider.** If you configure a remote embeddings
  API, your document text is sent to that provider. That is a deliberate
  configuration choice, not a vulnerability.

## Handling your own data

- The index under `data/` contains **your document text**. It is gitignored by
  default — keep it that way.
- `.env` holds provider keys and is gitignored. Never paste a key into a config
  file, an issue, or a screenshot.
- Deleting `data/` removes the index completely; it can be rebuilt from your
  documents at any time.
