# Nisaba RAG

Self-hosted retrieval over your own documents, exposed as an MCP server.

Nisaba indexes a folder of text files into a local vector store and answers
queries against it, either from an MCP client or from the terminal. Nothing
leaves the machine unless a remote embedding provider is configured explicitly.

- Local by default: embeddings come from your own server, and the index is a
  local ChromaDB directory plus a SQLite file.
- Harness-agnostic: plain MCP over stdio, so any client that speaks the protocol
  can use it.
- Dense and sparse retrieval: semantic search and BM25 keyword search, fused
  with Reciprocal Rank Fusion.
- Cross-encoder reranking: a second-stage ONNX model reorders the fused
  candidates. The reranker *code* is a core dependency, but the *model* is a
  separate one-time download — see [docs/reranker.md](docs/reranker.md). Until
  you install one, search works and simply returns the fused order.
- Incremental indexing: files are tracked by content hash, so re-indexing a
  large folder only pays for what changed.
- Small surface: eight modules and no required heavyweight ML framework.

---

## How it works

```
                    ┌──────────────────────────────┐
   your documents → │  indexer  (walk + hash)      │
                    └──────────────┬───────────────┘
                                   │ chunks
                                   ▼
                    ┌──────────────────────────────┐
                    │  embed()  → local / remote   │
                    └──────────────┬───────────────┘
                                   ▼
        ┌──────────────────────────────────────────────┐
        │  RagStore                                    │
        │   • ChromaDB   → vectors (cosine)            │
        │   • SQLite     → file hashes + chunk counts  │
        │   • BM25       → in-memory, built lazily     │
        └───────────────────────┬──────────────────────┘
                                │
              query ────────────┼─────────────► dense ─┐
                                │                      ├─ RRF ─► (rerank) ─► hits
                                └─────────────► BM25  ─┘
```

### What gets indexed, and where it goes

**Nisaba does not watch a folder.** It has no default corpus and no filesystem
watcher. Nothing is indexed until a caller passes a path to `index_folder`
(or runs `nisaba-rag index <path>`). Pass a different path and you get a
different corpus; index the same path twice and nothing is reprocessed.

What survives the walk is decided by `config.json`:

| Rule | Default | Effect |
|---|---|---|
| `extensions` | `.md .txt .json .yaml .py .js .ts .html .css …` | Only these file types. |
| `exclude_dirs` | `node_modules .git .venv dist build …` | Pruned before descending. |
| `exclude_files` | `package-lock.json pnpm-lock.yaml …` | Skipped by name. |
| `max_file_kb` | `150` | Larger files are skipped, not truncated. |
| `chunk.size` / `overlap` | `500` / `100` | Characters, with 100 characters shared between neighbours. |

The data lands next to the project (or under `NISABA_DATA_DIR`):

| Path | Contents |
|---|---|
| `data/chroma/` | The vectors — and therefore **your documents, in plain text** |
| `data/index.sqlite` | One SHA-256 per file, which is what makes re-indexing incremental |

Both are gitignored. Never commit them: the index is a copy of your corpus.

> **Nisaba does not read conversations.** It is not a memory of your sessions
> and it does not pick up what an agent writes to disk. A document created
> after the last `index_folder` call is **not** searchable until you index
> again. If you want a folder kept fresh, re-index it (for example at the end
> of a session) — there is no background refresh.

### Embedding in batches

Each file's chunks are sent in a **single** request to the embedding server,
which measured ~2x faster end to end than one request per chunk against a local
`bge-m3`. See `docs/architecture.md` for the fallback behaviour.

### Modules

| Module | Responsibility |
|---|---|
| `nisaba_rag/config.py` | Layered config + portable path resolution |
| `nisaba_rag/rag.py` | Embeddings, chunking, store, dense search |
| `nisaba_rag/sparse.py` | Okapi BM25 lexical index |
| `nisaba_rag/hybrid.py` | RRF fusion of dense + sparse |
| `nisaba_rag/indexer.py` | Incremental indexing pipeline |
| `nisaba_rag/rerank.py` | Cross-encoder reranker (ONNX Runtime) |
| `nisaba_rag/server.py` | MCP stdio server |
| `nisaba_rag/cli.py` | Command line interface |

---

## Install

Requires **Python 3.10+**.

```bash
git clone https://github.com/ViSchock66/nisaba-rag.git
cd nisaba-rag
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -e ".[dev]"
```

### Two runtime pieces, one of them optional

Nisaba needs **two** runtime pieces beyond Python, because it does two
different jobs:

| Piece | Purpose | Required? |
|---|---|---|
| An embeddings server | Turns text into vectors | **Yes** |
| A reranker model | Cross-encoder reordering of candidates | **No — recommended** |

The reranker *code* is a core dependency (installed above). The reranker
*model* is a separate 100–500 MB file that cannot be committed to a repository,
so it is a one-time download — see [docs/reranker.md](docs/reranker.md).

`reranker.enabled` is `true` in the default `config.json`, so a fresh clone
reports a `WARN` in `nisaba-rag doctor` until you install a model. That is
expected, not a broken install: search keeps working and returns the fused
order without reranking. Install a model to silence the warning, or set
`"enabled": false` to turn reranking off explicitly.

### Embedding backend

**Nisaba needs a running embedding server to index or search anything.** There
is no offline mode: without embeddings there is no RAG. The tests are the only
part that runs without one (see [Tests](#tests)).

By default Nisaba talks to an Ollama-compatible server:

```bash
ollama pull bge-m3
ollama serve          # listens on 127.0.0.1:11434
```

The model is named in `config.json`:

```json
"embed": {
    "model": "bge-m3",
    "base_url": "http://127.0.0.1:11434"
}
```

Change `model` to use a different one — but **pull it first**, and re-index
afterwards. Any server exposing `POST /api/embed` (and, as a fallback, the
legacy `POST /api/embeddings`) works, not just Ollama. Nisaba batches the
chunks of each file into a single request, which measures ~2x faster than one
call per chunk. If you would rather use a hosted provider, set
`NISABA_API_KEY` and the second tier (`embed.nim_url` / `embed.nim_model`)
takes over whenever the local server is unreachable.

> **Keep the same model for indexing and querying.** Vectors from different
> models are not comparable. Switching models silently ruins search until you
> re-index with --force. `nisaba-rag doctor` catches this.

### Check your setup first

```bash
nisaba-rag doctor
```

```
embeddings : http://127.0.0.1:11434  model=bge-m3
             OK - 1024 dimensions
reranker   : enabled
             OK - CPUExecutionProvider

all good - ready to index
```

It distinguishes the three failures that otherwise look identical:

```
# server not running
embeddings : ... FAIL - server unreachable
             fix: start the embedding server (ollama serve) or fix embed.base_url

# server running, model never pulled
embeddings : ... FAIL - model not installed
             detail: model "bge-m3" not found, try pulling it first
             fix: ollama pull bge-m3

# model swapped without re-indexing
embeddings : ... FAIL - dimension mismatch
             fix: config declares dims=1024 but returns 768; update embed.dims
                  and re-index with --force
```

Embeddings are the hard requirement, so they decide the exit code. A missing
reranker model is reported as a **warning**, not a failure — the engine still
works, only without reranking.

---

## Use

### Command line

```bash
nisaba-rag index ./docs                 # incremental
nisaba-rag index ./docs --force         # ignore the hash cache
nisaba-rag search "how does retry work" --mode hybrid --limit 5
nisaba-rag doctor                       # check the backend before indexing
nisaba-rag status                       # files + chunks
nisaba-rag sources                      # what is indexed
nisaba-rag delete ./docs/old.md
```

Add `--json` to any command for machine-readable output.

### As an MCP server

```bash
nisaba-rag serve        # or: python -m nisaba_rag.server
```

Register it with your MCP client. Example configuration:

```json
{
  "mcpServers": {
    "nisaba": {
      "command": "/absolute/path/to/nisaba-rag/.venv/bin/python",
      "args": ["-m", "nisaba_rag.server"],
      "env": {
        "NISABA_HOME": "/absolute/path/to/nisaba-rag",
        "NISABA_OLLAMA_URL": "http://127.0.0.1:11434"
      }
    }
  }
}
```

On Windows, point `command` at `.venv\Scripts\python.exe`.

### How a harness actually connects

There is no service to start and no port to open. **The client launches the
server as a child process** and they speak MCP over stdin/stdout:

```
you open the client
   └─► it reads its MCP config
        └─► launches: python -m nisaba_rag.server   (child process)
             └─► handshake: "I am nisaba, here are my 9 tools"
                  └─► the model sees those tools and calls them
you close the client
   └─► the child process exits
```

Three consequences worth knowing:

- **Nisaba is not running unless a client is.** Nothing is installed as a
  service, and it never listens on a network socket.
- **`cwd` matters.** The server resolves `config.json` and `data/` relative to
  `NISABA_HOME`, which defaults to the project root. Launch it from the wrong
  directory and it will read a different config — pin `cwd` in the client
  config, or set `NISABA_HOME` explicitly.
- **The client knows nothing about Nisaba.** It only knows a process declared
  some tools. That is what "harness-agnostic" means: the same server works
  with any MCP client, and no client-specific code exists in this repo.

Before wiring it into a client, verify the server alone:

```bash
python tools/mcp_client_probe.py --cwd . --index ./docs
```

If that passes, a failure afterwards is in the client registration, not here.

### MCP tools

| Tool | Description |
|---|---|
| `check_backend()` | Verify the embedding server and model before indexing. |
| `index_folder(path)` | Index every supported file under a folder (incremental). |
| `index_file(path)` | Index one file if it changed. |
| `reindex_folder(path)` | Force re-index, ignoring the hash cache. |
| `search_documents(query, limit, source, mode, rerank)` | Search; `mode` is `dense` or `hybrid`. |
| `get_index_status()` | Files and chunk counts. |
| `list_sources()` | Every indexed path. |
| `delete_source(source)` | Remove one path from the index. |
| `reranker_status()` | Whether the reranker is usable. |

---

## Configuration

Settings are layered: **environment variables → `config.json` → built-in
defaults**. See `config.json` for the full set and `.env.example` for every
environment variable.

Paths in `config.json` may be **relative**; they resolve against `NISABA_HOME`
(the checkout root by default). That is what keeps a clone portable — no
absolute paths need to be committed.

| Key | Default | Notes |
|---|---|---|
| `embed.model` | `bge-m3` | Must match between indexing and querying. |
| `embed.base_url` | `http://127.0.0.1:11434` | Ollama-compatible endpoint. |
| `store.path` | `data/chroma` | Vector store directory. |
| `index_db` | `data/index.sqlite` | Incremental hash tracking. |
| `chunk.size` / `chunk.overlap` | `500` / `100` | Characters. |
| `extensions` | `.md`, `.txt`, `.py`, … | Indexed file types. |
| `exclude_dirs` | `node_modules`, `.git`, … | Skipped during the walk. |
| `max_file_kb` | `150` | Larger files are skipped. |
| `reranker.enabled` | `true` | Set `false` for dense + hybrid search without reranking. |
| `reranker.model_dir` | `data/reranker/…` | Where the ONNX model lives. See `docs/reranker.md`. |
| `reranker.provider` | `cpu` | `cpu`, `cuda`, `dml`, `coreml`. Falls back to CPU. |

---

## Evaluation

`eval/eval_rag.py` measures recall@k and MRR against a golden set:

```bash
python eval/eval_rag.py --mode dense
python eval/eval_rag.py --mode hybrid --rerank
```

Edit `eval/golden_set.json` to describe *your* corpus: each entry pairs a query
with path fragments that count as a correct answer. The shipped example is
small and generic on purpose — swap it for your own before drawing conclusions
from the numbers.

---

## Tests

```bash
pytest                                  # unit tests; no model or server needed
pytest -m integration                   # also runs tests against a live Ollama
pytest tests/test_reranker_real.py      # only runs when a model is present
```

**Tests are for development, never for using Nisaba.** A user cloning this repo
runs the commands in [Use](#use); the tests never participate in indexing or
search.

They run without a server because a test does not need to search *well*, only
to prove the code does not break. Embeddings are therefore replaced inside
`tests/` by a small deterministic function. That substitution is confined to
the test folder — the product always calls the real HTTP API, and
`tests/test_backend_contract.py` asserts exactly that against a stub server.

The reranker tests load a real ONNX model and are **skipped** when none is
configured:

```bash
NISABA_TEST_RERANKER_DIR=/path/to/model pytest tests/test_reranker_real.py
```

### Sandboxed environments

Roughly half the suite uses pytest's `tmp_path`, and pytest allocates those by
*listing* its base temp directory to pick a numbered scratch dir. Some sandboxes
deny that listing. When they do, those tests fail at setup with `WinError 5` on
a temp path, and `pytest` exits non-zero even though every test that ran passed.

Run the suite in a normal terminal. There is no `--basetemp` value that avoids
it: the listing happens once per test, so pointing the base somewhere else does
not help.

### Verifying an install

Three standalone scripts check a real deployment. None of them needs pytest, and
none touches your harness configuration:

```bash
python tools/verify.py                                       # engine, fully offline
python tools/verify_reranker.py --model-dir /path/to/model   # the real ONNX reranker
python tools/mcp_client_probe.py --cwd . --index ./docs      # the MCP surface, end to end
```

`tools/verify.py` runs ~67 assertions over chunking, BM25, the store, the
incremental indexer, RRF fusion, backend diagnostics and the reranker contract.

`tools/verify_reranker.py` exercises the actual ONNX cross-encoder: that it
loads, that it separates a relevant passage from an unrelated one, times the
inference, checks scoring is batch-independent, and runs the full
dense → rerank pipeline.

`tools/mcp_client_probe.py` is a **minimal MCP client**: it launches the server
over stdio exactly as a harness would, performs the handshake, lists the tools,
and then indexes, searches (dense and hybrid), re-indexes and deletes through
the protocol. Use it to confirm the server works *before* wiring it into any
client — if this passes, the problem is in the harness registration, not in
Nisaba.

```
1. HANDSHAKE
   servidor    : nisaba
   protocolo   : 2025-11-25
   herramientas descubiertas: 9

2. ESTADO INICIAL
   status: {"files": 0, "chunks": 0}

3. DIAGNOSTICO
   embeddings ok : True   base_url: http://127.0.0.1:11434   dimensiones: 1024

4. INDEXAR     -> 2 archivos, 0 errores
5. BUSCAR denso   -> [sim 0.802] guia.md
6. BUSCAR hibrido -> [rrf 0.0328] fusion.md
7. INCREMENTAL    -> 0 indexados, 2 saltados
8. GESTION        -> borrada 1 fuente
```

> On Windows, running an MCP client requires creating pipes for the stdio
> transport. Some sandboxed or constrained environments deny that with
> `WinError 5`; run the probe in a normal terminal if you hit it.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no provider available` | Nothing is serving embeddings. Run `nisaba-rag doctor`. |
| `does not have the model 'bge-m3'` | The model was never pulled. Run the `ollama pull` command in the message. |
| `dimension mismatch` | The embedding model changed after indexing. Update `embed.dims` and re-index with `--force`. |
| Search returns nothing | Nothing indexed yet (`nisaba-rag status`), or the query has no content words (BM25 drops stopwords). |
| `reranker model not found` | Download a model (`docs/reranker.md`) or set `reranker.enabled` to `false`. |
| `database is locked` | Two processes writing the same SQLite file. Close the other one. |

---

## License

MIT — see [LICENSE](LICENSE).
