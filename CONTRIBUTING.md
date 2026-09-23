# Contributing

Thanks for taking a look. This project is small on purpose — the goal is a
readable retrieval engine, not a framework.

## Getting set up

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -e ".[dev]"
```

`pip install -e ".[dev,rerank]"` additionally installs the optional reranker
dependencies.

## Running the checks

```bash
pytest                      # full suite, fully offline
ruff check .                # lint
ruff format .               # format
```

The test suite must pass **without** an embedding server, a GPU or any network
access. Embeddings are stubbed in `tests/conftest.py` with a deterministic
local implementation. If you add a test that needs a real service, mark it
`@pytest.mark.integration` and make it skip when the service is absent.

## Guidelines

- **Never commit data.** `data/`, `*.sqlite`, `.env` and anything under
  `.venv/` are gitignored. If you are about to commit an index, a golden set
  derived from private documents, or a config with an absolute path, stop.
- **No absolute paths.** Paths in configs are relative to `NISABA_HOME`. This
  is what keeps a clone portable across machines.
- **No secrets in tracked files.** API keys go in `.env` (gitignored) or the
  environment. Config files never carry credentials.
- **Fail loudly on embeddings.** A failed embedding must raise, never fall back
  to a zero vector — that would silently corrupt the index.
- **Keep the engine framework-free.** No imports from any agent framework or
  harness. Integration happens over MCP.
- **Comment the *why*.** Non-obvious constants (RRF's `k = 60`, BM25's `k1` and
  `b`, the logit margin calibration) should carry a short explanation of the
  reasoning, not a restatement of the code.

## Pull requests

1. One concern per PR.
2. Tests for new behaviour, and a note in the description about how you
   verified it.
3. Update `README.md` when user-facing behaviour changes, and `docs/` when the
   design changes.

## Reporting bugs

Include the Python version, your OS, the output of `nisaba-rag status`, and
which embedding backend you use. Do **not** paste document contents — a
description of the shape of the corpus is enough.
