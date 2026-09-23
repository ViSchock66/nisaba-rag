"""Standalone verification runner — no pytest required.

Runs a focused set of assertions over the engine using the real dependencies
(chromadb, numpy, onnxruntime, tokenizers) but a stubbed embedder, so nothing
touches the network, a GPU or any personal document.

Usage:
    python tools/verify.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nisaba_rag.config import (  # noqa: E402
    DEFAULTS,
    load_config,
    resolve_path,
    validate_collection_name,
)
from nisaba_rag.hybrid import RRF_K, hybrid_search, rrf_fuse  # noqa: E402
from nisaba_rag.indexer import index_path, iter_files, search_documents  # noqa: E402
from nisaba_rag.rag import (  # noqa: E402
    EmbeddingModelMissing,
    RagStore,
    check_embedding_backend,
    chunk_text,
    embed,
)
from nisaba_rag.rerank import Reranker, RerankerUnavailable, reset_reranker  # noqa: E402
from nisaba_rag.sparse import SparseIndex, tokenize  # noqa: E402
import nisaba_rag.rag as rag  # noqa: E402
import nisaba_rag.indexer as indexer  # noqa: E402

PASSED = 0
FAILED: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
    else:
        FAILED.append(f"{label}{f' -- {detail}' if detail else ''}")


def expect_raises(label: str, exc_type, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc_type:
        check(label, True)
    except Exception as exc:  # noqa: BLE001
        check(label, False, f"expected {exc_type.__name__}, got {type(exc).__name__}: {exc}")
    else:
        check(label, False, f"expected {exc_type.__name__}, nothing raised")


# ── deterministic offline embedder ────────────────────────────────────

import hashlib  # noqa: E402

EMBED_DIM = 64


def fake_embed(texts, cfg: dict | None = None):
    """Deterministic offline embedder; accepts one text or a list."""
    single = isinstance(texts, str)
    batch = [texts] if single else list(texts)
    out = []
    for text in batch:
        vector = [0.0] * EMBED_DIM
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            vector[digest[0] % EMBED_DIM] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        out.append([v / norm for v in vector])
    return out[0] if single else out


def _embed_cfg(url: str, **overrides) -> dict:
    """Minimal config pointing the embedding tier at `url`."""
    return {
        **DEFAULTS,
        "embed": {**DEFAULTS["embed"], "base_url": url, **overrides},
    }


def main() -> int:
    # Some sandboxed environments deny writes to the system temp directory, so
    # the scratch space defaults to a folder next to the checkout.
    base = Path(os.environ.get(
        "NISABA_VERIFY_TMP",
        Path(__file__).resolve().parent.parent / ".verify-tmp" / "runs",
    ))
    # Reuse a dedicated scratch folder and clear it at the end, so repeated runs
    # never leave a transient tree behind. Override NISABA_VERIFY_TMP when the
    # checkout is read-only.
    base.mkdir(parents=True, exist_ok=True)
    tmp = base
    try:
        cfg = {
            **DEFAULTS,
            "store": {"path": str(tmp / "chroma"), "collection": "verify"},
            "index_db": str(tmp / "index.sqlite"),
        }

        # ── chunking ──────────────────────────────────────────────────
        check("chunk: empty text", chunk_text("") == [])
        check("chunk: short text is one chunk", chunk_text("abc", 500, 100) == ["abc"])
        chunks = chunk_text("abcdefghij" * 20, size=50, overlap=10)
        check("chunk: overlaps correctly", chunks[0][-10:] == chunks[1][:10])
        check("chunk: clamped overlap terminates", bool(chunk_text("abcdefghij", 10, 999)))
        expect_raises("chunk: size=0 raises", ValueError, chunk_text, "abc", 0, 0)

        # ── config ────────────────────────────────────────────────────
        check("config: no absolute path in defaults", DEFAULTS["store"]["path"] == "data/chroma")
        check("config: reranker on by default", DEFAULTS["reranker"]["enabled"] is True)
        check("config: loads", load_config()["chunk"]["size"] > 0)
        check("config: relative path resolves under home",
              str(resolve_path("data/chroma")).endswith("data\\chroma")
              or str(resolve_path("data/chroma")).endswith("data/chroma"))
        check("config: valid collection accepted", validate_collection_name("documents") == "documents")
        expect_raises("config: short collection rejected", ValueError, validate_collection_name, "x")
        expect_raises("config: punctuated collection rejected", ValueError,
                      validate_collection_name, "docs!")

        # ── sparse / BM25 ─────────────────────────────────────────────
        check("sparse: accent folding", tokenize("Dimensión") == ["dimension"])
        check("sparse: stopwords dropped", "de" not in tokenize("el tamaño de la ventana"))

        index = SparseIndex()
        index.build([
            {"id": "1", "source": "a.md", "text": "the retry policy uses exponential backoff"},
            {"id": "2", "source": "b.md", "text": "the embedding model has 1024 dimensions"},
            {"id": "3", "source": "c.md", "text": "chunk size and overlap configuration"},
        ])
        hits = index.search("exponential backoff")
        check("sparse: rare term ranks first", bool(hits) and hits[0]["id"] == "1")
        check("sparse: result shape", set(hits[0]) == {"id", "source", "text", "score"})
        check("sparse: empty query", index.search("de la el") == [])
        check("sparse: len", len(index) == 3)
        index.build([{"id": "9", "source": "new.md", "text": "totally different"}])
        check("sparse: rebuild replaces corpus", len(index) == 1 and index.search("exponential") == [])

        # ── store ─────────────────────────────────────────────────────
        store = RagStore(cfg)
        check("store: starts empty", store.stats() == {"files": 0, "chunks": 0})
        check("store: missing hash is None", store.stored_hash("/tmp/a.md") is None)
        store.upsert_file("/tmp/a.md", "abc", 2)
        check("store: hash roundtrip", store.stored_hash("/tmp/a.md") == "abc")
        store.remove_file("/tmp/a.md")
        check("store: remove clears hash", store.stored_hash("/tmp/a.md") is None)

        check("store: chunk ids stable", RagStore.chunk_id("a.md", 0) == RagStore.chunk_id("a.md", 0))
        check("store: chunk ids unique per index",
              RagStore.chunk_id("a.md", 0) != RagStore.chunk_id("a.md", 1))

        texts = chunk_text("retry policy exponential backoff " * 20, size=100, overlap=20)
        store.add_chunks("retry.md", texts, [fake_embed(t) for t in texts], ".md")
        store.upsert_file("retry.md", "h", len(texts))
        found = store.search(fake_embed("retry policy exponential backoff"), limit=3)
        check("store: dense search returns the right source",
              bool(found) and found[0]["source"] == "retry.md")
        check("store: similarity in range", 0.0 <= found[0]["similarity"] <= 1.0)
        check("store: sparse search works", bool(store.sparse_search("backoff")))

        store.add_chunks("other.md", ["palette teal amber"], [fake_embed("palette teal amber")], ".md")
        filtered = store.search(fake_embed("palette"), limit=5, source_filter="other.md")
        check("store: source filter", [h["source"] for h in filtered] == ["other.md"])
        total_chunks = store.stats()["chunks"]
        check("store: limit above corpus size is safe",
              len(store.search(fake_embed("palette"), limit=total_chunks + 50)) == total_chunks,
              f"chunks={total_chunks}")

        store.reset()
        check("store: reset empties", store.stats() == {"files": 0, "chunks": 0})
        store.close()

        # ── persistence ───────────────────────────────────────────────
        first = RagStore(cfg)
        first.add_chunks("p.md", ["persisted"], [fake_embed("persisted")], ".md")
        first.upsert_file("p.md", "h", 1)
        first.close()
        second = RagStore(cfg)
        check("store: persists between instances", second.stats() == {"files": 1, "chunks": 1})
        second.reset()
        second.close()

        # ── indexing pipeline ─────────────────────────────────────────
        # Keep the real network functions: `embed` delegates to `embed_many`
        # through the module global, so the stub below would otherwise hijack
        # the backend-diagnostics checks further down.
        rag_embed_orig = rag.embed
        rag_embed_many_orig = rag.embed_many
        indexer_embed_many_orig = indexer.embed_many

        rag.embed = fake_embed
        rag.embed_many = fake_embed
        indexer.embed_many = fake_embed

        corpus = tmp / "corpus"
        (corpus / "nested").mkdir(parents=True)
        (corpus / "intro.md").write_text(
            "# Intro\n\nthe retriever combines dense embeddings with sparse BM25 search\n", encoding="utf-8")
        (corpus / "retry.md").write_text(
            "# Retry\n\na retry policy with exponential backoff mitigates rate limiting\n", encoding="utf-8")
        (corpus / "nested" / "config.yaml").write_text("chunk:\n  size: 500\n  overlap: 100\n", encoding="utf-8")
        (corpus / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (corpus / "node_modules").mkdir()
        (corpus / "node_modules" / "junk.md").write_text("junk", encoding="utf-8")

        check("index: excluded dir skipped",
              not any("node_modules" in str(p) for p in iter_files(corpus, cfg)))

        fresh = RagStore(cfg)
        result = index_path(fresh, cfg, str(corpus), embed_fn=fake_embed)
        check("index: three supported files", result["indexed"] == 3, str(result))
        second_run = index_path(fresh, cfg, str(corpus), embed_fn=fake_embed)
        check("index: incremental second run", second_run["indexed"] == 0 and second_run["skipped"] == 3)

        (corpus / "retry.md").write_text("# Retry\n\nnow with jitter\n", encoding="utf-8")
        changed = index_path(fresh, cfg, str(corpus), embed_fn=fake_embed)
        check("index: changed file reindexed", changed["indexed"] == 1)

        forced = index_path(fresh, cfg, str(corpus), force=True, embed_fn=fake_embed)
        check("index: force reindexes all", forced["indexed"] == 3)

        big = corpus / "big.md"
        big.write_text("word " * 2000, encoding="utf-8")
        index_path(fresh, cfg, str(big), embed_fn=fake_embed)
        check("index: multi-chunk file", fresh.stats()["chunks"] > 1)
        big.write_text("word", encoding="utf-8")
        index_path(fresh, cfg, str(big), embed_fn=fake_embed)
        check("index: shrinking file leaves no stale chunks", fresh.stats()["chunks"] == 4,
              f"chunks={fresh.stats()['chunks']}")

        missing = index_path(fresh, cfg, str(tmp / "nope"), embed_fn=fake_embed)
        check("index: missing path reports error", "error" in missing)

        def broken_embed(texts, config):
            raise RuntimeError("embedding server down")

        errors = index_path(fresh, cfg, str(corpus), embed_fn=broken_embed, force=True)
        check("index: embed failure counted not raised",
              errors["errors"] >= 1 and errors["indexed"] == 0)
        fresh.close()

        # ── RRF ───────────────────────────────────────────────────────
        dense = [{"id": "a", "source": "a.md"}, {"id": "b", "source": "b.md"}]
        sparse = [{"id": "b", "source": "b.md"}, {"id": "c", "source": "c.md"}]
        fused = rrf_fuse(dense, sparse, top_n=3)
        check("rrf: doc in both runs wins", fused[0]["id"] == "b")
        check("rrf: formula", rrf_fuse([{"id": "a", "source": "a.md"}], [], top_n=1)[0]["rrf_score"]
              == 1.0 / (RRF_K + 1))
        check("rrf: empty input", rrf_fuse([], []) == [])
        check("rrf: hits without id ignored", rrf_fuse([{"source": "x.md"}], []) == [])
        check("rrf: top_n truncates",
              len(rrf_fuse([{"id": str(i), "source": f"{i}.md"} for i in range(10)], [], top_n=4)) == 4)

        # ── end to end ────────────────────────────────────────────────
        rag.embed = fake_embed
        rag.embed_many = fake_embed
        end_store = RagStore(cfg)
        index_path(end_store, cfg, str(corpus), force=True, embed_fn=fake_embed)

        dense_hits = search_documents(end_store, cfg, "retry policy backoff", limit=2, mode="dense")
        check("e2e: dense finds retry.md",
              bool(dense_hits) and dense_hits[0]["source"].endswith("retry.md"))

        hybrid_hits = hybrid_search(end_store, "chunk overlap", cfg, top_n=2)
        check("e2e: hybrid finds config.yaml",
              bool(hybrid_hits) and hybrid_hits[0]["source"].endswith("yaml"))
        check("e2e: hybrid exposes rrf_score", "rrf_score" in hybrid_hits[0])

        provenance = search_documents(end_store, cfg, "retriever", limit=1)[0]
        check("e2e: provenance fields",
              {"id", "text", "source", "chunk", "distance", "similarity"} <= set(provenance))

        end_store.remove_file(str(corpus / "retry.md"))
        after_delete = search_documents(end_store, cfg, "retry policy backoff", limit=5, mode="hybrid")
        check("e2e: deleted source gone",
              all(not (h["source"] or "").endswith("retry.md") for h in after_delete))

        end_store.reset()
        check("e2e: empty index search is safe",
              search_documents(end_store, cfg, "nothing", limit=5) == [])
        end_store.close()

        # ── reranker contract ─────────────────────────────────────────
        reset_reranker()
        expect_raises("rerank: missing model raises clearly", RerankerUnavailable,
                      Reranker, {"reranker": {"model_dir": str(tmp / "absent"), "onnx_file": "model.onnx"}})
        stub = Reranker.__new__(Reranker)
        stub.cfg = {"provider": "cpu"}
        check("rerank: cpu default", stub._pick_providers() == ["CPUExecutionProvider"])
        stub.cfg = {"provider": "nonsense"}
        check("rerank: unknown provider falls back to cpu",
              stub._pick_providers() == ["CPUExecutionProvider"])

        scored = Reranker.__new__(Reranker)
        scored.temperature = 1.0
        scored.relevant_logit_index = 1
        queue = iter([0.1, 5.0, -2.0])
        scored._score_one = lambda q, p: next(queue)
        ordered = scored.rerank("q", [{"id": "a"}, {"id": "b"}, {"id": "c"}])
        check("rerank: sorts by margin", [i["id"] for i in ordered] == ["b", "a", "c"])
        check("rerank: exposes margin and score",
              "rerank_margin" in ordered[0] and 0.0 < ordered[0]["score"] < 1.0)

        calls: list[str] = []
        one_at_a_time = Reranker.__new__(Reranker)
        one_at_a_time.temperature = 1.0
        one_at_a_time.relevant_logit_index = 1
        one_at_a_time._score_one = lambda q, p: (calls.append(p), 1.0)[1]
        one_at_a_time.rerank("q", [{"text": "one"}, {"text": "two"}])
        check("rerank: scores one pair per call", calls == ["one", "two"])
        check("rerank: empty input", one_at_a_time.rerank("q", []) == [])

        # ── backend diagnostics (against a stub server) ───────────────
        import json as _json
        import socket as _socket
        import threading as _threading
        from http.server import BaseHTTPRequestHandler as _BHRH, HTTPServer as _HTTPServer

        class _Stub(_BHRH):
            mode = "ok"
            dims = 1024

            def do_POST(self):
                body = _json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if type(self).mode == "missing":
                    payload = _json.dumps({"error": f'model "{body.get("model")}" not found, try pulling it first'}).encode()
                    self.send_response(404)
                elif type(self).mode == "no_endpoint":
                    payload = _json.dumps({"error": "unknown endpoint"}).encode()
                    self.send_response(404)
                else:
                    payload = _json.dumps({"embeddings": [[0.1] * type(self).dims]}).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        _Stub.mode = "ok"
        server = _HTTPServer(("127.0.0.1", 0), _Stub)
        _threading.Thread(target=server.serve_forever, daemon=True).start()
        stub_url = f"http://127.0.0.1:{server.server_port}"

        probe = _socket.socket(); probe.bind(("127.0.0.1", 0))
        dead_url = f"http://127.0.0.1:{probe.getsockname()[1]}"; probe.close()

        try:
            healthy = check_embedding_backend(_embed_cfg(stub_url, dims=1024))
            check("doctor: healthy backend reports ok", healthy["ok"] is True)
            check("doctor: reports dimensions", healthy["dimensions"] == 1024)

            _Stub.mode = "missing"
            missing = check_embedding_backend(_embed_cfg(stub_url, model="absent-model"))
            check("doctor: missing model detected", missing["reason"] == "model not installed")
            check("doctor: suggests the pull command",
                  missing["hint"] == "ollama pull absent-model", missing["hint"])
            # `embed` must surface the same condition as a specific error, not
            # fold it into the generic "no provider available" failure.
            # `embed` delegates to `embed_many` by module-global lookup, so the
            # stubs installed earlier for the indexing section must be lifted
            # before exercising the real network path.
            rag.embed, rag.embed_many = rag_embed_orig, rag_embed_many_orig
            indexer.embed_many = indexer_embed_many_orig
            expect_raises("doctor: embed raises EmbeddingModelMissing", EmbeddingModelMissing,
                          rag_embed_orig, "hi", _embed_cfg(stub_url, model="absent-model"))

            # A server that answers 404 without naming a model is not a missing
            # model (it may simply lack /api/embed), so it must fall back to the
            # legacy endpoint rather than claim the model is absent.
            _Stub.mode = "no_endpoint"
            fallback = check_embedding_backend(_embed_cfg(stub_url))
            check("doctor: endpoint-less server reported distinctly",
                  fallback.get("reason") == "server does not support /api/embed", str(fallback))

            _Stub.mode = "ok"
            mismatched = check_embedding_backend(_embed_cfg(stub_url, dims=64))
            check("doctor: dimension mismatch detected",
                  mismatched.get("reason") == "dimension mismatch", str(mismatched))

            unreachable = check_embedding_backend(_embed_cfg(dead_url))
            check("doctor: unreachable server detected", unreachable["reason"] == "server unreachable")
        finally:
            server.shutdown()
            server.server_close()

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{PASSED} checks passed, {len(FAILED)} failed")
    for failure in FAILED:
        print(f"  FAIL  {failure}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
