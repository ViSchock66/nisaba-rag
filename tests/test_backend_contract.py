"""Tests for the CLI and for the HTTP contract with the embedding backend.

`TestEmbeddingBackend` starts a stub server that speaks the Ollama
`/api/embeddings` protocol and points the engine at it. This is what proves
Nisaba talks to *your* configured server over HTTP and has nothing hardcoded:
the stub runs on an ephemeral port, so a passing run also means no fixed port
or baked-in URL is involved.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from nisaba_rag.config import DEFAULTS
from nisaba_rag.indexer import index_path, search_documents
from nisaba_rag.rag import embed, embed_many

EMBED_DIM = 32


class _EmbeddingHandler(BaseHTTPRequestHandler):
    """Minimal Ollama-compatible embeddings endpoint.

    Serves both the current ``/api/embed`` (batched, ``input``) and the legacy
    ``/api/embeddings`` (single, ``prompt``) so the fallback path is covered.
    """

    received: list[dict] = []
    supports_batch = True

    def do_POST(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).received.append({"path": self.path, "body": body})

        if self.path == "/api/embed":
            if not type(self).supports_batch:
                payload = json.dumps({"error": "unknown endpoint"}).encode()
                self.send_response(404)
            else:
                texts = body.get("input", [])
                if isinstance(texts, str):
                    texts = [texts]
                payload = json.dumps({"embeddings": [_vector(t) for t in texts]}).encode()
                self.send_response(200)
        else:
            payload = json.dumps({"embedding": _vector(body.get("prompt", ""))}).encode()
            self.send_response(200)

        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def _vector(text: str) -> list[float]:
    vector = [0.0] * EMBED_DIM
    for word in text.lower().split():
        vector[sum(map(ord, word)) % EMBED_DIM] += 1.0
    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    return [v / norm for v in vector]


@pytest.fixture
def fake_ollama():
    """A stub embedding server on an ephemeral port."""
    _EmbeddingHandler.received = []
    _EmbeddingHandler.supports_batch = True
    server = HTTPServer(("127.0.0.1", 0), _EmbeddingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", _EmbeddingHandler.received
    finally:
        server.shutdown()
        server.server_close()


class TestEmbeddingBackend:
    def test_single_embed_uses_the_batched_endpoint(self, fake_ollama):
        url, received = fake_ollama
        cfg = {**DEFAULTS, "embed": {**DEFAULTS["embed"], "base_url": url}}

        vector = embed("hello world", cfg)

        assert len(vector) == EMBED_DIM
        assert len(received) == 1, "a single text must not fan out into extra requests"
        assert received[0]["path"] == "/api/embed"
        assert received[0]["body"]["input"] == ["hello world"]
        assert received[0]["body"]["model"] == cfg["embed"]["model"]

    def test_embed_many_sends_one_request_for_the_whole_batch(self, fake_ollama):
        url, received = fake_ollama
        cfg = {**DEFAULTS, "embed": {**DEFAULTS["embed"], "base_url": url}}

        vectors = embed_many(["one", "two", "three"], cfg)

        assert len(vectors) == 3
        assert len(received) == 1, f"expected a single batched request, got {len(received)}"
        assert received[0]["body"]["input"] == ["one", "two", "three"]

    def test_order_is_preserved_across_the_batch(self, fake_ollama):
        url, _ = fake_ollama
        cfg = {**DEFAULTS, "embed": {**DEFAULTS["embed"], "base_url": url}}

        vectors = embed_many(["alpha", "beta"], cfg)
        assert vectors[0] == _vector("alpha")
        assert vectors[1] == _vector("beta")

    def test_empty_batch_makes_no_request(self, fake_ollama):
        url, received = fake_ollama
        cfg = {**DEFAULTS, "embed": {**DEFAULTS["embed"], "base_url": url}}

        assert embed_many([], cfg) == []
        assert received == []

    def test_falls_back_to_the_legacy_endpoint(self, fake_ollama):
        """A server that predates /api/embed must still work, per text."""
        url, received = fake_ollama
        _EmbeddingHandler.supports_batch = False
        cfg = {**DEFAULTS, "embed": {**DEFAULTS["embed"], "base_url": url}}

        vectors = embed_many(["one", "two"], cfg)

        assert len(vectors) == 2
        legacy = [r for r in received if r["path"] == "/api/embeddings"]
        assert len(legacy) == 2, "each text needs its own legacy request"
        assert legacy[0]["body"]["prompt"] == "one"

    def test_a_short_batch_response_falls_back_instead_of_misaligning(self, fake_ollama):
        """A truncated batch would silently pair vectors with the wrong chunks."""
        url, received = fake_ollama

        original = _EmbeddingHandler.do_POST

        def truncated(self):
            if self.path == "/api/embed":
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                type(self).received.append({"path": self.path, "body": body})
                payload = json.dumps({"embeddings": [_vector("only-one")]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                original(self)

        _EmbeddingHandler.do_POST = truncated
        try:
            cfg = {**DEFAULTS, "embed": {**DEFAULTS["embed"], "base_url": url}}
            vectors = embed_many(["a", "b", "c"], cfg)
        finally:
            _EmbeddingHandler.do_POST = original

        assert len(vectors) == 3, "must not return fewer vectors than inputs"
        assert len([r for r in received if r["path"] == "/api/embeddings"]) == 3

    def test_indexing_and_searching_use_the_configured_server(self, fake_ollama, tmp_path):
        url, received = fake_ollama
        cfg = {
            **DEFAULTS,
            "embed": {**DEFAULTS["embed"], "base_url": url},
            "store": {"path": str(tmp_path / "chroma"), "collection": "stub_test"},
            "index_db": str(tmp_path / "index.sqlite"),
        }

        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "note.md").write_text(
            "the chunk size is 500 characters with an overlap of 100", encoding="utf-8"
        )

        from nisaba_rag.rag import RagStore

        store = RagStore(cfg)
        try:
            result = index_path(store, cfg, str(docs))
            assert result["indexed"] == 1
            assert result["errors"] == 0
            assert received, "the stub server was never called"

            hits = search_documents(store, cfg, "chunk size overlap", limit=1)
            assert hits
            assert hits[0]["source"].endswith("note.md")
        finally:
            store.close()

    def test_a_dead_server_raises_instead_of_returning_a_zero_vector(self, tmp_path):
        """Silently indexing with a zero vector would poison the index."""
        # Bind then close, to get a port nothing is listening on.
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
        probe.close()

        cfg = {
            **DEFAULTS,
            "embed": {**DEFAULTS["embed"], "base_url": f"http://127.0.0.1:{dead_port}"},
        }
        with pytest.raises(RuntimeError, match="no provider available"):
            embed("anything", cfg)

    def test_indexing_reports_errors_when_the_server_is_down(self, tmp_path):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
        probe.close()

        cfg = {
            **DEFAULTS,
            "embed": {**DEFAULTS["embed"], "base_url": f"http://127.0.0.1:{dead_port}"},
            "store": {"path": str(tmp_path / "chroma"), "collection": "down_test"},
            "index_db": str(tmp_path / "index.sqlite"),
        }
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "note.md").write_text("some content", encoding="utf-8")

        from nisaba_rag.rag import RagStore

        store = RagStore(cfg)
        try:
            result = index_path(store, cfg, str(docs))
            assert result["indexed"] == 0
            assert result["errors"] == 1
            assert "no provider available" in result["error_details"][0]
        finally:
            store.close()


class TestCli:
    def test_stdio_is_reconfigured_for_unicode(self):
        """A BOM, an emoji or CJK text in a document must not crash the CLI."""
        from nisaba_rag.cli import _configure_stdio

        _configure_stdio()  # must not raise

    def test_index_reports_a_missing_path(self, tmp_path, monkeypatch, capsys):
        from nisaba_rag.cli import main

        monkeypatch.setenv("NISABA_DATA_DIR", str(tmp_path / "cli-data"))
        rc = main(["index", str(tmp_path / "does-not-exist")])
        captured = capsys.readouterr()
        assert rc == 1
        assert "error" in (captured.err + captured.out).lower()

    def test_status_runs(self, tmp_path, monkeypatch, capsys):
        from nisaba_rag.cli import main

        monkeypatch.setenv("NISABA_DATA_DIR", str(tmp_path / "cli-stat"))
        assert main(["status"]) == 0
        assert "files=" in capsys.readouterr().out

    def test_sources_emits_json(self, tmp_path, monkeypatch, capsys):
        from nisaba_rag.cli import main

        monkeypatch.setenv("NISABA_DATA_DIR", str(tmp_path / "cli-json"))
        assert main(["sources", "--json"]) == 0
        assert json.loads(capsys.readouterr().out) == []
