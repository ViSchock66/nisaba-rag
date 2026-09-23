"""Tests for backend diagnostics and real-model error reporting.

The unit tests use a stub server. The tests marked `integration` talk to a
genuine Ollama instance and are skipped when it is not reachable, so the suite
still passes on a machine without one.

Run them explicitly with:

    pytest -m integration
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from nisaba_rag.config import DEFAULTS
from nisaba_rag.rag import EmbeddingModelMissing, check_embedding_backend, embed

EMBED_DIM = 16


class _StubHandler(BaseHTTPRequestHandler):
    """Configurable stub: healthy, model-missing, or broken.

    It answers both the current batched endpoint (`/api/embed`, returning
    `embeddings`) and the legacy single-text one (`/api/embeddings`, returning
    `embedding`), because the engine uses the first and falls back to the second.
    """

    mode = "ok"
    received: list[dict] = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).received.append(body)

        if type(self).mode == "missing":
            payload = json.dumps(
                {"error": f'model "{body.get("model")}" not found, try pulling it first'}
            ).encode()
            self.send_response(404)
        elif type(self).mode == "broken":
            payload = b'{"error": "internal server error"}'
            self.send_response(500)
        elif self.path == "/api/embed":
            texts = body.get("input", [])
            if isinstance(texts, str):
                texts = [texts]
            payload = json.dumps(
                {"embeddings": [[0.1] * EMBED_DIM for _ in texts]}
            ).encode()
            self.send_response(200)
        else:
            payload = json.dumps({"embedding": [0.1] * EMBED_DIM}).encode()
            self.send_response(200)

        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_server():
    _StubHandler.mode = "ok"
    _StubHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _cfg_for(url: str, **embed_overrides) -> dict:
    return {
        **DEFAULTS,
        "embed": {**DEFAULTS["embed"], "base_url": url, **embed_overrides},
    }


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class TestModelMissingDetection:
    """A missing model is a one-command fix, so it must be reported as such."""

    def test_embed_raises_a_specific_error(self, stub_server):
        _StubHandler.mode = "missing"
        cfg = _cfg_for(stub_server, model="not-installed")

        with pytest.raises(EmbeddingModelMissing) as excinfo:
            embed("hello", cfg)

        message = str(excinfo.value)
        assert "not-installed" in message
        assert "ollama pull not-installed" in message

    def test_the_server_message_is_preserved(self, stub_server):
        _StubHandler.mode = "missing"
        with pytest.raises(EmbeddingModelMissing, match="try pulling it first"):
            embed("hello", _cfg_for(stub_server, model="nope"))

    def test_model_missing_is_not_masked_as_a_generic_failure(self, stub_server):
        """It must not fall through to the vague 'no provider available' path."""
        _StubHandler.mode = "missing"
        try:
            embed("hello", _cfg_for(stub_server, model="nope"))
        except EmbeddingModelMissing:
            pass
        except RuntimeError as exc:
            pytest.fail(f"raised the generic error instead: {exc}")
        else:
            pytest.fail("no error raised at all")

    def test_other_http_errors_still_fall_through(self, stub_server):
        _StubHandler.mode = "broken"
        with pytest.raises(RuntimeError, match="no provider available"):
            embed("hello", _cfg_for(stub_server))


class TestCheckEmbeddingBackend:
    def test_reports_ok_with_dimensions(self, stub_server):
        report = check_embedding_backend(_cfg_for(stub_server, dims=EMBED_DIM))
        assert report["ok"] is True
        assert report["dimensions"] == EMBED_DIM

    def test_reports_an_unreachable_server(self):
        report = check_embedding_backend(_cfg_for(f"http://127.0.0.1:{_free_port()}"))
        assert report["ok"] is False
        assert report["reason"] == "server unreachable"
        assert "ollama serve" in report["hint"]

    def test_reports_a_missing_model_with_the_pull_command(self, stub_server):
        _StubHandler.mode = "missing"
        report = check_embedding_backend(_cfg_for(stub_server, model="absent-model"))
        assert report["ok"] is False
        assert report["reason"] == "model not installed"
        assert report["hint"] == "ollama pull absent-model"

    def test_detects_a_declared_dimension_mismatch(self, stub_server):
        """Changing the embedding model without re-indexing breaks every vector."""
        report = check_embedding_backend(_cfg_for(stub_server, dims=1024))
        assert report["ok"] is False
        assert report["reason"] == "dimension mismatch"
        assert "--force" in report["hint"]

    def test_never_raises_on_a_broken_server(self, stub_server):
        _StubHandler.mode = "broken"
        report = check_embedding_backend(_cfg_for(stub_server))
        assert report["ok"] is False
        assert "HTTP 500" in report["reason"]


class TestDoctorCommand:
    def test_succeeds_when_embeddings_work(self, stub_server, tmp_path, monkeypatch, capsys):
        from nisaba_rag.cli import main

        monkeypatch.setenv("NISABA_DATA_DIR", str(tmp_path / "doc-data"))
        monkeypatch.setenv("NISABA_OLLAMA_URL", stub_server)
        monkeypatch.setenv("NISABA_CONFIG", str(tmp_path / "absent.json"))
        # The stub answers with EMBED_DIM dimensions, so the config must declare
        # the same or `doctor` correctly reports a dimension mismatch.
        monkeypatch.setenv("NISABA_EMBED_DIMS", str(EMBED_DIM))

        rc = main(["doctor", "--json"])
        report = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert report["embeddings"]["ok"] is True

    def test_a_missing_reranker_is_a_warning_not_a_failure(self, stub_server, tmp_path, monkeypatch, capsys):
        from nisaba_rag.cli import main

        monkeypatch.setenv("NISABA_DATA_DIR", str(tmp_path / "warn-data"))
        monkeypatch.setenv("NISABA_OLLAMA_URL", stub_server)
        monkeypatch.setenv("NISABA_CONFIG", str(tmp_path / "absent.json"))
        monkeypatch.setenv("NISABA_EMBED_DIMS", str(EMBED_DIM))
        monkeypatch.setenv("NISABA_RERANKER_DIR", str(tmp_path / "no-model"))

        rc = main(["doctor", "--json"])
        report = json.loads(capsys.readouterr().out)

        # Embeddings are the hard requirement; the reranker only degrades quality.
        assert rc == 0, "a clone with no reranker model must still be usable"
        assert report["ok"] is True
        assert report["reranker"]["ok"] is False
        assert report["warnings"]

    def test_fails_when_embeddings_are_unreachable(self, tmp_path, monkeypatch, capsys):
        from nisaba_rag.cli import main

        monkeypatch.setenv("NISABA_DATA_DIR", str(tmp_path / "fail-data"))
        monkeypatch.setenv("NISABA_OLLAMA_URL", f"http://127.0.0.1:{_free_port()}")
        monkeypatch.setenv("NISABA_CONFIG", str(tmp_path / "absent.json"))

        rc = main(["doctor", "--json"])
        report = json.loads(capsys.readouterr().out)

        assert rc == 1
        assert report["ok"] is False


def _ollama_reachable() -> bool:
    """True when a real Ollama answers on the configured base_url."""
    import httpx

    try:
        httpx.get(f"{DEFAULTS['embed']['base_url']}/api/tags", timeout=3.0)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.integration
@pytest.mark.skipif(
    not _ollama_reachable(),
    reason="no Ollama reachable on the configured base_url",
)
class TestAgainstRealOllama:
    """Runs only where a real Ollama is reachable on the configured port."""

    def test_configured_model_returns_an_embedding(self):
        cfg = {**DEFAULTS, "embed": {**DEFAULTS["embed"]}}
        vector = embed("a short probe sentence", cfg)
        assert isinstance(vector, list)
        assert len(vector) > 100, "expected a real embedding, got something suspiciously small"

    def test_dimensions_match_the_config(self):
        cfg = {**DEFAULTS, "embed": {**DEFAULTS["embed"]}}
        report = check_embedding_backend(cfg)
        if report["ok"]:
            assert report["dimensions"] == cfg["embed"]["dims"]

    def test_a_nonexistent_model_is_reported_with_the_pull_command(self):
        cfg = {**DEFAULTS, "embed": {**DEFAULTS["embed"], "model": "definitely-not-a-real-model-xyz"}}
        report = check_embedding_backend(cfg)
        assert report["ok"] is False
        assert report["reason"] == "model not installed"
        assert report["hint"] == "ollama pull definitely-not-a-real-model-xyz"
