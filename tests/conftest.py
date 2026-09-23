"""Shared pytest fixtures.

No test requires an embedding server, a GPU or any personal document: the
embedding function is stubbed with a deterministic local implementation.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

# Make the package importable when running pytest from a plain checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nisaba_rag.config import DEFAULTS  # noqa: E402
from nisaba_rag.rag import RagStore  # noqa: E402

EMBED_DIM = 64


def fake_embed(texts, cfg: dict | None = None) -> list[list[float]]:
    """Deterministic bag-of-words embeddings for a batch of texts.

    Not semantically meaningful, but stable and cheap: identical text yields
    identical vectors and overlapping vocabulary yields overlapping vectors,
    which is enough to exercise the whole pipeline offline.

    Accepts a single string as well as a list, mirroring `embed_many`, so it
    can stand in for either the real batched call or a single query embedding.
    """
    single = isinstance(texts, str)
    batch = [texts] if single else list(texts)

    out = []
    for text in batch:
        vector = [0.0] * EMBED_DIM
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            vector[digest[0] % EMBED_DIM] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        out.append([value / norm for value in vector])

    return out[0] if single else out


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
    """Config pointing at an isolated temporary data directory."""
    config = {
        **DEFAULTS,
        "store": {"path": str(tmp_path / "chroma"), "collection": "test_documents"},
        "index_db": str(tmp_path / "index.sqlite"),
    }
    return config


@pytest.fixture
def store(cfg: dict):
    """A RagStore backed by a temporary directory."""
    instance = RagStore(cfg)
    yield instance
    instance.close()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A small documentation corpus with a deliberate duplicate file."""
    root = tmp_path / "corpus"
    (root / "nested").mkdir(parents=True)
    (root / "intro.md").write_text(
        "# Introduction\n\nThe retriever combines a dense embedding search with a "
        "sparse BM25 keyword search.\n",
        encoding="utf-8",
    )
    (root / "retry.md").write_text(
        "# Retry policy\n\nA retry policy with exponential backoff mitigates rate limiting.\n",
        encoding="utf-8",
    )
    (root / "nested" / "config.yaml").write_text(
        "chunk:\n  size: 500\n  overlap: 100\n", encoding="utf-8"
    )
    # Should be ignored: unsupported extension.
    (root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    # Should be ignored: excluded directory.
    (root / "node_modules").mkdir()
    (root / "node_modules" / "junk.md").write_text("junk", encoding="utf-8")
    return root
