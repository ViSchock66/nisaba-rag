"""End-to-end tests: build an index from scratch and query it.

These prove the pipeline works with no external service, no GPU and no user
data — the same guarantee a CI runner gets.
"""

from __future__ import annotations

import nisaba_rag.rag as rag
from nisaba_rag.indexer import index_path, search_documents

from .conftest import fake_embed


def _build(store, cfg, corpus, monkeypatch):
    monkeypatch.setattr(rag, "embed", fake_embed)
    monkeypatch.setattr(rag, "embed_many", fake_embed)
    monkeypatch.setattr("nisaba_rag.indexer.embed_many", fake_embed)
    return index_path(store, cfg, str(corpus), embed_fn=fake_embed)


class TestEndToEnd:
    def test_dense_search_finds_the_right_document(self, store, cfg, corpus, monkeypatch):
        _build(store, cfg, corpus, monkeypatch)

        hits = search_documents(store, cfg, "retry policy backoff", limit=2, mode="dense")
        assert hits
        assert hits[0]["source"].endswith("retry.md")

    def test_hybrid_search_finds_the_right_document(self, store, cfg, corpus, monkeypatch):
        _build(store, cfg, corpus, monkeypatch)

        hits = search_documents(store, cfg, "chunk overlap", limit=2, mode="hybrid")
        assert hits
        assert hits[0]["source"].endswith("config.yaml")

    def test_results_expose_provenance(self, store, cfg, corpus, monkeypatch):
        _build(store, cfg, corpus, monkeypatch)

        hit = search_documents(store, cfg, "retriever", limit=1)[0]
        assert {"id", "text", "source", "chunk", "distance", "similarity"} <= set(hit)

    def test_deleting_a_source_removes_it_from_search(self, store, cfg, corpus, monkeypatch):
        _build(store, cfg, corpus, monkeypatch)
        target = str(corpus / "retry.md")

        store.remove_file(target)
        hits = search_documents(store, cfg, "retry policy backoff", limit=5, mode="hybrid")
        assert all(not (hit["source"] or "").endswith("retry.md") for hit in hits)

    def test_search_on_empty_index_is_safe(self, store, cfg, monkeypatch):
        monkeypatch.setattr(rag, "embed", fake_embed)
        monkeypatch.setattr(rag, "embed_many", fake_embed)
        assert search_documents(store, cfg, "nothing indexed yet", limit=5) == []

    def test_rerank_flag_degrades_gracefully_without_a_model(self, store, cfg, corpus, monkeypatch):
        """With no reranker installed, search must still return results."""
        _build(store, cfg, corpus, monkeypatch)

        from nisaba_rag import rerank as rerank_module

        rerank_module.reset_reranker()
        hits = search_documents(store, cfg, "retry policy", limit=2, mode="dense", rerank=True)
        assert hits
        rerank_module.reset_reranker()
