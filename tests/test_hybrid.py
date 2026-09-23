"""Tests for RRF fusion and hybrid retrieval (offline, with a stubbed embedder)."""

from __future__ import annotations

import nisaba_rag.hybrid as hybrid
import nisaba_rag.rag as rag

from .conftest import fake_embed


class TestRrfFusion:
    def test_empty_inputs_yield_nothing(self):
        assert hybrid.rrf_fuse([], []) == []

    def test_documents_found_by_both_runs_are_reinforced(self):
        dense = [{"id": "a", "source": "a.md"}, {"id": "b", "source": "b.md"}]
        sparse = [{"id": "b", "source": "b.md"}, {"id": "c", "source": "c.md"}]

        fused = hybrid.rrf_fuse(dense, sparse, top_n=3)
        # "b" is ranked by both runs, so it beats "a" and "c".
        assert fused[0]["id"] == "b"

    def test_scores_follow_the_rrf_formula(self):
        fused = hybrid.rrf_fuse([{"id": "a", "source": "a.md"}], [], top_n=1)
        assert fused[0]["rrf_score"] == 1.0 / (hybrid.RRF_K + 1)

    def test_top_n_truncates(self):
        dense = [{"id": str(i), "source": f"{i}.md"} for i in range(10)]
        assert len(hybrid.rrf_fuse(dense, [], top_n=4)) == 4

    def test_hits_without_an_id_are_ignored(self):
        assert hybrid.rrf_fuse([{"source": "x.md"}], []) == []

    def test_metadata_survives_the_fusion(self):
        dense = [{"id": "a", "source": "a.md", "text": "hello", "similarity": 0.9}]
        sparse = [{"id": "a", "source": "a.md", "text": "hello", "score": 3.2}]

        fused = hybrid.rrf_fuse(dense, sparse, top_n=1)[0]
        assert fused["text"] == "hello"
        assert fused["source"] == "a.md"


class TestHybridSearch:
    def test_hybrid_returns_the_lexically_matching_document(self, store, cfg, monkeypatch):
        monkeypatch.setattr(rag, "embed", fake_embed)

        store.add_chunks("retry.md", ["retry policy with exponential backoff"],
                         [fake_embed("retry policy with exponential backoff")], ".md")
        store.add_chunks("colors.md", ["the palette uses teal and amber"],
                         [fake_embed("the palette uses teal and amber")], ".md")

        hits = hybrid.hybrid_search(store, "exponential backoff", cfg, top_n=2)
        assert hits
        assert hits[0]["source"] == "retry.md"
        assert "rrf_score" in hits[0]

    def test_hybrid_without_rerank_still_returns_results(self, store, cfg, monkeypatch):
        monkeypatch.setattr(rag, "embed", fake_embed)
        store.add_chunks("a.md", ["alpha beta"], [fake_embed("alpha beta")], ".md")

        hits = hybrid.hybrid_search(store, "alpha", cfg, top_n=5, rerank=False)
        assert len(hits) == 1

    def test_hybrid_on_empty_store_returns_nothing(self, store, cfg, monkeypatch):
        monkeypatch.setattr(rag, "embed", fake_embed)
        assert hybrid.hybrid_search(store, "anything", cfg, top_n=5) == []
