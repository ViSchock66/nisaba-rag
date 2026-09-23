"""Tests for the indexing pipeline and the store (fully offline)."""

from __future__ import annotations

from nisaba_rag.indexer import index_path, iter_files
from nisaba_rag.rag import RagStore, chunk_text

from .conftest import fake_embed


class TestIndexing:
    def test_supported_files_are_indexed_and_junk_is_ignored(self, store, cfg, corpus):
        result = index_path(store, cfg, str(corpus), embed_fn=fake_embed)

        assert result["indexed"] == 3
        names = sorted(p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in result["files"])
        assert names == ["config.yaml", "intro.md", "retry.md"]

    def test_excluded_directories_are_skipped(self, cfg, corpus):
        files = [str(p) for p in iter_files(corpus, cfg)]
        assert not any("node_modules" in path for path in files)

    def test_second_run_is_incremental(self, store, cfg, corpus):
        first = index_path(store, cfg, str(corpus), embed_fn=fake_embed)
        second = index_path(store, cfg, str(corpus), embed_fn=fake_embed)

        assert first["indexed"] == 3
        assert second["indexed"] == 0
        assert second["skipped"] == 3

    def test_changed_file_is_reindexed(self, store, cfg, corpus):
        index_path(store, cfg, str(corpus), embed_fn=fake_embed)
        target = corpus / "retry.md"
        target.write_text("# Retry policy\n\nNow with jitter and a circuit breaker.\n", encoding="utf-8")

        result = index_path(store, cfg, str(corpus), embed_fn=fake_embed)
        assert result["indexed"] == 1
        assert result["files"] == [str(target)]

    def test_force_reindexes_everything(self, store, cfg, corpus):
        index_path(store, cfg, str(corpus), embed_fn=fake_embed)
        assert index_path(store, cfg, str(corpus), force=True, embed_fn=fake_embed)["indexed"] == 3

    def test_rshrinking_file_leaves_no_stale_chunks(self, store, cfg, corpus):
        big = corpus / "big.md"
        big.write_text("word " * 2000, encoding="utf-8")
        index_path(store, cfg, str(big), embed_fn=fake_embed)
        before = store.stats()["chunks"]
        assert before > 1

        big.write_text("word", encoding="utf-8")
        index_path(store, cfg, str(big), embed_fn=fake_embed)

        assert store.stats()["chunks"] == 1

    def test_indexing_a_missing_path_reports_an_error(self, store, cfg, tmp_path):
        result = index_path(store, cfg, str(tmp_path / "nope"), embed_fn=fake_embed)
        assert "error" in result
        assert result["indexed"] == 0

    def test_oversized_files_are_skipped(self, store, cfg, corpus):
        cfg = {**cfg, "max_file_kb": 0}  # disabled -> indexed
        assert index_path(store, cfg, str(corpus / "intro.md"), embed_fn=fake_embed)["indexed"] == 1

        store.reset()
        cfg = {**cfg, "max_file_kb": 0.0001}
        assert index_path(store, cfg, str(corpus / "intro.md"), embed_fn=fake_embed)["indexed"] == 0

    def test_embedding_failure_is_counted_not_raised(self, store, cfg, corpus):
        def broken_embed(text, config):
            raise RuntimeError("embedding server down")

        result = index_path(store, cfg, str(corpus), embed_fn=broken_embed)
        assert result["errors"] == 3
        assert result["indexed"] == 0


class TestStore:
    def test_stats_start_empty(self, store):
        assert store.stats() == {"files": 0, "chunks": 0}

    def test_hash_roundtrip(self, store):
        assert store.stored_hash("/tmp/a.md") is None
        store.upsert_file("/tmp/a.md", "abc", 2)
        assert store.stored_hash("/tmp/a.md") == "abc"

    def test_chunk_ids_are_stable_and_unique(self):
        assert RagStore.chunk_id("a.md", 0) == RagStore.chunk_id("a.md", 0)
        assert RagStore.chunk_id("a.md", 0) != RagStore.chunk_id("a.md", 1)
        assert RagStore.chunk_id("a.md", 0) != RagStore.chunk_id("b.md", 0)

    def test_add_and_search_roundtrip(self, store):
        texts = chunk_text("retry policy exponential backoff " * 20, size=100, overlap=20)
        store.add_chunks("retry.md", texts, [fake_embed(t) for t in texts], ".md")

        hits = store.search(fake_embed("retry policy exponential backoff"), limit=3)
        assert hits
        assert hits[0]["source"] == "retry.md"
        assert 0.0 <= hits[0]["similarity"] <= 1.0

    def test_source_filter_restricts_results(self, store):
        store.add_chunks("a.md", ["alpha content"], [fake_embed("alpha content")], ".md")
        store.add_chunks("b.md", ["beta content"], [fake_embed("beta content")], ".md")

        hits = store.search(fake_embed("content"), limit=5, source_filter="b.md")
        assert [hit["source"] for hit in hits] == ["b.md"]

    def test_remove_file_clears_vectors_hashes_and_sparse(self, store):
        store.add_chunks("a.md", ["alpha content"], [fake_embed("alpha content")], ".md")
        store.upsert_file("a.md", "hash", 1)
        assert store.sparse_search("alpha")

        store.remove_file("a.md")

        assert store.stats() == {"files": 0, "chunks": 0}
        assert store.list_sources() == []
        assert store.sparse_search("alpha") == []

    def test_list_sources_is_sorted(self, store):
        for name in ("c.md", "a.md", "b.md"):
            store.upsert_file(name, "h", 1)
        assert store.list_sources() == ["a.md", "b.md", "c.md"]

    def test_limit_larger_than_the_corpus_is_safe(self, store):
        """Asking for more results than exist must not error."""
        store.add_chunks("a.md", ["only chunk"], [fake_embed("only chunk")], ".md")
        hits = store.search(fake_embed("only chunk"), limit=50)
        assert len(hits) == 1

    def test_reset_empties_the_store(self, store):
        store.add_chunks("a.md", ["alpha"], [fake_embed("alpha")], ".md")
        store.upsert_file("a.md", "h", 1)
        store.reset()
        assert store.stats() == {"files": 0, "chunks": 0}

    def test_sparse_index_is_rebuilt_after_a_mutation(self, store):
        store.add_chunks("a.md", ["unique-keyword-here"], [fake_embed("unique-keyword-here")], ".md")
        assert store.sparse_search("unique")
        store.add_chunks("b.md", ["another-token"], [fake_embed("another-token")], ".md")
        assert store.sparse_search("another")[0]["source"] == "b.md"

    def test_data_is_persisted_between_instances(self, cfg):
        first = RagStore(cfg)
        first.add_chunks("a.md", ["persisted"], [fake_embed("persisted")], ".md")
        first.upsert_file("a.md", "h", 1)
        first.close()

        second = RagStore(cfg)
        try:
            assert second.stats() == {"files": 1, "chunks": 1}
            assert second.stored_hash("a.md") == "h"
        finally:
            second.close()
