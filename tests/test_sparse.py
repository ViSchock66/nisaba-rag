"""Tests for the lexical (BM25) index."""

from __future__ import annotations

from nisaba_rag.sparse import SparseIndex, normalize, tokenize


class TestTokenize:
    def test_lowercases_and_strips_accents(self):
        assert tokenize("Dimensión") == ["dimension"]

    def test_removes_stopwords_in_both_languages(self):
        assert "de" not in tokenize("el tamaño de la ventana")
        assert "the" not in tokenize("the size of the window")

    def test_drops_single_character_tokens(self):
        assert "a" not in tokenize("a b cd")

    def test_normalize_is_idempotent(self):
        once = normalize("Configuración")
        assert normalize(once) == once


class TestSparseIndex:
    @staticmethod
    def _index():
        index = SparseIndex()
        index.build([
            {"id": "1", "source": "a.md", "text": "the retry policy uses exponential backoff"},
            {"id": "2", "source": "b.md", "text": "the embedding model has 1024 dimensions"},
            {"id": "3", "source": "c.md", "text": "chunk size and overlap configuration"},
        ])
        return index

    def test_empty_corpus_returns_no_hits(self):
        index = SparseIndex()
        index.build([])
        assert index.search("anything") == []
        assert len(index) == 0

    def test_rare_terms_rank_their_document_first(self):
        hits = self._index().search("exponential backoff")
        assert hits
        assert hits[0]["id"] == "1"

    def test_matching_is_accent_insensitive(self):
        index = SparseIndex()
        index.build([{"id": "1", "source": "a.md", "text": "la configuración del índice"}])
        assert index.search("configuracion")[0]["id"] == "1"

    def test_query_without_content_tokens_returns_nothing(self):
        assert self._index().search("de la el") == []

    def test_limit_is_respected(self):
        index = SparseIndex()
        index.build([
            {"id": str(i), "source": f"{i}.md", "text": "shared keyword here"} for i in range(10)
        ])
        assert len(index.search("shared keyword", k=3)) == 3

    def test_rebuild_replaces_the_previous_corpus(self):
        index = self._index()
        index.build([{"id": "9", "source": "new.md", "text": "totally different"}])
        assert len(index) == 1
        assert index.search("exponential") == []

    def test_results_carry_source_and_score(self):
        hit = self._index().search("embedding")[0]
        assert set(hit) == {"id", "source", "text", "score"}
        assert hit["score"] > 0
