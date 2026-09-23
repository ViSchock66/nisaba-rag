"""Unit tests for chunking, config resolution and path portability."""

from __future__ import annotations

import json

import pytest

from nisaba_rag.config import (
    DEFAULTS,
    load_config,
    resolve_path,
    validate_collection_name,
)
from nisaba_rag.rag import chunk_text


class TestChunking:
    def test_empty_text_yields_nothing(self):
        assert chunk_text("") == []

    def test_short_text_is_a_single_chunk(self):
        assert chunk_text("short document", size=500, overlap=100) == ["short document"]

    def test_chunks_overlap_by_the_configured_amount(self):
        text = "abcdefghij" * 20  # 200 chars
        chunks = chunk_text(text, size=50, overlap=10)
        assert len(chunks) > 1
        # Step is size - overlap = 40, so consecutive chunks share 10 chars.
        assert chunks[0][-10:] == chunks[1][:10]

    def test_every_character_is_covered(self):
        text = "".join(str(i % 10) for i in range(1000))
        chunks = chunk_text(text, size=100, overlap=20)
        assert chunks[0].startswith(text[0])
        assert chunks[-1].endswith(text[-1])
        assert "".join(chunks).count(text[:10]) >= 1

    def test_overlap_larger_than_size_is_clamped(self):
        # Must not loop forever or produce empty chunks.
        chunks = chunk_text("abcdefghij", size=10, overlap=999)
        assert chunks
        assert all(chunks)

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            chunk_text("abc", size=0)


class TestConfig:
    def test_defaults_contain_no_absolute_personal_path(self):
        assert DEFAULTS["store"]["path"] == "data/chroma"
        assert DEFAULTS["index_db"] == "data/index.sqlite"
        assert DEFAULTS["reranker"]["model_dir"].startswith("data/")

    def test_reranker_is_on_by_default(self):
        """Reranking is a headline feature, so a default install must not
        silently disable it; the model is the only thing that varies."""
        assert DEFAULTS["reranker"]["enabled"] is True

    def test_relative_paths_resolve_under_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NISABA_HOME", str(tmp_path))
        assert resolve_path("data/chroma") == tmp_path / "data" / "chroma"

    def test_absolute_paths_are_left_alone(self, tmp_path):
        target = tmp_path / "elsewhere"
        assert resolve_path(target) == target

    def test_user_config_file_is_deep_merged(self, tmp_path, monkeypatch):
        user_config = tmp_path / "config.json"
        user_config.write_text(
            json.dumps({"chunk": {"size": 200}, "embed": {"model": "custom-model"}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("NISABA_CONFIG", str(user_config))
        config = load_config()

        assert config["chunk"]["size"] == 200
        # Untouched keys survive the merge.
        assert config["chunk"]["overlap"] == DEFAULTS["chunk"]["overlap"]
        assert config["embed"]["model"] == "custom-model"
        assert config["store"]["collection"] == DEFAULTS["store"]["collection"]

    def test_env_overrides_win_over_the_config_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NISABA_CONFIG", str(tmp_path / "missing.json"))
        monkeypatch.setenv("NISABA_EMBED_MODEL", "from-env")
        monkeypatch.setenv("NISABA_DATA_DIR", str(tmp_path / "custom-data"))
        config = load_config()

        assert config["embed"]["model"] == "from-env"
        assert config["index_db"].endswith("index.sqlite")
        assert "custom-data" in config["index_db"]

    def test_legacy_ollama_base_url_key_is_honoured(self, tmp_path, monkeypatch):
        user_config = tmp_path / "config.json"
        user_config.write_text(
            json.dumps({"embed": {"ollama_base_url": "http://127.0.0.1:11500"}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("NISABA_CONFIG", str(user_config))
        assert load_config()["embed"]["base_url"] == "http://127.0.0.1:11500"


class TestCollectionNameValidation:
    """The vector store rejects malformed names with an opaque error, so the
    check happens up front with a message that says what is actually wrong."""

    @pytest.mark.parametrize("name", ["documents", "my_docs", "docs.v2", "a-b-c", "abc"])
    def test_valid_names_pass_through(self, name):
        assert validate_collection_name(name) == name

    @pytest.mark.parametrize("name", ["", "ab", "docs!", "-docs", "docs-", "a" * 513])
    def test_invalid_names_raise(self, name):
        with pytest.raises(ValueError, match="invalid store.collection"):
            validate_collection_name(name)

    def test_store_rejects_an_invalid_name_at_construction(self, cfg, tmp_path):
        from nisaba_rag.rag import RagStore

        bad = {**cfg, "store": {"path": str(tmp_path / "chroma"), "collection": "x"}}
        with pytest.raises(ValueError, match="invalid store.collection"):
            RagStore(bad)
