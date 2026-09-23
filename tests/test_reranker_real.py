"""Tests that exercise the real ONNX reranker end to end.

These are the tests that prove the cross-encoder path actually works, as
opposed to `test_rerank.py`, which only checks the contract around it.

They need a real model on disk (an ONNX graph plus its tokenizer), so they are
**skipped** when none is configured. Point them at one with:

    NISABA_TEST_RERANKER_DIR=/path/to/model pytest tests/test_reranker_real.py

or by placing a model where `config.json` expects it.

The model file is large and separately licensed, so it is never committed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nisaba_rag.config import load_config, resolve_path
from nisaba_rag.rerank import Reranker, RerankerUnavailable


def _model_dir() -> Path:
    override = os.environ.get("NISABA_TEST_RERANKER_DIR")
    if override:
        return Path(override).expanduser()
    return resolve_path(load_config()["reranker"]["model_dir"])


def _has_model(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    onnx_files = list(model_dir.glob("*.onnx"))
    return bool(onnx_files) and (model_dir / "tokenizer.json").is_file()


MODEL_DIR = _model_dir()

pytestmark = pytest.mark.skipif(
    not _has_model(MODEL_DIR),
    reason=f"no reranker model at {MODEL_DIR} (set NISABA_TEST_RERANKER_DIR to enable)",
)


@pytest.fixture(scope="module")
def reranker() -> Reranker:
    """Load the model once; it is not cheap to initialise."""
    onnx_file = next(iter(sorted(MODEL_DIR.glob("*.onnx")))).name
    return Reranker({
        "reranker": {
            "enabled": True,
            "model_dir": str(MODEL_DIR),
            "onnx_file": onnx_file,
            "tokenizer_file": "tokenizer.json",
            "max_length": 512,
            "provider": "cpu",
        }
    })


class TestRealInference:
    def test_model_loads_onto_an_execution_provider(self, reranker):
        assert reranker.active_provider
        assert "ExecutionProvider" in reranker.active_provider

    def test_query_passage_pair_produces_a_margin(self, reranker):
        margin = reranker._score_one(
            "how is the chunk size configured",
            "The chunk size is 500 characters with an overlap of 100.",
        )
        assert isinstance(margin, float)
        assert margin != 0.0

    def test_relevant_passage_outscores_an_unrelated_one(self, reranker):
        """The whole point of the reranker: separate signal from noise."""
        query = "how is the chunk size and overlap configured"

        relevant = reranker._score_one(
            query, "The chunk size is 500 characters and the overlap is 100."
        )
        irrelevant = reranker._score_one(
            query, "The colour palette uses teal and amber for the buttons."
        )

        assert relevant > irrelevant, f"relevant={relevant} irrelevant={irrelevant}"

    def test_ranking_puts_the_relevant_passage_first(self, reranker):
        query = "how is the chunk size and overlap configured"
        candidates = [
            {"id": "c", "source": "colours.md",
             "text": "The colour palette uses teal and amber for the buttons."},
            {"id": "a", "source": "chunking.md",
             "text": "The chunk size is 500 characters and the overlap is 100."},
            {"id": "b", "source": "networking.md",
             "text": "The network switch manages VLANs and inter-subnet routing."},
        ]

        ranked = reranker.rerank(query, candidates)

        assert [item["id"] for item in ranked] == ["a", "c", "b"] or ranked[0]["id"] == "a"
        assert ranked[0]["source"] == "chunking.md"

    def test_scores_are_in_the_unit_interval(self, reranker):
        candidates = [
            {"id": str(i), "text": text}
            for i, text in enumerate([
                "The chunk size is 500 characters.",
                "Unrelated content about colours.",
                "Another passage mentioning overlap and chunking.",
            ])
        ]
        for item in reranker.rerank("chunk size overlap", candidates):
            assert 0.0 <= item["score"] <= 1.0

    def test_margins_are_ordered_like_the_scores(self, reranker):
        candidates = [
            {"id": "1", "text": "The chunk size is 500 characters with overlap 100."},
            {"id": "2", "text": "A completely unrelated sentence about furniture."},
        ]
        ranked = reranker.rerank("chunk size overlap", candidates)
        margins = [item["rerank_margin"] for item in ranked]
        assert margins == sorted(margins, reverse=True)

    def test_scoring_is_independent_of_batch_composition(self, reranker):
        """Same pair, different companions — the margin must not move.

        This is why pairs are scored one at a time instead of in a padded
        batch: padding shifts the logits and can invert close candidates.
        """
        query = "how is the chunk size configured"
        target = "The chunk size is 500 characters with an overlap of 100."

        alone = reranker._score_one(query, target)
        with_short = reranker._score_one(query, target)
        with_long_companions = reranker.rerank(query, [
            {"id": "x", "text": "short"},
            {"id": "target", "text": target},
            {"id": "y", "text": "a much longer passage " * 50},
        ])
        target_margin = next(i["rerank_margin"] for i in with_long_companions if i["id"] == "target")

        assert alone == pytest.approx(with_short, abs=1e-6)
        assert alone == pytest.approx(target_margin, abs=1e-6)

    def test_empty_passage_list_is_safe(self, reranker):
        assert reranker.rerank("anything", []) == []

    def test_top_n_limits_the_output(self, reranker):
        candidates = [{"id": str(i), "text": f"passage number {i}"} for i in range(5)]
        assert len(reranker.rerank("passage", candidates, top_n=2)) == 2


class TestRealModelMissing:
    def test_missing_model_raises_with_actionable_text(self, tmp_path):
        with pytest.raises(RerankerUnavailable) as excinfo:
            Reranker({"reranker": {"model_dir": str(tmp_path / "absent"), "onnx_file": "model.onnx"}})
        message = str(excinfo.value)
        assert "reranker model not found" in message
        assert "docs/reranker.md" in message
