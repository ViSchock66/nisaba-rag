"""Tests for the reranker's config handling and failure modes.

The model itself is not shipped (it is large and license-encumbered), so these
tests verify the contract around it: a missing model must degrade gracefully
rather than break search.
"""

from __future__ import annotations

import numpy as np
import pytest

from nisaba_rag.rerank import Reranker, RerankerUnavailable, get_reranker, reset_reranker


class TestRerankerAvailability:
    def test_missing_model_raises_a_clear_error(self, tmp_path):
        cfg = {"reranker": {"model_dir": str(tmp_path / "absent"), "onnx_file": "model.onnx"}}
        with pytest.raises(RerankerUnavailable) as excinfo:
            Reranker(cfg)
        assert "not found" in str(excinfo.value)

    def test_tokenizer_is_required_too(self, tmp_path):
        model_dir = tmp_path / "half"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_bytes(b"not a real graph")

        cfg = {"reranker": {"model_dir": str(model_dir), "onnx_file": "model.onnx"}}
        with pytest.raises(RerankerUnavailable) as excinfo:
            Reranker(cfg)
        assert "tokenizer" in str(excinfo.value)

    def test_singleton_is_cached_and_resettable(self, tmp_path, monkeypatch):
        reset_reranker()
        sentinel = object()
        monkeypatch.setattr("nisaba_rag.rerank.Reranker", lambda cfg=None: sentinel)

        assert get_reranker({}) is sentinel
        assert get_reranker({}) is sentinel
        reset_reranker()


class TestProviderSelection:
    @staticmethod
    def _reranker_stub(provider: str):
        instance = Reranker.__new__(Reranker)
        instance.cfg = {"provider": provider}
        return instance

    def test_cpu_is_the_default(self):
        providers = self._reranker_stub("cpu")._pick_providers()
        assert providers == ["CPUExecutionProvider"]

    def test_unknown_provider_falls_back_to_cpu(self):
        assert self._reranker_stub("nonsense")._pick_providers() == ["CPUExecutionProvider"]

    def test_gpu_request_falls_back_when_the_provider_is_missing(self, monkeypatch):
        monkeypatch.setattr(
            "nisaba_rag.rerank.ort.get_available_providers", lambda: ["CPUExecutionProvider"]
        )
        assert self._reranker_stub("cuda")._pick_providers() == ["CPUExecutionProvider"]

    def test_gpu_request_is_honoured_when_available(self, monkeypatch):
        monkeypatch.setattr(
            "nisaba_rag.rerank.ort.get_available_providers",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        assert self._reranker_stub("cuda")._pick_providers() == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]


class TestScoringMath:
    @staticmethod
    def _reranker_with_margins(margins: list[float]):
        """Build a reranker whose scoring is stubbed with fixed margins."""
        instance = Reranker.__new__(Reranker)
        instance.temperature = 1.0
        instance.relevant_logit_index = 1
        queue = iter(margins)
        instance._score_one = lambda query, passage: next(queue)
        return instance

    def test_passages_are_sorted_by_margin(self):
        reranker = self._reranker_with_margins([0.1, 5.0, -2.0])
        passages = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

        ordered = reranker.rerank("q", passages)
        assert [item["id"] for item in ordered] == ["b", "a", "c"]

    def test_margin_is_preserved_alongside_the_score(self):
        reranker = self._reranker_with_margins([3.0])
        item = reranker.rerank("q", [{"id": "a"}])[0]
        assert item["rerank_margin"] == 3.0
        assert 0.0 < item["score"] < 1.0

    def test_top_n_truncates(self):
        reranker = self._reranker_with_margins([1.0, 2.0, 3.0])
        assert len(reranker.rerank("q", [{"id": str(i)} for i in range(3)], top_n=2)) == 2

    def test_empty_input_short_circuits(self):
        reranker = self._reranker_with_margins([])
        assert reranker.rerank("q", []) == []

    def test_sigmoid_is_monotonic_in_the_margin(self):
        margins = [-5.0, -1.0, 0.0, 1.0, 5.0]
        scores = 1.0 / (1.0 + np.exp(-np.array(margins)))
        assert all(a < b for a, b in zip(scores, scores[1:]))

    def test_input_shape_is_one_pair_at_a_time(self):
        """`_score_one` must receive exactly one passage per call."""
        calls: list[str] = []
        instance = Reranker.__new__(Reranker)
        instance.temperature = 1.0
        instance.relevant_logit_index = 1

        def recording_score(query, passage):
            calls.append(passage)
            return 1.0

        instance._score_one = recording_score
        instance.rerank("q", [{"text": "one"}, {"text": "two"}])
        assert calls == ["one", "two"]
