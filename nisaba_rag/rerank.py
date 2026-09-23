"""Nisaba RAG — optional cross-encoder reranker (ONNX Runtime).

Reorders retrieval candidates by query-passage relevance. Runs on CPU by
default; a GPU execution provider can be selected when ONNX Runtime was built
with it (``NISABA_RERANKER_PROVIDER``).

The reranker is entirely optional: if no model is present, search keeps
working and simply loses the fine reordering. See ``docs/reranker.md`` for how
to obtain and export a compatible model.

Two implementation details matter for output quality, and both are documented
where they are applied below: pairs are scored one at a time, and the ranking
uses the logit margin rather than the saturated softmax probability.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .config import load_config, resolve_path


class RerankerUnavailable(RuntimeError):
    """Raised when the reranker is requested but no usable model is installed."""


class Reranker:
    """Cross-encoder reranker on top of ONNX Runtime."""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = (cfg or load_config()).get("reranker", {})
        self.model_dir = resolve_path(self.cfg.get("model_dir", "data/reranker/multilingual-cross-encoder"))
        self.onnx_path = self.model_dir / self.cfg.get("onnx_file", "model.onnx")
        self.tokenizer_path = self.model_dir / self.cfg.get("tokenizer_file", "tokenizer.json")
        self.max_length = int(self.cfg.get("max_length", 512))
        self.temperature = float(self.cfg.get("temperature", 1.0))
        self.relevant_logit_index = int(self.cfg.get("relevant_logit_index", 1))

        if not self.onnx_path.is_file():
            raise RerankerUnavailable(
                f"reranker model not found at {self.onnx_path}. "
                "See docs/reranker.md to install one, or set "
                "reranker.enabled=false in config.json to silence this."
            )
        if not self.tokenizer_path.is_file():
            raise RerankerUnavailable(f"tokenizer not found at {self.tokenizer_path}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=session_options,
            providers=self._pick_providers(),
        )
        self.active_provider = self.session.get_providers()[0]

        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        # Truncate to the model maximum. Padding is deliberately not enabled:
        # pairs are scored one at a time, so there is never a batch to pad.
        self.tokenizer.enable_truncation(max_length=self.max_length)

    def _pick_providers(self) -> list[str]:
        """Preferred execution provider, falling back to CPU when unavailable.

        GPU providers are opt-in because some of them are unstable on certain
        driver/runtime combinations, and a native crash in the inference thread
        cannot be caught from Python.
        """
        requested = str(self.cfg.get("provider", "cpu")).strip().lower()
        available = ort.get_available_providers()

        candidates = {
            "dml": "DmlExecutionProvider",
            "directml": "DmlExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "coreml": "CoreMLExecutionProvider",
        }
        preferred = candidates.get(requested)
        if preferred and preferred in available:
            return [preferred, "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def rerank(
        self,
        query: str,
        passages: list[dict],
        top_n: Optional[int] = None,
    ) -> list[dict]:
        """Reorder `passages` by relevance, adding a ``score`` in [0, 1].

        Pairs are scored one at a time on purpose. With variable-size batches,
        padding up to the longest member of the batch contaminates the logits:
        the same pair can score differently depending on which other passages
        accompany it, which is enough to invert the order of close candidates.
        Scoring singly costs almost nothing and makes the ranking independent
        of batch composition.
        """
        if not passages:
            return []

        margins = np.empty(len(passages), dtype=np.float64)
        for i, passage in enumerate(passages):
            margins[i] = self._score_one(query, passage.get("text", ""))

        # Calibration. A 2-class softmax over this kind of checkpoint saturates
        # (0.9990 vs 0.9999), which destroys fine ordering: most chunks collapse
        # into the extremes and the top-1 can be a chunk that never mentions the
        # query. The logit margin (relevant - irrelevant) is monotonic with that
        # probability but keeps its resolution, so it is what we rank on; the
        # sigmoid below only maps it onto a readable 0..1 score.
        scores = 1.0 / (1.0 + np.exp(-margins / max(self.temperature, 1e-6)))

        out: list[dict] = []
        for score, margin, passage in zip(scores, margins, passages):
            item = dict(passage)
            item["score"] = float(score)
            item["rerank_margin"] = float(margin)
            out.append(item)

        out.sort(key=lambda item: item["rerank_margin"], reverse=True)
        return out[:top_n] if top_n else out

    def _score_one(self, query: str, passage: str) -> float:
        """Logit margin (relevant - irrelevant) for a single query-passage pair."""
        encoding = self.tokenizer.encode(query, passage)
        onnx_input = {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        }
        # Some exported graphs declare token_type_ids as a required input;
        # omitting it makes the session fail, so it is always supplied.
        if any(inp.name == "token_type_ids" for inp in self.session.get_inputs()):
            onnx_input["token_type_ids"] = np.array([encoding.type_ids], dtype=np.int64)

        logits = self.session.run(None, onnx_input)[0][0]
        if logits.shape[0] == 1:
            return float(logits[0])
        return float(logits[self.relevant_logit_index] - logits[1 - self.relevant_logit_index])


_reranker: Optional[Reranker] = None


def get_reranker(cfg: Optional[dict] = None) -> Reranker:
    """Return a process-wide singleton (the MCP server is a single process)."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker(cfg)
    return _reranker


def reset_reranker() -> None:
    """Drop the cached instance (used by tests and after a config change)."""
    global _reranker
    _reranker = None
