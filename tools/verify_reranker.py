"""Verify the engine against a real reranker model and a real embedding server.

Unlike `tools/verify.py` (which stubs embeddings and needs no model), this
script exercises the actual ONNX cross-encoder path. It is the check that
proves hybrid search + reranking works end to end on this machine.

Usage:
    # Against a model you already have:
    python tools/verify_reranker.py --model-dir "D:/models/my-cross-encoder"

    # Or let it read reranker.model_dir from config.json:
    python tools/verify_reranker.py

Exit codes: 0 all checks passed, 1 a check failed, 2 no model was found.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nisaba_rag.config import DEFAULTS, load_config, resolve_path  # noqa: E402
from nisaba_rag.rag import RagStore  # noqa: E402
from nisaba_rag.rerank import Reranker, RerankerUnavailable  # noqa: E402

PASSED = 0
FAILED: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
        print(f"  ok    {label}")
    else:
        FAILED.append(f"{label}{f' -- {detail}' if detail else ''}")
        print(f"  FAIL  {label}{f' -- {detail}' if detail else ''}")


def fake_embed(text: str, cfg: dict | None = None) -> list[float]:
    """Deterministic offline embedder, used only to seed candidate passages.

    The reranker is a cross-encoder: it reads the query and the passage itself,
    so its quality does not depend on how the candidates were retrieved. A
    stub embedder keeps this script from needing an embedding server while
    still exercising 100% of the real reranker code path.
    """
    vector = [0.0] * 64
    for word in text.lower().split():
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        vector[digest[0] % 64] += 1.0
    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    return [v / norm for v in vector]


CORPUS = [
    ("chunking.md",
     "El chunk size es de 500 caracteres y el overlap de 100; asi se parte el "
     "texto en fragmentos solapados que conservan el contexto."),
    ("colores.md",
     "La paleta de colores usa teal y ambar para los botones de la interfaz de "
     "usuario y los estados de hover."),
    ("redes.md",
     "El switch de red administra las VLAN y el enrutamiento entre subredes, "
     "con politicas de calidad de servicio."),
    ("facturacion.md",
     "El modulo de facturacion emite boletas y facturas electronicas contra el "
     "servicio de impuestos."),
    ("embeddings.md",
     "Los embeddings se generan con un modelo local y se almacenan como "
     "vectores normalizados en espacio coseno."),
]

QUERY = "como se calcula el chunk size y el overlap del indice"
EXPECTED = "chunking.md"


def find_model_dir(explicit: str | None) -> Path | None:
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_dir():
            return candidate
        print(f"model dir not found: {candidate}")
        return None

    configured = resolve_path(load_config()["reranker"]["model_dir"])
    if configured.is_dir():
        return configured
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the real ONNX reranker path.")
    parser.add_argument("--model-dir", default=None, help="Folder with the .onnx and tokenizer.json.")
    parser.add_argument("--provider", default="cpu", help="Execution provider: cpu, cuda, dml, coreml.")
    args = parser.parse_args()

    model_dir = find_model_dir(args.model_dir)
    if model_dir is None:
        print(
            "No reranker model found.\n"
            "Pass --model-dir, set reranker.model_dir in config.json, or follow docs/reranker.md."
        )
        return 2

    onnx_files = sorted(model_dir.glob("*.onnx"))
    if not onnx_files:
        print(f"No .onnx file in {model_dir}")
        return 2
    if not (model_dir / "tokenizer.json").is_file():
        print(f"No tokenizer.json in {model_dir}")
        return 2

    print(f"model dir : {model_dir}")
    print(f"graph     : {onnx_files[0].name} ({onnx_files[0].stat().st_size / 1e6:.1f} MB)")
    print(f"provider  : {args.provider}\n")

    cfg = {
        "reranker": {
            "enabled": True,
            "model_dir": str(model_dir),
            "onnx_file": onnx_files[0].name,
            "tokenizer_file": "tokenizer.json",
            "max_length": 512,
            "provider": args.provider,
        }
    }

    print("-- loading --")
    try:
        reranker = Reranker(cfg)
    except RerankerUnavailable as exc:
        print(f"  FAIL  could not load: {exc}")
        return 1
    check("model loads", True)
    print(f"        active provider: {reranker.active_provider}")
    check("an execution provider is active", bool(reranker.active_provider))

    print("\n-- discrimination --")
    relevant = reranker._score_one(QUERY, CORPUS[0][1])
    irrelevant = reranker._score_one(QUERY, CORPUS[2][1])
    print(f"        relevant margin   : {relevant:+.3f}")
    print(f"        irrelevant margin : {irrelevant:+.3f}")
    check("relevant passage outscores an unrelated one", relevant > irrelevant)

    print("\n-- latency --")
    samples = [
        reranker._score_one(QUERY, text) for _, text in CORPUS
    ] * 4
    start = time.perf_counter()
    for _, text in CORPUS:
        reranker._score_one(QUERY, text)
    elapsed = (time.perf_counter() - start) / len(CORPUS) * 1000
    print(f"        ~{elapsed:.0f} ms per pair ({len(samples)} pairs scored)")
    check("scoring is under 2 s per pair", elapsed < 2000)

    print("\n-- scoring is batch-independent --")
    alone = reranker._score_one(QUERY, CORPUS[0][1])
    ranked = reranker.rerank(QUERY, [
        {"id": "short", "text": "corto"},
        {"id": "target", "text": CORPUS[0][1]},
        {"id": "long", "text": "relleno largo " * 60},
    ])
    in_batch = next(i["rerank_margin"] for i in ranked if i["id"] == "target")
    print(f"        alone   : {alone:+.6f}")
    print(f"        in batch: {in_batch:+.6f}")
    check("same pair scores identically regardless of companions", abs(alone - in_batch) < 1e-5)

    print("\n-- full pipeline (dense -> rerank) --")
    scratch = Path(os.environ.get(
        "NISABA_VERIFY_TMP",
        Path(__file__).resolve().parent.parent / ".verify-tmp" / "reranker",
    ))
    scratch.mkdir(parents=True, exist_ok=True)

    store_cfg = {
        **DEFAULTS,
        "store": {"path": str(scratch / "chroma"), "collection": "reranker_verify"},
        "index_db": str(scratch / "index.sqlite"),
        **cfg,
    }
    store = RagStore(store_cfg)
    try:
        for source, text in CORPUS:
            store.add_chunks(source, [text], [fake_embed(text)], ".md")

        candidates = store.search(fake_embed(QUERY), limit=5)
        print("        dense order:")
        for hit in candidates:
            print(f"          sim={hit['similarity']:.3f}  {hit['source']}")

        ranked_hits = reranker.rerank(QUERY, candidates, top_n=5)
        print("        after rerank:")
        for hit in ranked_hits:
            print(f"          margin={hit['rerank_margin']:+.3f}  score={hit['score']:.3f}  {hit['source']}")

        check("rerank returns results", len(ranked_hits) > 0)
        check(f"top-1 after rerank is {EXPECTED}", ranked_hits[0]["source"] == EXPECTED,
              f"got {ranked_hits[0]['source']}")
        check("all scores within [0, 1]",
              all(0.0 <= hit["score"] <= 1.0 for hit in ranked_hits))
        check("margins are sorted descending",
              [h["rerank_margin"] for h in ranked_hits]
              == sorted([h["rerank_margin"] for h in ranked_hits], reverse=True))
    finally:
        store.reset()
        store.close()
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"\n{PASSED} checks passed, {len(FAILED)} failed")
    for failure in FAILED:
        print(f"  FAIL  {failure}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
