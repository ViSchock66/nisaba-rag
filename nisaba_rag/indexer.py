"""Nisaba RAG — indexing pipeline shared by the MCP server and the CLI.

Walks a path, applies the extension/exclusion/size rules from the config, and
upserts only what changed. Kept separate from ``server.py`` so it can be
imported and tested without starting an MCP transport.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .rag import RagStore, chunk_text, embed_many


def iter_files(root: Path, cfg: dict) -> list[Path]:
    """Expand `root` into the list of candidate files (no filtering yet)."""
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []

    exclude_dirs = set(cfg.get("exclude_dirs", []))
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        files.extend(Path(dirpath) / name for name in filenames)
    return files


def index_path(
    store: RagStore,
    cfg: dict,
    path: str,
    force: bool = False,
    embed_fn=None,
) -> dict:
    """Index every supported file under `path`.

    Incremental by content hash unless ``force`` is set. ``embed_fn`` is
    injectable so the pipeline can be tested without an embedding server; it
    receives a list of chunk texts and returns one vector per chunk.
    """
    embed_fn = embed_fn or embed_many
    root = Path(path).expanduser()
    if not root.exists():
        return {"error": f"path not found: {path}", "indexed": 0, "skipped": 0, "errors": 0, "files": []}

    allowed_exts = {ext.lower() for ext in cfg.get("extensions", [])}
    exclude_files = set(cfg.get("exclude_files", []))
    max_file_kb = cfg.get("max_file_kb", 0)
    chunk_size = cfg["chunk"]["size"]
    chunk_overlap = cfg["chunk"]["overlap"]

    stats: dict = {"indexed": 0, "skipped": 0, "errors": 0, "files": [], "error_details": []}

    for file_path in iter_files(root, cfg):
        if file_path.name in exclude_files or file_path.suffix.lower() not in allowed_exts:
            continue
        try:
            if max_file_kb and file_path.stat().st_size > max_file_kb * 1024:
                stats["skipped"] += 1
                continue
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            stats["errors"] += 1
            stats["error_details"].append(f"{file_path}: {exc}")
            continue

        if not text.strip():
            stats["skipped"] += 1
            continue

        rel = str(file_path)
        content_hash = store.hash_of(text)
        if not force and store.stored_hash(rel) == content_hash:
            stats["skipped"] += 1
            continue

        try:
            chunks = chunk_text(text, chunk_size, chunk_overlap)
            if not chunks:
                stats["skipped"] += 1
                continue
            # One request for the whole file instead of one per chunk.
            embeddings = embed_fn(chunks, cfg)
            # Replace-then-add keeps the chunk numbering consistent when a file
            # shrinks; otherwise stale chunks from the previous version survive.
            store.remove_file(rel)
            store.add_chunks(rel, chunks, embeddings, file_path.suffix.lower())
            store.upsert_file(rel, content_hash, len(chunks))
            stats["indexed"] += 1
            stats["files"].append(rel)
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            stats["error_details"].append(f"{rel}: {exc}")

    return stats


def search_documents(
    store: RagStore,
    cfg: dict,
    query: str,
    limit: int = 5,
    source: Optional[str] = None,
    mode: str = "dense",
    rerank: bool = False,
) -> list[dict]:
    """Search the index in dense or hybrid mode, optionally reranked."""
    if mode == "hybrid":
        from .hybrid import hybrid_search

        return hybrid_search(store, query, cfg, top_n=limit, rerank=rerank)

    from .rag import embed

    query_embedding = embed(query, cfg)
    hits = store.search(query_embedding, limit=limit, source_filter=source)

    if rerank and hits:
        hits = _try_rerank(cfg, query, hits, limit)
    return hits


def _try_rerank(cfg: dict, query: str, hits: list[dict], limit: int) -> list[dict]:
    """Rerank `hits`, falling back to the original order when unavailable.

    Reranking is a quality improvement, never a requirement. If no model is
    installed the search must still return results — failing here would make a
    fresh clone unusable, which is exactly what `reranker.enabled` is meant to
    let people control.
    """
    try:
        from .rerank import get_reranker

        return get_reranker(cfg).rerank(query, hits, top_n=limit)
    except Exception:  # noqa: BLE001 - degrade to the unranked ordering
        return hits
