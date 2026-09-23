"""Nisaba RAG — MCP stdio server.

Exposes the engine as MCP tools so any MCP client can index and search a
document folder. The tool surface is deliberately small:

    index_folder / index_file / reindex_folder
    search_documents / get_index_status / list_sources / delete_source
    reranker_status

Run it with ``python -m nisaba_rag.server`` (or the ``nisaba-rag`` console
script). It speaks the stdio transport, so it is meant to be launched *by* an
MCP client, not by hand.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from mcp.server.mcpserver import MCPServer

from .config import load_config
from .indexer import index_path
from .indexer import search_documents as _search_documents
from .rag import RagStore

_HERE = Path(__file__).resolve().parent


def _load_env_file(path: Path) -> None:
    """Load ``KEY=VALUE`` lines from a .env file without python-dotenv.

    Existing environment variables win, so an MCP client can inject values
    through its own server configuration.
    """
    if not path.is_file():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip().strip('"').strip("'")


_load_env_file(Path(os.environ.get("NISABA_ENV_FILE", _HERE.parent / ".env")))

cfg = load_config()
store = RagStore(cfg)
mcp = MCPServer("nisaba")


@mcp.tool()
def index_folder(path: str) -> dict:
    """Index all supported text files under a folder (incremental by content hash)."""
    return index_path(store, cfg, path)


@mcp.tool()
def index_file(path: str) -> dict:
    """Index a single file if its content changed."""
    return index_path(store, cfg, path)


@mcp.tool()
def reindex_folder(path: str) -> dict:
    """Force re-index of every file under a folder, ignoring change detection."""
    return index_path(store, cfg, path, force=True)


@mcp.tool()
def search_documents(
    query: str,
    limit: int = 5,
    source: Optional[str] = None,
    mode: str = "dense",
    rerank: bool = False,
) -> list:
    """Search the indexed documents and return matching chunks with provenance.

    Args:
        query: Natural-language or keyword query.
        limit: Maximum number of chunks to return.
        source: Optional exact file-path filter (only honoured in "dense" mode).
        mode: "dense" (semantic only) or "hybrid" (dense + BM25 fused with RRF).
        rerank: Reorder results with the cross-encoder when one is installed.
    """
    return _search_documents(store, cfg, query, limit=limit, source=source, mode=mode, rerank=rerank)


@mcp.tool()
def get_index_status() -> dict:
    """Index statistics: how many files and chunks are stored."""
    return store.stats()


@mcp.tool()
def list_sources() -> list:
    """List every indexed source path."""
    return store.list_sources()


@mcp.tool()
def delete_source(source: str) -> str:
    """Remove one indexed source (exact file path) from the index."""
    store.remove_file(source)
    return f"removed {source}"


@mcp.tool()
def reranker_status() -> dict:
    """Report whether the optional cross-encoder reranker is usable."""
    try:
        from .rerank import get_reranker

        reranker = get_reranker(cfg)
        return {
            "available": True,
            "provider": reranker.active_provider,
            "model": str(reranker.onnx_path),
        }
    except Exception as exc:  # noqa: BLE001 - status tool must never raise
        return {"available": False, "reason": str(exc)}


@mcp.tool()
def check_backend() -> dict:
    """Check that the embedding server is reachable and serves the configured model.

    Call this first when indexing fails: it tells "the server is down" apart
    from "the server is up but the model was never pulled", and returns the
    exact command needed to fix the latter.
    """
    from .rag import check_embedding_backend

    return check_embedding_backend(cfg)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
