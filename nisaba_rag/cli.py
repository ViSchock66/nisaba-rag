"""Nisaba RAG — command line interface.

Useful for indexing from a terminal or a scheduled task without wiring up an
MCP client, and for verifying an install end to end.

    nisaba-rag index ./docs
    nisaba-rag index ./docs --force
    nisaba-rag search "how does the retry policy work" --mode hybrid
    nisaba-rag status
    nisaba-rag sources
    nisaba-rag delete ./docs/old.md
    nisaba-rag serve
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_config, resolve_path
from .indexer import index_path
from .indexer import search_documents as _search_documents
from .rag import RagStore


def _configure_stdio() -> None:
    """Make stdout/stderr survive arbitrary document text.

    Indexed text can contain any Unicode character. On Windows the console
    defaults to a legacy code page (cp1252 and friends), so printing a
    character outside it raises UnicodeEncodeError and kills the command —
    a BOM, an emoji or a CJK glyph in one document is enough. Reconfiguring
    both streams to UTF-8 with a replacement fallback keeps output readable
    instead of crashing.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # Stream may be redirected or already detached; printing still
                # works with whatever encoding it has.
                pass


def _build_parser() -> argparse.ArgumentParser:
    # `--json` is accepted both before and after the subcommand. It cannot be
    # declared once and inherited, because argparse lets the subparser shadow
    # the parent namespace. Instead each subparser declares its own copy with
    # SUPPRESS as the default: when the flag is absent, the subparser writes
    # nothing and the parent's value survives; when present, it wins.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="Emit machine-readable JSON.")

    parser = argparse.ArgumentParser(prog="nisaba-rag", description="Local RAG over your documents.")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Emit machine-readable JSON.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", parents=[common], help="Index a file or folder.")
    p_index.add_argument("path")
    p_index.add_argument("--force", action="store_true", help="Ignore content-hash change detection.")

    p_search = sub.add_parser("search", parents=[common], help="Search the index.")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=5)
    p_search.add_argument("--mode", choices=["dense", "hybrid"], default="dense")
    p_search.add_argument("--source", default=None, help="Exact file-path filter (dense mode only).")
    p_search.add_argument("--rerank", action="store_true")

    sub.add_parser("status", parents=[common], help="Show index statistics.")
    sub.add_parser("sources", parents=[common], help="List indexed source paths.")
    sub.add_parser("doctor", parents=[common],
        help="Check that the embedding backend and reranker are usable before indexing.",
    )

    p_delete = sub.add_parser("delete", parents=[common], help="Remove one source from the index.")
    p_delete.add_argument("source")

    sub.add_parser("serve", parents=[common], help="Run the MCP stdio server.")
    return parser


def _doctor(cfg: dict, as_json: bool = False) -> int:
    """Check every external dependency and print an actionable report.

    This is the first command a new user should run: it separates "the server
    is not up" from "the server is up but the model was never pulled", which
    are the two failures that look identical from a distance.
    """
    from .rag import check_embedding_backend

    backend = check_embedding_backend(cfg)

    reranker_cfg = cfg.get("reranker", {})
    reranker_enabled = bool(reranker_cfg.get("enabled"))
    reranker: dict = {"enabled": reranker_enabled}
    if reranker_enabled:
        try:
            from .rerank import get_reranker

            instance = get_reranker(cfg)
            reranker.update({
                "ok": True,
                "provider": instance.active_provider,
                "model": str(instance.onnx_path),
            })
        except Exception as exc:  # noqa: BLE001
            reranker.update({
                "ok": False,
                "reason": str(exc),
                "hint": "install a model (docs/reranker.md) or set reranker.enabled=false",
            })
    else:
        reranker["ok"] = True
        reranker["note"] = "disabled; hybrid search runs without reranking"

    report = {
        "embeddings": backend,
        "reranker": reranker,
        # Embeddings are the hard requirement: without them nothing indexes.
        # A missing reranker degrades quality but leaves the engine usable,
        # so it is reported as a warning rather than a failure.
        "ok": bool(backend.get("ok")),
        "warnings": [] if reranker.get("ok") else ["reranker is enabled but not usable"],
    }

    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ok"] else 1

    print(f"embeddings : {backend['base_url']}  model={backend['model']}")
    if backend.get("ok"):
        print(f"             OK - {backend['dimensions']} dimensions")
    else:
        print(f"             FAIL - {backend.get('reason')}")
        if backend.get("detail"):
            print(f"             detail: {backend['detail']}")
        if backend.get("hint"):
            print(f"             fix: {backend['hint']}")

    print(f"reranker   : {'enabled' if reranker_enabled else 'disabled'}")
    if reranker.get("ok") and reranker.get("provider"):
        print(f"             OK - {reranker['provider']}")
    elif not reranker.get("ok"):
        print(f"             WARN - {reranker.get('reason')}")
        if reranker.get("hint"):
            print(f"             fix: {reranker['hint']}")
    elif reranker.get("note"):
        print(f"             {reranker['note']}")

    print()
    if not report["ok"]:
        print("not ready - fix the embeddings failure above")
    elif report["warnings"]:
        print("ready to index (with warnings above)")
    else:
        print("all good - ready to index")
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = _build_parser().parse_args(argv)
    # `--json` may arrive from either position; the subparser default is False,
    # so OR-ing both copies preserves whichever one the user supplied.
    args.json = bool(getattr(args, "json", False))

    if args.command == "serve":
        from .server import main as serve

        serve()
        return 0

    cfg = load_config()
    store = RagStore(cfg)

    try:
        if args.command == "index":
            result = index_path(store, cfg, args.path, force=args.force)
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                if "error" in result:
                    print(f"error: {result['error']}", file=sys.stderr)
                    return 1
                print(f"indexed={result['indexed']} skipped={result['skipped']} errors={result['errors']}")
                for detail in result.get("error_details", []):
                    print(f"  ! {detail}", file=sys.stderr)
            return 0 if result.get("errors", 0) == 0 else 1

        if args.command == "search":
            hits = _search_documents(
                store, cfg, args.query,
                limit=args.limit, source=args.source, mode=args.mode, rerank=args.rerank,
            )
            if args.json:
                print(json.dumps(hits, indent=2, ensure_ascii=False))
            else:
                for hit in hits:
                    score = hit.get("rerank_margin", hit.get("rrf_score", hit.get("similarity")))
                    print(f"[{score:.3f}] {hit.get('source')}#{hit.get('chunk')}")
                    print(f"    {hit.get('text', '')[:200].strip()}")
            return 0

        if args.command == "status":
            stats = store.stats()
            if args.json:
                print(json.dumps(stats, indent=2))
            else:
                print(f"files={stats['files']} chunks={stats['chunks']}")
                print(f"data dir: {resolve_path(cfg['store']['path'])}")
            return 0

        if args.command == "doctor":
            return _doctor(cfg, as_json=args.json)

        if args.command == "sources":
            sources = store.list_sources()
            if args.json:
                print(json.dumps(sources, indent=2, ensure_ascii=False))
            else:
                for source in sources:
                    print(source)
            return 0

        if args.command == "delete":
            store.remove_file(args.source)
            print(f"removed {args.source}")
            return 0
    finally:
        store.close()

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
