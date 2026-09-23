"""Standalone MCP client: talks to Nisaba over stdio the way any harness would.

It depends on no particular harness or its configuration. It launches the
server as a child process, performs the handshake, lists the tools and calls
them. This is exactly what Claude Code, Codex or any other MCP client does.

Response framing observed (mcp 2.x):
  * tools that return an object -> one content block containing JSON
  * tools that return a list   -> one block per element, plain text

Usage:
    python tools/mcp_client_probe.py --command <python> --cwd <root> [--index <folder>]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def show(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def configure_stdio() -> None:
    """Survive arbitrary document text on a legacy Windows code page.

    Retrieved chunks can contain a BOM, an emoji or CJK glyphs; the default
    cp1252 console raises UnicodeEncodeError on those.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def blocks_of(result) -> list[str]:
    """Every text content block, in order."""
    return [b.text for b in result.content if hasattr(b, "text")]


def obj_of(result):
    """Parse a single-object result. Returns None when the payload is not JSON."""
    texts = blocks_of(result)
    if len(texts) != 1:
        return None
    try:
        return json.loads(texts[0])
    except json.JSONDecodeError:
        return None


def list_of(result) -> list:
    """Normalise a list-returning tool into a Python list.

    Handles both framings: one JSON array in a single block, or one block per
    element.
    """
    texts = blocks_of(result)
    if len(texts) == 1:
        try:
            parsed = json.loads(texts[0])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    out = []
    for text in texts:
        try:
            out.append(json.loads(text))
        except json.JSONDecodeError:
            out.append(text)
    return out


async def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", default=sys.executable, help="Interpreter to launch the server with.")
    parser.add_argument("--cwd", required=True, help="Project root the server should run in.")
    parser.add_argument("--index", default=None, help="Optional folder to index during the probe.")
    args = parser.parse_args()

    params = StdioServerParameters(
        command=args.command,
        args=["-m", "nisaba_rag.server"],
        cwd=args.cwd,
        env={**os.environ},
    )

    show("1. HANDSHAKE - the client launches the server and discovers its tools")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            info = getattr(init, "server_info", None) or getattr(init, "serverInfo", None)
            protocol = getattr(init, "protocol_version", None) or getattr(init, "protocolVersion", "?")
            print(f"  server      : {info.name if info else '?'}")
            print(f"  protocol    : {protocol}")

            tools = await session.list_tools()
            print(f"\n  tools discovered: {len(tools.tools)}")
            for tool in tools.tools:
                first_line = (tool.description or "").strip().splitlines()[0]
                print(f"    - {tool.name:20} {first_line}")

            show("2. INITIAL STATE - virgin index, no inherited data")
            print(f"  status: {json.dumps(obj_of(await session.call_tool('get_index_status', {})))}")
            listed = list_of(await session.call_tool("list_sources", {}))
            print(f"  sources indexed at startup: {len(listed)}")

            show("3. DIAGNOSTICS - the server checks its own backend")
            backend = obj_of(await session.call_tool("check_backend", {})) or {}
            print(f"  embeddings ok : {backend.get('ok')}")
            print(f"  base_url      : {backend.get('base_url')}")
            print(f"  model         : {backend.get('model')}")
            print(f"  dimensions    : {backend.get('dimensions')}")
            reranker = obj_of(await session.call_tool("reranker_status", {})) or {}
            print(f"  reranker      : available={reranker.get('available')}")

            if args.index:
                show(f"4. INDEX - {args.index}")
                payload = obj_of(await session.call_tool("index_folder", {"path": args.index})) or {}
                print(f"  indexed : {payload.get('indexed')}")
                print(f"  skipped : {payload.get('skipped')}")
                print(f"  errors  : {payload.get('errors')}")
                for detail in payload.get("error_details", [])[:5]:
                    print(f"    ! {detail}")

                show("5. SEARCH - dense mode")
                hits = list_of(await session.call_tool(
                    "search_documents",
                    {"query": "how do I install the MCP server", "limit": 3, "mode": "dense"},
                ))
                for hit in hits:
                    if not isinstance(hit, dict):
                        print(f"  {hit}")
                        continue
                    src = Path(hit.get("source") or "?")
                    print(f"  [sim {hit.get('similarity', 0):.3f}] {src.name}#{hit.get('chunk')}")
                    print(f"          {hit.get('text', '')[:100].strip()}...")

                show("6. SEARCH - hybrid mode (dense + BM25 + RRF)")
                hits = list_of(await session.call_tool(
                    "search_documents",
                    {"query": "reciprocal rank fusion", "limit": 3, "mode": "hybrid"},
                ))
                for hit in hits:
                    if isinstance(hit, dict):
                        src = Path(hit.get("source") or "?")
                        print(f"  [rrf {hit.get('rrf_score', 0):.4f}] {src.name}#{hit.get('chunk')}")

                show("7. INCREMENTAL - re-indexing the same folder must reprocess nothing")
                again = obj_of(await session.call_tool("index_folder", {"path": args.index})) or {}
                print(f"  indexed : {again.get('indexed')}   (must be 0)")
                print(f"  skipped : {again.get('skipped')}")

                show("8. MANAGEMENT - list and delete one source")
                listed = list_of(await session.call_tool("list_sources", {}))
                print(f"  sources indexed: {len(listed)}")
                for entry in listed:
                    print(f"    - {Path(str(entry)).name}")
                if listed:
                    victim = str(listed[0])
                    await session.call_tool("delete_source", {"source": victim})
                    remaining = list_of(await session.call_tool("list_sources", {}))
                    print(f"  deleted: {Path(victim).name}")
                    print(f"  sources remaining: {len(remaining)}")

            show("RESULT")
            print("  The server answered every MCP call over stdio.")
            print("  No harness configuration was read or modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
