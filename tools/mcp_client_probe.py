"""Cliente MCP independiente: habla con Nisaba por stdio como cualquier harness.

No depende de DSH ni de su configuracion. Lanza el server como proceso hijo,
hace el handshake, lista las herramientas y las invoca. Es exactamente lo que
haria Claude Code, Codex o cualquier otro cliente MCP.

Formato de respuesta observado (mcp 2.x):
  * las herramientas que devuelven un objeto  -> 1 bloque de contenido con JSON
  * las que devuelven una lista              -> 1 bloque por elemento, texto plano

Uso:
    python tools/mcp_client_probe.py --command <python> --cwd <raiz> [--index <carpeta>]
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

    show("1. HANDSHAKE - el cliente lanza el server y descubre sus herramientas")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            info = getattr(init, "server_info", None) or getattr(init, "serverInfo", None)
            protocol = getattr(init, "protocol_version", None) or getattr(init, "protocolVersion", "?")
            print(f"  servidor    : {info.name if info else '?'}")
            print(f"  protocolo   : {protocol}")

            tools = await session.list_tools()
            print(f"\n  herramientas descubiertas: {len(tools.tools)}")
            for tool in tools.tools:
                first_line = (tool.description or "").strip().splitlines()[0]
                print(f"    - {tool.name:20} {first_line}")

            show("2. ESTADO INICIAL - indice virgen, sin datos heredados")
            print(f"  status: {json.dumps(obj_of(await session.call_tool('get_index_status', {})))}")
            listed = list_of(await session.call_tool("list_sources", {}))
            print(f"  fuentes indexadas al arrancar: {len(listed)}")

            show("3. DIAGNOSTICO - el server verifica su backend por su cuenta")
            backend = obj_of(await session.call_tool("check_backend", {})) or {}
            print(f"  embeddings ok : {backend.get('ok')}")
            print(f"  base_url      : {backend.get('base_url')}")
            print(f"  modelo        : {backend.get('model')}")
            print(f"  dimensiones   : {backend.get('dimensions')}")
            reranker = obj_of(await session.call_tool("reranker_status", {})) or {}
            print(f"  reranker      : disponible={reranker.get('available')}")

            if args.index:
                show(f"4. INDEXAR - {args.index}")
                payload = obj_of(await session.call_tool("index_folder", {"path": args.index})) or {}
                print(f"  indexados : {payload.get('indexed')}")
                print(f"  saltados  : {payload.get('skipped')}")
                print(f"  errores   : {payload.get('errors')}")
                for detail in payload.get("error_details", [])[:5]:
                    print(f"    ! {detail}")

                show("5. BUSCAR - modo denso")
                hits = list_of(await session.call_tool(
                    "search_documents",
                    {"query": "como se instala el servidor MCP", "limit": 3, "mode": "dense"},
                ))
                for hit in hits:
                    if not isinstance(hit, dict):
                        print(f"  {hit}")
                        continue
                    src = Path(hit.get("source") or "?")
                    print(f"  [sim {hit.get('similarity', 0):.3f}] {src.name}#{hit.get('chunk')}")
                    print(f"          {hit.get('text', '')[:100].strip()}...")

                show("6. BUSCAR - modo hibrido (denso + BM25 + RRF)")
                hits = list_of(await session.call_tool(
                    "search_documents",
                    {"query": "reciprocal rank fusion", "limit": 3, "mode": "hybrid"},
                ))
                for hit in hits:
                    if isinstance(hit, dict):
                        src = Path(hit.get("source") or "?")
                        print(f"  [rrf {hit.get('rrf_score', 0):.4f}] {src.name}#{hit.get('chunk')}")

                show("7. INCREMENTAL - reindexar lo mismo no debe reprocesar nada")
                again = obj_of(await session.call_tool("index_folder", {"path": args.index})) or {}
                print(f"  indexados : {again.get('indexed')}   (debe ser 0)")
                print(f"  saltados  : {again.get('skipped')}")

                show("8. GESTION - listar y borrar una fuente")
                listed = list_of(await session.call_tool("list_sources", {}))
                print(f"  fuentes indexadas: {len(listed)}")
                for entry in listed:
                    print(f"    - {Path(str(entry)).name}")
                if listed:
                    victim = str(listed[0])
                    await session.call_tool("delete_source", {"source": victim})
                    remaining = list_of(await session.call_tool("list_sources", {}))
                    print(f"  borrada: {Path(victim).name}")
                    print(f"  fuentes restantes: {len(remaining)}")

            show("RESULTADO")
            print("  El server respondio a todas las llamadas MCP por stdio.")
            print("  Ninguna configuracion de DSH fue leida ni modificada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
