"""Configuration loading and path resolution.

Layering (highest priority first):

1. Environment variables (``NISABA_*``).
2. A user config file: ``$NISABA_CONFIG`` or ``./config.json``.
3. Built-in defaults.

Data paths in the config may be relative — they are resolved against
``NISABA_HOME`` (the project root by default) so a checkout stays portable.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

# Package directory: <root>/nisaba_rag
PACKAGE_DIR = Path(__file__).resolve().parent
# Project root: <root>
PROJECT_ROOT = PACKAGE_DIR.parent

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"

DEFAULTS: dict[str, Any] = {
    "embed": {
        "provider": "ollama",
        "model": "bge-m3",
        "base_url": "http://127.0.0.1:11434",
        "dims": 1024,
        "timeout_s": 30.0,
        # Optional second tier. Left disabled unless an API key is present.
        "nim_model": "nvidia/nv-embedqa-e5-v5",
        "nim_url": "https://integrate.api.nvidia.com/v1/embeddings",
    },
    "store": {
        "path": "data/chroma",
        "collection": "documents",
    },
    "index_db": "data/index.sqlite",
    "reranker": {
        # On by default: hybrid retrieval with reranking is the headline
        # feature. Turn it off for an install with no model present.
        "enabled": True,
        "model_dir": "data/reranker/multilingual-cross-encoder",
        "onnx_file": "model.onnx",
        "tokenizer_file": "tokenizer.json",
        "max_length": 512,
        "provider": "cpu",
        "temperature": 1.0,
        "relevant_logit_index": 1,
    },
    "chunk": {"size": 500, "overlap": 100},
    "extensions": [
        ".md", ".txt", ".json", ".yaml", ".yml", ".py", ".js", ".ts", ".mjs",
        ".cjs", ".html", ".css", ".toml", ".ini", ".cfg", ".sh", ".ps1", ".bat",
    ],
    "exclude_dirs": [
        "node_modules", ".git", ".venv", "venv", "__pycache__", "dist",
        "build", ".next", "target", ".cache", ".chromadb", "chroma_db", "vendor",
    ],
    "exclude_files": [
        "package-lock.json", "pnpm-lock.yaml", "desktop.ini", "Thumbs.db",
    ],
    "max_file_kb": 150,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _apply_env_overrides(cfg: dict) -> dict:
    """Apply the handful of NISABA_* environment overrides we support."""
    embed = cfg.setdefault("embed", {})
    if os.getenv("NISABA_EMBED_MODEL"):
        embed["model"] = os.environ["NISABA_EMBED_MODEL"]
    if os.getenv("NISABA_EMBED_DIMS"):
        # Declared dimensions must match what the model actually returns;
        # `doctor` reports a mismatch rather than letting the index silently
        # fill with unusable vectors.
        try:
            embed["dims"] = int(os.environ["NISABA_EMBED_DIMS"])
        except ValueError:
            pass
    if os.getenv("NISABA_OLLAMA_URL"):
        embed["base_url"] = os.environ["NISABA_OLLAMA_URL"]
    elif os.getenv("OLLAMA_HOST"):
        host = os.environ["OLLAMA_HOST"].strip()
        if not host.startswith("http"):
            host = f"http://{host}"
        embed["base_url"] = host

    if os.getenv("NISABA_COLLECTION"):
        cfg.setdefault("store", {})["collection"] = os.environ["NISABA_COLLECTION"]
    if os.getenv("NISABA_DATA_DIR"):
        data_dir = os.environ["NISABA_DATA_DIR"]
        cfg.setdefault("store", {})["path"] = f"{data_dir}/chroma"
        cfg["index_db"] = f"{data_dir}/index.sqlite"
    if os.getenv("NISABA_RERANKER_DIR"):
        cfg.setdefault("reranker", {})["model_dir"] = os.environ["NISABA_RERANKER_DIR"]
    if os.getenv("NISABA_RERANKER_PROVIDER"):
        cfg.setdefault("reranker", {})["provider"] = os.environ["NISABA_RERANKER_PROVIDER"]
    return cfg


def home() -> Path:
    """Project home: ``NISABA_HOME`` if set, else the checkout root."""
    return Path(os.environ.get("NISABA_HOME") or PROJECT_ROOT).expanduser()


def resolve_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a possibly-relative path against :func:`home`."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (home() / path)


# ChromaDB rejects collection names outside this shape, and its error message
# does not say which rule was broken — so the check happens here instead.
_COLLECTION_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")


def validate_collection_name(name: str) -> str:
    """Return `name` if the vector store will accept it, else raise."""
    if not _COLLECTION_RE.match(name or ""):
        raise ValueError(
            f"invalid store.collection {name!r}: it must be 3-512 characters of "
            "letters, digits, '.', '_' or '-', starting and ending with a letter or digit"
        )
    return name


def config_path() -> Path:
    return Path(os.environ.get("NISABA_CONFIG") or DEFAULT_CONFIG_PATH)


def load_config() -> dict:
    """Load the effective configuration.

    The returned tree is safe to mutate: every level is copied, so a caller
    that adjusts a nested value (or a test that patches one) cannot corrupt the
    module-level ``DEFAULTS`` for the rest of the process.
    """
    cfg = copy.deepcopy(DEFAULTS)
    path = config_path()
    user_cfg: dict = {}
    if path.is_file():
        with open(path, "r", encoding="utf-8") as fh:
            user_cfg = json.load(fh) or {}
        cfg = _deep_merge(cfg, user_cfg)

    # Legacy key compatibility: older configs named this key `ollama_base_url`.
    # The check has to look at the *user's* file, not at the merged result:
    # after merging, `base_url` is always present because the defaults supply
    # it, so a naive `not in embed` test would never fire and the user's URL
    # would be ignored without any warning.
    user_embed = user_cfg.get("embed") or {}
    if "ollama_base_url" in user_embed and "base_url" not in user_embed:
        merged_embed = cfg.setdefault("embed", {})
        merged_embed["base_url"] = user_embed["ollama_base_url"]
        merged_embed.pop("ollama_base_url", None)

    return _apply_env_overrides(cfg)
