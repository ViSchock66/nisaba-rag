"""Nisaba RAG — sparse lexical index (Okapi BM25).

Complementary to the dense (embedding) retriever: BM25 catches exact keyword
matches and rare identifiers that a semantic vector often blurs.

Self-contained (no external dependency), which keeps a corpus of a few
thousand chunks transparent, fast and auditable.

It is built from the chunks already stored in the vector store — nothing is
re-embedded — and rebuilt lazily whenever the index mutates.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

_ALLOWED_LANGS = {"es", "en"}

# Minimal stopword lists. A long list hurts more than it helps on short
# queries; these only remove the emptiest words.
_STOPWORDS: dict[str, set[str]] = {
    "es": {
        "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o",
        "u", "a", "al", "del", "en", "con", "por", "para", "que", "se", "es",
        "son", "su", "sus", "lo", "como", "pero", "mas", "ya", "si", "no",
        "me", "te", "le", "les", "mi", "tu", "tiene", "hay", "sobre", "entre",
        "hacia", "este", "esta", "esto", "ese", "esa", "eso",
    },
    "en": {
        "the", "and", "or", "of", "to", "for", "in", "is", "are", "was",
        "were", "a", "an", "on", "at", "by", "with", "from", "that", "this",
        "it", "as", "be", "been",
    },
}

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# BM25 hyper-parameters (standard Okapi defaults).
_K1 = 1.5
_B = 0.75


def normalize(token: str) -> str:
    """Lowercase and strip accents so 'dimensión' matches 'dimension'."""
    token = token.lower()
    token = unicodedata.normalize("NFKD", token)
    return "".join(ch for ch in token if not unicodedata.combining(ch))


def tokenize(text: str, languages: tuple[str, ...] = ("es", "en")) -> list[str]:
    """Split `text` into normalized content tokens."""
    stopwords: set[str] = set()
    for lang in languages:
        if lang in _ALLOWED_LANGS:
            stopwords |= _STOPWORDS[lang]

    tokens = []
    for word in _TOKEN_RE.findall(text or ""):
        token = normalize(word)
        if len(token) < 2 or token in stopwords:
            continue
        tokens.append(token)
    return tokens


class SparseIndex:
    """In-memory Okapi BM25 index over a corpus of chunks."""

    def __init__(self, languages: tuple[str, ...] = ("es", "en")):
        self.languages = languages
        self._docs: list[dict] = []
        self._doc_freq: Counter = Counter()
        self._avgdl: float = 0.0

    def build(self, docs: list[dict]) -> None:
        """(Re)build the index from ``[{"id", "source", "text"}, ...]``."""
        self._docs = []
        self._doc_freq = Counter()
        total_len = 0

        for doc in docs:
            tokens = tokenize(doc.get("text", ""), self.languages)
            self._docs.append({
                "id": doc.get("id"),
                "source": doc.get("source"),
                "text": doc.get("text", ""),
                "tokens": tokens,
                "len": len(tokens),
            })
            total_len += len(tokens)
            for token in set(tokens):
                self._doc_freq[token] += 1

        n = len(self._docs)
        self._avgdl = (total_len / n) if n else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._doc_freq.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 20) -> list[dict]:
        """Return the top-k documents by BM25 score."""
        query_tokens = tokenize(query, self.languages)
        if not query_tokens or not self._docs:
            return []

        scored: list[tuple[float, dict]] = []
        for doc in self._docs:
            if not doc["len"]:
                continue
            term_freq = Counter(doc["tokens"])
            doc_len = doc["len"]
            score = 0.0
            for token in query_tokens:
                freq = term_freq.get(token, 0)
                if not freq:
                    continue
                if self._avgdl:
                    norm = 1.0 - _B + _B * (doc_len / self._avgdl)
                    denom = freq + _K1 * norm
                else:
                    denom = freq + _K1
                score += self._idf(token) * ((freq * (_K1 + 1.0)) / denom)
            if score > 0.0:
                scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {"id": doc["id"], "source": doc["source"], "text": doc["text"], "score": score}
            for score, doc in scored[:k]
        ]

    def __len__(self) -> int:
        return len(self._docs)
