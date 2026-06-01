"""
BM25 Retriever – production-grade lexical search
=================================================
Uses rank-bm25 (BM25Okapi) with a lightweight, regex-based tokeniser
that handles code snippets well (camelCase splitting, operator removal).
"""

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .utils import setup_logger, timing_decorator

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_TOKEN_RE = re.compile(r"\b\w+\b")


def tokenize_text(text: str) -> List[str]:
    """
    Tokenise *text* into lowercase tokens.

    Steps
    -----
    1. Split camelCase / PascalCase identifiers.
    2. Extract all word-character sequences.
    3. Lowercase.
    4. Discard pure-digit tokens shorter than 2 chars and very short tokens.
    """
    text = _CAMEL_RE.sub(" ", text)
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if len(t) > 1 and not (t.isdigit() and len(t) < 3)]


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    BM25-based lexical retriever.

    Parameters
    ----------
    k1 : float   Term-frequency saturation  (typical 1.2–2.0)
    b  : float   Length normalisation       (0 = off, 1 = full)
    epsilon : float  IDF floor value
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: List[List[str]] = []
        self.corpus_size: int = 0

        logger.info(f"BM25Retriever(k1={k1}, b={b}, epsilon={epsilon})")

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    @timing_decorator
    def build_index(self, corpus: List[str]) -> None:
        logger.info(f"Tokenising {len(corpus):,} documents …")
        self.tokenized_corpus = [tokenize_text(doc) for doc in corpus]
        self.corpus_size = len(corpus)

        logger.info("Building BM25Okapi index …")
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon,
        )
        logger.info("BM25 index ready")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @timing_decorator
    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            raise RuntimeError("Call build_index() first.")

        tokens = tokenize_text(query)
        scores = self.bm25.get_scores(tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def batch_search(
        self, queries: List[str], top_k: int = 50
    ) -> List[List[Tuple[int, float]]]:
        return [self.search(q, top_k) for q in queries]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def explain_score(self, query: str, doc_index: int) -> Dict[str, Any]:
        if self.bm25 is None:
            raise RuntimeError("Not built.")
        tokens = tokenize_text(query)
        doc_tokens = self.tokenized_corpus[doc_index]
        score = float(self.bm25.get_scores(tokens)[doc_index])
        term_details = {}
        for t in tokens:
            term_details[t] = {
                "tf": doc_tokens.count(t),
                "df": sum(1 for d in self.tokenized_corpus if t in d),
                "in_doc": t in doc_tokens,
            }
        return {
            "doc_index": doc_index,
            "score": score,
            "doc_length": len(doc_tokens),
            "terms": term_details,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        if self.bm25 is None:
            raise RuntimeError("Nothing to save.")
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bm25": self.bm25,
            "tokenized_corpus": self.tokenized_corpus,
            "corpus_size": self.corpus_size,
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
        }
        with open(p, "wb") as f:
            pickle.dump(payload, f, protocol=4)
        logger.info(f"BM25 index saved → {filepath}")

    def load(self, filepath: str) -> None:
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(filepath)
        with open(p, "rb") as f:
            d = pickle.load(f)
        self.bm25 = d["bm25"]
        self.tokenized_corpus = d["tokenized_corpus"]
        self.corpus_size = d["corpus_size"]
        self.k1 = d["k1"]
        self.b = d["b"]
        self.epsilon = d["epsilon"]
        logger.info(f"BM25 index loaded ← {filepath} ({self.corpus_size:,} docs)")

    # ------------------------------------------------------------------
    # Stats / repr
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        if self.bm25 is None:
            return {"status": "not_built"}
        vocab = set(t for doc in self.tokenized_corpus for t in doc)
        avg_len = np.mean([len(d) for d in self.tokenized_corpus])
        return {
            "status": "built",
            "corpus_size": self.corpus_size,
            "vocabulary_size": len(vocab),
            "avg_document_length": float(avg_len),
            "parameters": {"k1": self.k1, "b": self.b, "epsilon": self.epsilon},
        }

    def __repr__(self) -> str:
        if self.bm25 is None:
            return "BM25Retriever(not_built)"
        return f"BM25Retriever(n={self.corpus_size:,}, k1={self.k1}, b={self.b})"
