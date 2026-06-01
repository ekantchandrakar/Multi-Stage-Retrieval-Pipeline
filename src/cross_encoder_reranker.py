"""
Cross-Encoder Re-ranker
=======================
Provides fine-grained (query, document) relevance scoring using a
cross-encoder model from sentence-transformers.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from .utils import setup_logger, timing_decorator

logger = setup_logger(__name__)


class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        logger.info("Cross-encoder loaded")

    # ------------------------------------------------------------------

    @timing_decorator
    def rerank(
        self,
        query: str,
        documents: List[str],
        doc_ids: List[int],
        top_k: int = 5,
        batch_size: int = 32,
    ) -> List[Tuple[int, float]]:
        if len(documents) != len(doc_ids):
            raise ValueError("documents and doc_ids must have equal length")
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        raw_scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=False,
        )

        ranked = sorted(
            zip(doc_ids, raw_scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [(int(d), float(s)) for d, s in ranked]

    def score_pair(self, query: str, document: str) -> float:
        return float(self.model.predict([[query, document]])[0])

    def score_pairs(
        self, pairs: List[Tuple[str, str]], batch_size: int = 32
    ) -> List[float]:
        return [float(s) for s in self.model.predict(
            [[q, d] for q, d in pairs],
            batch_size=batch_size,
            show_progress_bar=False,
        )]

    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
        }

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model={self.model_name}, device={self.device})"
