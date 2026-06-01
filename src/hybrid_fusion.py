"""
Reciprocal Rank Fusion (RRF)
============================
Combines ranked lists from multiple retrievers without requiring
score normalisation.

Formula: RRF(d) = Σ_r  1 / (k + rank_r(d))

Reference: Cormack et al., SIGIR 2009.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import setup_logger, timing_decorator

logger = setup_logger(__name__)


class HybridFusion:
    """
    Reciprocal Rank Fusion with optional per-retriever weighting.

    Parameters
    ----------
    k : int
        Smoothing constant (standard value = 60).
        Higher k → less emphasis on top ranks vs lower ranks.
    """

    def __init__(self, k: int = 60):
        self.k = k
        logger.info(f"HybridFusion(k={k})")

    # ------------------------------------------------------------------
    # Core fusion
    # ------------------------------------------------------------------

    @timing_decorator
    def fuse(
        self,
        *ranked_lists: List[Tuple[int, float]],
        top_k: int = 50,
    ) -> List[Tuple[int, float]]:
        """
        Fuse an arbitrary number of ranked lists.

        Parameters
        ----------
        *ranked_lists : List[(doc_id, score)]  – each sorted best-first
        top_k         : how many fused results to return

        Returns
        -------
        List[(doc_id, rrf_score)] sorted descending by rrf_score
        """
        if not ranked_lists:
            return []

        rrf: Dict[int, float] = defaultdict(float)
        for rl in ranked_lists:
            for rank, (doc_id, _) in enumerate(rl, start=1):
                rrf[doc_id] += 1.0 / (self.k + rank)

        return sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]

    @timing_decorator
    def weighted_fuse(
        self,
        ranked_lists: List[List[Tuple[int, float]]],
        weights: List[float],
        top_k: int = 50,
    ) -> List[Tuple[int, float]]:
        """Fuse with per-retriever weights (weights are auto-normalised)."""
        if len(ranked_lists) != len(weights):
            raise ValueError("len(ranked_lists) must equal len(weights)")

        total = sum(weights)
        if total <= 0:
            raise ValueError("Weights must sum to a positive value")
        weights = [w / total for w in weights]

        rrf: Dict[int, float] = defaultdict(float)
        for rl, w in zip(ranked_lists, weights):
            for rank, (doc_id, _) in enumerate(rl, start=1):
                rrf[doc_id] += w / (self.k + rank)

        return sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def explain_fusion(
        self,
        doc_id: int,
        *ranked_lists: List[Tuple[int, float]],
    ) -> Dict[str, Any]:
        total = 0.0
        retriever_info = []
        for i, rl in enumerate(ranked_lists, start=1):
            rank = next(
                (r + 1 for r, (d, _) in enumerate(rl) if d == doc_id), None
            )
            contrib = 1.0 / (self.k + rank) if rank is not None else 0.0
            total += contrib
            retriever_info.append(
                {"retriever": i, "rank": rank, "contribution": contrib}
            )
        return {"doc_id": doc_id, "rrf_score": total, "retrievers": retriever_info}

    def get_overlap(
        self,
        *ranked_lists: List[Tuple[int, float]],
        top_k: int = 50,
    ) -> Dict[str, Any]:
        sets = [set(d for d, _ in rl[:top_k]) for rl in ranked_lists]
        all_docs = set.union(*sets) if sets else set()
        common = set.intersection(*sets) if sets else set()
        return {
            "num_retrievers": len(sets),
            "top_k": top_k,
            "total_unique": len(all_docs),
            "common_to_all": len(common),
            "overlap_pct": round(len(common) / len(all_docs) * 100, 2) if all_docs else 0.0,
        }

    def __repr__(self) -> str:
        return f"HybridFusion(k={self.k})"
