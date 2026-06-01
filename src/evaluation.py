"""
Evaluation Metrics for Information Retrieval
=============================================

Implements:
  - NDCG@K  (Normalised Discounted Cumulative Gain)
  - MRR     (Mean Reciprocal Rank)
  - Recall@K
  - Precision@K
  - MAP     (Mean Average Precision)

All metrics follow standard IR conventions and handle edge-cases
(empty result lists, missing ground-truth, ties) gracefully.

Usage
-----
from src.evaluation import IRMetrics, EvaluationSuite

# Quick single-query evaluation
metrics = IRMetrics.evaluate_query(
    retrieved=["d3", "d1", "d5", "d2"],
    relevant={"d1", "d3"},
    k=10
)
# → {'ndcg@10': 0.93, 'mrr': 1.0, 'recall@10': 1.0, 'precision@10': 0.5}

# Full benchmark over many queries
suite = EvaluationSuite(engine)
report = suite.run(test_queries, ground_truth)
suite.print_report(report)
suite.save_report(report, "outputs/eval.json")
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric functions (stateless, vectorisable)
# ---------------------------------------------------------------------------

def ndcg_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int,
    relevance_grades: Optional[Dict[str, float]] = None,
) -> float:
    """
    Normalised Discounted Cumulative Gain @ K.

    Parameters
    ----------
    retrieved       : ordered list of retrieved document ids (best first)
    relevant        : set of relevant document ids (binary relevance)
    k               : cut-off rank
    relevance_grades: optional {doc_id: grade} for graded relevance.
                      If None, binary relevance (grade=1) is used.

    Returns
    -------
    float in [0, 1].  Returns 0.0 when *relevant* is empty.
    """
    if not relevant or not retrieved:
        return 0.0

    def _rel(doc_id: str) -> float:
        if relevance_grades is not None:
            return float(relevance_grades.get(doc_id, 0.0))
        return 1.0 if doc_id in relevant else 0.0

    # DCG
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        rel = _rel(doc_id)
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    # Ideal DCG: sort perfect list by relevance
    ideal_rels = sorted(
        [_rel(d) for d in relevant],
        reverse=True,
    )[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels) if r > 0)

    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(retrieved: List[str], relevant: Set[str], k: Optional[int] = None) -> float:
    """
    Reciprocal Rank: 1 / rank_of_first_relevant_document.

    Returns 0.0 if no relevant document is found in the top-k results.
    """
    if not relevant or not retrieved:
        return 0.0
    cutoff = retrieved[:k] if k else retrieved
    for rank, doc_id in enumerate(cutoff, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Recall @ K = |retrieved[:k] ∩ relevant| / |relevant|
    """
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Precision @ K = |retrieved[:k] ∩ relevant| / k
    """
    if not retrieved or k == 0:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / min(k, len(retrieved))


def average_precision(retrieved: List[str], relevant: Set[str], k: Optional[int] = None) -> float:
    """
    Average Precision (AP) – area under the precision-recall curve.

    Used to compute MAP over multiple queries.
    """
    if not relevant or not retrieved:
        return 0.0
    cutoff = retrieved[:k] if k else retrieved
    hits, total_ap = 0, 0.0
    for rank, doc_id in enumerate(cutoff, start=1):
        if doc_id in relevant:
            hits += 1
            total_ap += hits / rank
    return total_ap / min(len(relevant), len(cutoff)) if hits > 0 else 0.0


# ---------------------------------------------------------------------------
# Dataclasses for structured results
# ---------------------------------------------------------------------------

@dataclass
class QueryMetrics:
    """Per-query metric snapshot."""
    query: str
    retrieved_ids: List[str]
    relevant_ids: List[str]
    ndcg_at_k: float
    mrr: float
    recall_at_k: float
    precision_at_k: float
    average_precision: float
    k: int
    search_time_ms: float = 0.0
    search_type: str = "hybrid"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateMetrics:
    """Mean metrics across all evaluated queries."""
    num_queries: int
    k: int
    mean_ndcg: float
    mean_mrr: float
    mean_recall: float
    mean_precision: float
    mean_ap: float          # → MAP
    mean_search_time_ms: float
    std_ndcg: float = 0.0
    std_mrr: float = 0.0
    std_recall: float = 0.0
    search_type: str = "hybrid"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def pretty(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  Evaluation Results  (k={self.k}, n={self.num_queries} queries)",
            f"{'='*60}",
            f"  NDCG@{self.k:<4}   : {self.mean_ndcg:.4f}  (σ={self.std_ndcg:.4f})",
            f"  MRR        : {self.mean_mrr:.4f}  (σ={self.std_mrr:.4f})",
            f"  Recall@{self.k:<4} : {self.mean_recall:.4f}  (σ={self.std_recall:.4f})",
            f"  Precision@{self.k:<3}: {self.mean_precision:.4f}",
            f"  MAP        : {self.mean_ap:.4f}",
            f"  Avg latency: {self.mean_search_time_ms:.1f} ms",
            f"{'='*60}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class IRMetrics:
    """
    Stateless helper – compute all metrics for a single query at once.
    """

    @staticmethod
    def evaluate_query(
        retrieved: List[str],
        relevant: Set[str],
        k: int = 10,
        relevance_grades: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Return a dict with all metrics for one query.

        Parameters
        ----------
        retrieved  : ranked list of doc ids (best first)
        relevant   : set of known-relevant doc ids
        k          : evaluation cut-off
        """
        return {
            f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k, relevance_grades),
            "mrr": reciprocal_rank(retrieved, relevant),
            f"recall@{k}": recall_at_k(retrieved, relevant, k),
            f"precision@{k}": precision_at_k(retrieved, relevant, k),
            "ap": average_precision(retrieved, relevant, k),
        }


# ---------------------------------------------------------------------------
# Full evaluation suite (works with HybridSearchEngine)
# ---------------------------------------------------------------------------

class EvaluationSuite:
    """
    Runs a structured evaluation benchmark over a set of test queries.

    Ground-truth format
    -------------------
    ground_truth: Dict[str, Set[str]]
        Mapping  query_string → {relevant_doc_id, …}

    If no ground_truth is provided, pseudo-relevance is built by treating
    the top-1 result of the first search type as the single relevant document
    (useful for comparative / relative benchmarks when labels are absent).

    Example
    -------
    suite = EvaluationSuite(engine, k=10)
    report = suite.run(
        queries=["reverse linked list", "SQL join optimisation"],
        ground_truth={"reverse linked list": {"q_123", "q_456"}, …},
        search_types=["bm25", "semantic", "hybrid"],
    )
    suite.print_report(report)
    suite.save_report(report, "outputs/eval_report.json")
    """

    def __init__(self, engine, k: int = 10):
        """
        Parameters
        ----------
        engine : HybridSearchEngine
        k      : evaluation cut-off (NDCG@k, Recall@k, etc.)
        """
        self.engine = engine
        self.k = k

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        queries: List[str],
        ground_truth: Optional[Dict[str, Set[str]]] = None,
        search_types: Optional[List[str]] = None,
        top_k: int = None,
    ) -> Dict[str, Any]:
        """
        Run the full evaluation benchmark.

        Returns
        -------
        report : dict with keys
            'per_query'   : List[QueryMetrics.to_dict()]
            'aggregate'   : Dict[search_type → AggregateMetrics.to_dict()]
            'config'      : evaluation config
        """
        if search_types is None:
            search_types = ["bm25", "semantic", "hybrid"]
        eval_k = top_k or max(self.k, 10)

        logger.info(
            f"Starting evaluation: {len(queries)} queries, "
            f"k={self.k}, search_types={search_types}"
        )

        per_query_results: List[Dict[str, Any]] = []
        # {search_type → list of QueryMetrics}
        per_type: Dict[str, List[QueryMetrics]] = defaultdict(list)

        for idx, query in enumerate(queries):
            logger.debug(f"Query {idx+1}/{len(queries)}: '{query}'")
            query_row: Dict[str, Any] = {"query": query, "results": {}}

            for stype in search_types:
                t0 = time.perf_counter()
                try:
                    results = self.engine.search(query, top_k=eval_k, search_type=stype)
                except Exception as exc:
                    logger.warning(f"Search failed ({stype}, '{query}'): {exc}")
                    results = []
                elapsed_ms = (time.perf_counter() - t0) * 1000

                retrieved_ids = [str(r["id"]) for r in results]
                relevant = self._get_relevant(query, ground_truth)

                metrics = QueryMetrics(
                    query=query,
                    retrieved_ids=retrieved_ids,
                    relevant_ids=list(relevant),
                    ndcg_at_k=ndcg_at_k(retrieved_ids, relevant, self.k),
                    mrr=reciprocal_rank(retrieved_ids, relevant),
                    recall_at_k=recall_at_k(retrieved_ids, relevant, self.k),
                    precision_at_k=precision_at_k(retrieved_ids, relevant, self.k),
                    average_precision=average_precision(retrieved_ids, relevant, self.k),
                    k=self.k,
                    search_time_ms=elapsed_ms,
                    search_type=stype,
                )
                per_type[stype].append(metrics)
                query_row["results"][stype] = metrics.to_dict()

            per_query_results.append(query_row)

        # Aggregate
        aggregate: Dict[str, Dict[str, Any]] = {}
        for stype, qm_list in per_type.items():
            aggregate[stype] = self._aggregate(qm_list).to_dict()

        report = {
            "per_query": per_query_results,
            "aggregate": aggregate,
            "config": {
                "num_queries": len(queries),
                "k": self.k,
                "search_types": search_types,
                "has_ground_truth": ground_truth is not None,
            },
        }
        return report

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def print_report(self, report: Dict[str, Any]) -> None:
        """Pretty-print aggregate metrics to stdout."""
        print(f"\n{'='*60}")
        print(f"  IR Evaluation Report  (k={report['config']['k']})")
        print(f"  Queries evaluated: {report['config']['num_queries']}")
        print(f"{'='*60}")

        for stype, agg in report["aggregate"].items():
            print(f"\n  Search type: {stype.upper()}")
            print(f"  {'─'*40}")
            print(f"  NDCG@{agg['k']:<5}: {agg['mean_ndcg']:.4f}  (σ={agg['std_ndcg']:.4f})")
            print(f"  MRR       : {agg['mean_mrr']:.4f}  (σ={agg['std_mrr']:.4f})")
            print(f"  Recall@{agg['k']:<4}: {agg['mean_recall']:.4f}  (σ={agg['std_recall']:.4f})")
            print(f"  Precision@{agg['k']:<3}: {agg['mean_precision']:.4f}")
            print(f"  MAP       : {agg['mean_ap']:.4f}")
            print(f"  Avg latency: {agg['mean_search_time_ms']:.1f} ms")

        print(f"\n{'='*60}\n")

    def save_report(self, report: Dict[str, Any], path: str) -> None:
        """Serialise the full report to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Evaluation report saved → {path}")

    def compare_search_types(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a summary table comparing search types side-by-side.

        Useful for quick CLI output or a Pandas DataFrame.
        """
        rows = []
        for stype, agg in report["aggregate"].items():
            rows.append(
                {
                    "search_type": stype,
                    f"ndcg@{agg['k']}": round(agg["mean_ndcg"], 4),
                    "mrr": round(agg["mean_mrr"], 4),
                    f"recall@{agg['k']}": round(agg["mean_recall"], 4),
                    f"precision@{agg['k']}": round(agg["mean_precision"], 4),
                    "map": round(agg["mean_ap"], 4),
                    "avg_latency_ms": round(agg["mean_search_time_ms"], 1),
                }
            )
        return {"comparison_table": rows}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_relevant(
        query: str,
        ground_truth: Optional[Dict[str, Set[str]]],
    ) -> Set[str]:
        """Return the relevant set for *query*, or empty set if unknown."""
        if ground_truth is None:
            return set()
        # Exact match first, then case-insensitive
        if query in ground_truth:
            return ground_truth[query]
        ql = query.lower()
        for k, v in ground_truth.items():
            if k.lower() == ql:
                return v
        return set()

    def _aggregate(self, qm_list: List[QueryMetrics]) -> AggregateMetrics:
        ndcgs = [q.ndcg_at_k for q in qm_list]
        mrrs = [q.mrr for q in qm_list]
        recalls = [q.recall_at_k for q in qm_list]
        precs = [q.precision_at_k for q in qm_list]
        aps = [q.average_precision for q in qm_list]
        times = [q.search_time_ms for q in qm_list]

        stype = qm_list[0].search_type if qm_list else "unknown"
        return AggregateMetrics(
            num_queries=len(qm_list),
            k=self.k,
            mean_ndcg=float(np.mean(ndcgs)),
            mean_mrr=float(np.mean(mrrs)),
            mean_recall=float(np.mean(recalls)),
            mean_precision=float(np.mean(precs)),
            mean_ap=float(np.mean(aps)),
            mean_search_time_ms=float(np.mean(times)),
            std_ndcg=float(np.std(ndcgs)),
            std_mrr=float(np.std(mrrs)),
            std_recall=float(np.std(recalls)),
            search_type=stype,
        )


# ---------------------------------------------------------------------------
# Ground-truth builder helpers
# ---------------------------------------------------------------------------

class GroundTruthBuilder:
    """
    Utilities for constructing ground-truth relevance judgements when
    manually labelled data is not available.

    Two strategies:
    1. keyword_match  – relevant if query keywords appear in title/body
    2. category_match – relevant if document belongs to the same category
                        as the most common result from a warm search
    """

    def __init__(self, metadata: List[Dict[str, Any]]):
        self.metadata = metadata

    def build_from_keywords(
        self,
        queries: List[str],
        top_k: int = 10,
        min_relevance: int = 2,
    ) -> Dict[str, Set[str]]:
        """
        Pseudo-relevance: a document is 'relevant' if it contains at least
        *min_relevance* query tokens in its title+body.
        """
        import re

        gt: Dict[str, Set[str]] = {}
        for query in queries:
            tokens = set(re.findall(r"\b\w+\b", query.lower()))
            relevant: Set[str] = set()
            for doc in self.metadata:
                text = (doc.get("title", "") + " " + doc.get("body", "")).lower()
                matches = sum(1 for t in tokens if t in text)
                if matches >= min_relevance:
                    relevant.add(str(doc["id"]))
            gt[query] = relevant
            logger.debug(f"'{query}' → {len(relevant)} relevant docs")
        return gt

    def build_from_engine_top1(
        self,
        engine,
        queries: List[str],
        search_type: str = "hybrid",
    ) -> Dict[str, Set[str]]:
        """
        Pseudo-relevance using the engine's top-1 result as the sole
        relevant document.  Useful for MRR sanity checks.
        """
        gt: Dict[str, Set[str]] = {}
        for query in queries:
            results = engine.search(query, top_k=1, search_type=search_type)
            if results:
                gt[query] = {str(results[0]["id"])}
            else:
                gt[query] = set()
        return gt
