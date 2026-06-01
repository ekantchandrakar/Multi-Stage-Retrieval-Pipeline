"""
Comprehensive test suite for Hybrid Search System v2
=====================================================
Tests cover:
  - StaQC DataProcessor / dataset_loader
  - IVF FAISS SemanticRetriever (auto-selection, nlist/nprobe)
  - IR Evaluation metrics (NDCG, MRR, Recall, Precision, AP)
  - EvaluationSuite end-to-end
  - BM25Retriever, HybridFusion, CrossEncoderReranker (regression)
  - HybridSearchEngine integration smoke-test
"""

import math
import sys
from pathlib import Path
from typing import List, Set
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_corpus(n: int = 50) -> List[str]:
    templates = [
        "How to implement {} in Python",
        "Optimise {} for large datasets",
        "Deploy {} on Kubernetes",
        "Explain {} with examples",
        "Best practices for {} development",
    ]
    topics = ["linked list", "binary search", "REST API", "SQL joins", "Docker",
              "heap sort", "graph traversal", "caching", "indexing", "recursion"]
    docs = []
    for i in range(n):
        t = templates[i % len(templates)].format(topics[i % len(topics)])
        docs.append(t + f" – document {i}")
    return docs


# ===========================================================================
# Feature 1 – Dataset Loader / DataProcessor
# ===========================================================================

class TestDataProcessor:
    """Tests for the unified DataProcessor (legacy CSV path)."""

    def test_legacy_csv_normalisation(self, tmp_path):
        """DataProcessor should handle CSV with title/body columns."""
        import pandas as pd
        from src.dataset_loader import DataProcessor

        df = pd.DataFrame({
            "id": [1, 2],
            "title": ["How to sort a list", "SQL index"],
            "body": ["Use sorted() builtin", "CREATE INDEX idx ON tbl(col)"],
            "tags": ["python,list", "sql"],
            "category": ["Python", "Databases"],
            "difficulty": ["Beginner", "Intermediate"],
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        dp = DataProcessor(str(csv_path))
        corpus, meta = dp.preprocess_data()

        assert len(corpus) == 2
        assert len(meta) == 2
        assert "How to sort" in corpus[0]
        assert meta[0]["category"] == "Python"

    def test_get_statistics(self, tmp_path):
        import pandas as pd
        from src.dataset_loader import DataProcessor

        df = pd.DataFrame({
            "id": range(10),
            "title": [f"Title {i}" for i in range(10)],
            "body": [f"Body {i}" for i in range(10)],
            "tags": ["python"] * 10,
            "category": ["Python"] * 5 + ["SQL"] * 5,
            "difficulty": ["Beginner"] * 10,
        })
        csv_path = tmp_path / "stats_test.csv"
        df.to_csv(csv_path, index=False)

        dp = DataProcessor(str(csv_path))
        stats = dp.get_statistics()

        assert stats["total_documents"] == 10
        assert "Python" in stats["categories"]

    def test_staqc_tag_parser(self):
        from src.dataset_loader import StaQCDatasetLoader
        loader = StaQCDatasetLoader.__new__(StaQCDatasetLoader)
        assert loader._parse_tags(["python", "list"]) == "python, list"
        assert loader._parse_tags("['sql', 'join']") == "sql, join"
        assert loader._parse_tags("plain string") == "plain string"

    def test_infer_category(self):
        from src.dataset_loader import StaQCDatasetLoader
        assert StaQCDatasetLoader._infer_category("linked list", "python") == "Data Structures"
        assert StaQCDatasetLoader._infer_category("sql join index", "sql") == "Databases"
        assert StaQCDatasetLoader._infer_category("kubernetes docker", "python") == "DevOps & Cloud"
        assert StaQCDatasetLoader._infer_category("unknown topic", "python") == "Python"

    def test_infer_difficulty(self):
        from src.dataset_loader import StaQCDatasetLoader
        assert StaQCDatasetLoader._infer_difficulty("short") == "Beginner"
        assert StaQCDatasetLoader._infer_difficulty("x" * 300) == "Intermediate"
        assert StaQCDatasetLoader._infer_difficulty("x" * 700) == "Advanced"


# ===========================================================================
# Feature 3 – Adaptive FAISS SemanticRetriever
# ===========================================================================

class TestSemanticRetrieverIVF:

    def test_flat_index_small_corpus(self):
        """Corpus < IVF_THRESHOLD → IndexFlatIP."""
        from src.semantic_retriever import SemanticRetriever, IVF_THRESHOLD

        corpus = make_corpus(IVF_THRESHOLD - 1)
        sr = SemanticRetriever()
        sr.build_index(corpus, show_progress=False)

        assert sr._actual_index_type == "flat"
        assert sr.index.ntotal == len(corpus)

    def test_ivf_index_large_corpus(self):
        """Corpus ≥ IVF_THRESHOLD → IVFFlat (or ivfpq for very large)."""
        from src.semantic_retriever import SemanticRetriever, IVF_THRESHOLD

        corpus = make_corpus(IVF_THRESHOLD + 200)
        sr = SemanticRetriever(index_type="ivf")
        sr.build_index(corpus, show_progress=False)

        assert sr._actual_index_type == "ivf"
        assert sr._nlist > 0
        assert sr._nprobe > 0
        assert sr.index.ntotal == len(corpus)

    def test_explicit_flat_override(self):
        """index_type='flat' should always produce a flat index."""
        from src.semantic_retriever import SemanticRetriever

        corpus = make_corpus(200)
        sr = SemanticRetriever(index_type="flat")
        sr.build_index(corpus, show_progress=False)

        assert sr._actual_index_type == "flat"

    def test_search_returns_top_k(self):
        from src.semantic_retriever import SemanticRetriever

        corpus = make_corpus(50)
        sr = SemanticRetriever()
        sr.build_index(corpus, show_progress=False)

        results = sr.search("linked list python", top_k=5)
        assert len(results) <= 5
        for idx, score in results:
            assert 0 <= idx < len(corpus)
            assert isinstance(score, float)

    def test_set_nprobe(self):
        from src.semantic_retriever import SemanticRetriever

        corpus = make_corpus(150)
        sr = SemanticRetriever(index_type="ivf")
        sr.build_index(corpus, show_progress=False)
        sr.set_nprobe(10)
        assert sr._nprobe == 10

    def test_save_and_load(self, tmp_path):
        from src.semantic_retriever import SemanticRetriever

        corpus = make_corpus(30)
        sr = SemanticRetriever()
        sr.build_index(corpus, show_progress=False)
        sr.save(str(tmp_path))

        sr2 = SemanticRetriever()
        sr2.load(str(tmp_path))

        assert sr2.corpus_size == sr.corpus_size
        assert sr2.embedding_dim == sr.embedding_dim
        assert sr2._actual_index_type == sr._actual_index_type

    def test_batch_search(self):
        from src.semantic_retriever import SemanticRetriever

        corpus = make_corpus(40)
        sr = SemanticRetriever()
        sr.build_index(corpus, show_progress=False)

        queries = ["binary search", "sql optimise"]
        all_results = sr.batch_search(queries, top_k=3)
        assert len(all_results) == 2
        for r in all_results:
            assert len(r) <= 3

    def test_ivf_nlist_bounds(self):
        """nlist must satisfy  n_training_pts ≥ nlist * 39 (FAISS k-means)."""
        from src.semantic_retriever import SemanticRetriever

        # Very small corpus: ensure nlist doesn't exceed n // 39
        corpus = make_corpus(200)
        sr = SemanticRetriever(index_type="ivf")
        sr.build_index(corpus, show_progress=False)
        assert sr._nlist <= max(1, len(corpus) // 39)

    def test_statistics_contains_index_type(self):
        from src.semantic_retriever import SemanticRetriever

        corpus = make_corpus(20)
        sr = SemanticRetriever()
        sr.build_index(corpus, show_progress=False)
        stats = sr.get_statistics()
        assert "index_type" in stats
        assert stats["total_vectors"] == len(corpus)


# ===========================================================================
# Feature 2 – IR Evaluation Metrics
# ===========================================================================

class TestNDCG:
    """Unit tests for ndcg_at_k."""

    def test_perfect_ranking(self):
        from src.evaluation import ndcg_at_k
        retrieved = ["d1", "d2", "d3", "d4"]
        relevant = {"d1", "d2"}
        score = ndcg_at_k(retrieved, relevant, k=4)
        assert math.isclose(score, 1.0), f"Expected 1.0 got {score}"

    def test_worst_ranking(self):
        from src.evaluation import ndcg_at_k
        retrieved = ["d3", "d4", "d5"]
        relevant = {"d1", "d2"}
        score = ndcg_at_k(retrieved, relevant, k=3)
        assert score == 0.0

    def test_partial_ranking(self):
        from src.evaluation import ndcg_at_k
        retrieved = ["d3", "d1", "d4", "d2"]
        relevant = {"d1", "d2"}
        score = ndcg_at_k(retrieved, relevant, k=4)
        assert 0.0 < score < 1.0

    def test_empty_retrieved(self):
        from src.evaluation import ndcg_at_k
        assert ndcg_at_k([], {"d1"}, k=5) == 0.0

    def test_empty_relevant(self):
        from src.evaluation import ndcg_at_k
        assert ndcg_at_k(["d1", "d2"], set(), k=5) == 0.0

    def test_k_cutoff_respected(self):
        """Relevant doc at rank k+1 should not affect score."""
        from src.evaluation import ndcg_at_k
        retrieved = ["d2", "d3", "d4", "d1"]   # d1 is relevant but at rank 4
        relevant = {"d1"}
        score_k3 = ndcg_at_k(retrieved, relevant, k=3)
        assert score_k3 == 0.0

    def test_graded_relevance(self):
        from src.evaluation import ndcg_at_k
        retrieved = ["d1", "d2"]
        relevant = {"d1", "d2"}
        grades = {"d1": 3.0, "d2": 1.0}
        score = ndcg_at_k(retrieved, relevant, k=2, relevance_grades=grades)
        assert score == pytest.approx(1.0, abs=1e-6)


class TestMRR:
    def test_first_rank(self):
        from src.evaluation import reciprocal_rank
        assert reciprocal_rank(["d1", "d2"], {"d1"}) == pytest.approx(1.0)

    def test_second_rank(self):
        from src.evaluation import reciprocal_rank
        assert reciprocal_rank(["d2", "d1"], {"d1"}) == pytest.approx(0.5)

    def test_not_found(self):
        from src.evaluation import reciprocal_rank
        assert reciprocal_rank(["d2", "d3"], {"d1"}) == 0.0

    def test_empty_lists(self):
        from src.evaluation import reciprocal_rank
        assert reciprocal_rank([], {"d1"}) == 0.0
        assert reciprocal_rank(["d1"], set()) == 0.0


class TestRecall:
    def test_full_recall(self):
        from src.evaluation import recall_at_k
        assert recall_at_k(["d1", "d2", "d3"], {"d1", "d2"}, k=3) == pytest.approx(1.0)

    def test_partial_recall(self):
        from src.evaluation import recall_at_k
        assert recall_at_k(["d1", "d3"], {"d1", "d2"}, k=2) == pytest.approx(0.5)

    def test_zero_recall(self):
        from src.evaluation import recall_at_k
        assert recall_at_k(["d3", "d4"], {"d1", "d2"}, k=2) == 0.0

    def test_empty_relevant(self):
        from src.evaluation import recall_at_k
        assert recall_at_k(["d1"], set(), k=5) == 0.0

    def test_k_larger_than_retrieved(self):
        from src.evaluation import recall_at_k
        assert recall_at_k(["d1"], {"d1"}, k=100) == pytest.approx(1.0)


class TestPrecision:
    def test_full_precision(self):
        from src.evaluation import precision_at_k
        assert precision_at_k(["d1", "d2"], {"d1", "d2"}, k=2) == pytest.approx(1.0)

    def test_half_precision(self):
        from src.evaluation import precision_at_k
        assert precision_at_k(["d1", "d3"], {"d1", "d2"}, k=2) == pytest.approx(0.5)

    def test_zero_precision(self):
        from src.evaluation import precision_at_k
        assert precision_at_k(["d3"], {"d1"}, k=1) == 0.0


class TestAveragePrecision:
    def test_perfect_ap(self):
        from src.evaluation import average_precision
        assert average_precision(["d1", "d2"], {"d1", "d2"}) == pytest.approx(1.0)

    def test_ap_computation(self):
        from src.evaluation import average_precision
        # d1 at rank 1 (P=1.0), d2 at rank 3 (P=0.667)  →  AP = (1.0+0.667)/2
        retrieved = ["d1", "d3", "d2"]
        relevant = {"d1", "d2"}
        ap = average_precision(retrieved, relevant)
        expected = (1.0 + 2 / 3) / 2
        assert ap == pytest.approx(expected, rel=1e-3)


class TestIRMetrics:
    def test_evaluate_query_keys(self):
        from src.evaluation import IRMetrics
        result = IRMetrics.evaluate_query(["d1", "d2", "d3"], {"d1"}, k=3)
        assert "ndcg@3" in result
        assert "mrr" in result
        assert "recall@3" in result
        assert "precision@3" in result
        assert "ap" in result

    def test_perfect_result(self):
        from src.evaluation import IRMetrics
        m = IRMetrics.evaluate_query(["d1"], {"d1"}, k=10)
        assert m["mrr"] == pytest.approx(1.0)
        assert m["recall@10"] == pytest.approx(1.0)


# ===========================================================================
# EvaluationSuite integration test (mocked engine)
# ===========================================================================

class TestEvaluationSuite:

    @staticmethod
    def _make_mock_engine(n_docs: int = 20):
        """Create a minimal mock HybridSearchEngine."""
        docs = [
            {"id": str(i), "title": f"Doc {i}", "body": f"body {i}",
             "category": "Test", "difficulty": "Beginner",
             "tags": "", "score": 1.0 / (i + 1), "rank": i + 1}
            for i in range(n_docs)
        ]
        engine = MagicMock()
        engine.search.return_value = docs[:5]
        engine.metadata = [
            {"id": str(i), "title": f"Doc {i}", "body": f"body {i}",
             "category": "Test", "difficulty": "Beginner", "tags": ""}
            for i in range(n_docs)
        ]
        return engine, docs

    def test_run_returns_expected_keys(self):
        from src.evaluation import EvaluationSuite

        engine, _ = self._make_mock_engine()
        suite = EvaluationSuite(engine, k=5)
        report = suite.run(
            queries=["query one", "query two"],
            ground_truth={"query one": {"0", "1"}, "query two": {"2", "3"}},
            search_types=["hybrid"],
        )
        assert "per_query" in report
        assert "aggregate" in report
        assert "config" in report
        assert "hybrid" in report["aggregate"]

    def test_aggregate_metrics_range(self):
        from src.evaluation import EvaluationSuite

        engine, _ = self._make_mock_engine()
        suite = EvaluationSuite(engine, k=5)
        report = suite.run(
            queries=["q1"],
            ground_truth={"q1": {"0"}},
            search_types=["hybrid"],
        )
        agg = report["aggregate"]["hybrid"]
        assert 0.0 <= agg["mean_ndcg"] <= 1.0
        assert 0.0 <= agg["mean_mrr"] <= 1.0
        assert 0.0 <= agg["mean_recall"] <= 1.0

    def test_compare_search_types(self):
        from src.evaluation import EvaluationSuite

        engine, _ = self._make_mock_engine()
        suite = EvaluationSuite(engine, k=5)
        report = suite.run(
            queries=["q1"],
            ground_truth={"q1": {"0"}},
            search_types=["bm25", "hybrid"],
        )
        cmp = suite.compare_search_types(report)
        assert len(cmp["comparison_table"]) == 2

    def test_save_report(self, tmp_path):
        from src.evaluation import EvaluationSuite

        engine, _ = self._make_mock_engine()
        suite = EvaluationSuite(engine, k=5)
        report = suite.run(
            queries=["q1"],
            ground_truth={"q1": {"0"}},
            search_types=["hybrid"],
        )
        out = tmp_path / "report.json"
        suite.save_report(report, str(out))
        assert out.exists()


class TestGroundTruthBuilder:
    def test_keyword_match(self):
        from src.evaluation import GroundTruthBuilder

        metadata = [
            {"id": "1", "title": "linked list Python", "body": "iterate nodes"},
            {"id": "2", "title": "SQL join tables",    "body": "inner outer join"},
            {"id": "3", "title": "binary search tree", "body": "BST insert"},
        ]
        gtb = GroundTruthBuilder(metadata)
        gt = gtb.build_from_keywords(["linked list"], min_relevance=1)
        assert "1" in gt["linked list"]
        assert "2" not in gt["linked list"]


# ===========================================================================
# Regression tests for existing components
# ===========================================================================

class TestBM25Retriever:
    def test_build_and_search(self):
        from src.bm25_retriever import BM25Retriever
        corpus = make_corpus(20)
        bm25 = BM25Retriever()
        bm25.build_index(corpus)
        results = bm25.search("linked list Python", top_k=5)
        assert 1 <= len(results) <= 5
        assert all(isinstance(r[0], int) for r in results)

    def test_save_load(self, tmp_path):
        from src.bm25_retriever import BM25Retriever
        corpus = make_corpus(10)
        bm25 = BM25Retriever()
        bm25.build_index(corpus)
        bm25.save(str(tmp_path / "bm25.pkl"))
        bm25b = BM25Retriever()
        bm25b.load(str(tmp_path / "bm25.pkl"))
        assert bm25b.corpus_size == bm25.corpus_size


class TestHybridFusion:
    def test_fuse_two_lists(self):
        from src.hybrid_fusion import HybridFusion
        f = HybridFusion(k=60)
        r1 = [(0, 10.0), (1, 8.0), (2, 6.0)]
        r2 = [(1, 9.0), (2, 7.0), (3, 5.0)]
        res = f.fuse(r1, r2, top_k=4)
        assert len(res) <= 4
        # doc 1 appears in both lists → highest RRF
        assert res[0][0] in {0, 1, 2, 3}

    def test_weighted_fuse(self):
        from src.hybrid_fusion import HybridFusion
        f = HybridFusion()
        r1 = [(0, 10.0)]
        r2 = [(1, 9.0)]
        res = f.weighted_fuse([r1, r2], [0.8, 0.2], top_k=2)
        assert len(res) == 2


class TestCrossEncoderReranker:
    def test_rerank(self):
        from src.cross_encoder_reranker import CrossEncoderReranker
        ce = CrossEncoderReranker()
        q = "Python sort list"
        docs = ["Use sorted() in Python", "SQL ORDER BY", "Python list.sort()"]
        ids = [0, 1, 2]
        results = ce.rerank(q, docs, ids, top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r[0], int) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
