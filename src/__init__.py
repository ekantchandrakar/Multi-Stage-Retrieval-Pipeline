"""
Hybrid Semantic Search System
==============================
Production-grade search combining BM25, Bi-Encoder (IVF-FAISS),
RRF fusion, and Cross-Encoder re-ranking.

New in this release
-------------------
- StaQC HuggingFace dataset support (2 K → 100 K+ documents)
- Adaptive FAISS indexing (Flat / IVFFlat / IVFPQ auto-selection)
- Full IR evaluation suite: NDCG@K, MRR, Recall@K, MAP
"""

__version__ = "2.0.0"
__author__ = "Ekant Chandrakar"

from .bm25_retriever import BM25Retriever
from .cross_encoder_reranker import CrossEncoderReranker
from .dataset_loader import DataProcessor, StaQCDatasetLoader
from .evaluation import (
    EvaluationSuite,
    GroundTruthBuilder,
    IRMetrics,
    AggregateMetrics,
    QueryMetrics,
    ndcg_at_k,
    reciprocal_rank,
    recall_at_k,
    precision_at_k,
    average_precision,
)
from .hybrid_fusion import HybridFusion
from .search_engine import HybridSearchEngine
from .semantic_retriever import SemanticRetriever

__all__ = [
    # Core components
    "HybridSearchEngine",
    "BM25Retriever",
    "SemanticRetriever",
    "HybridFusion",
    "CrossEncoderReranker",
    # Data
    "DataProcessor",
    "StaQCDatasetLoader",
    # Evaluation
    "EvaluationSuite",
    "GroundTruthBuilder",
    "IRMetrics",
    "AggregateMetrics",
    "QueryMetrics",
    # Metric functions
    "ndcg_at_k",
    "reciprocal_rank",
    "recall_at_k",
    "precision_at_k",
    "average_precision",
]
