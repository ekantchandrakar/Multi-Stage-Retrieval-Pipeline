"""
Hybrid Search Engine Orchestrator
==================================
Coordinates:
  BM25Retriever  → lexical search
  SemanticRetriever (IVF-FAISS) → dense semantic search
  HybridFusion (RRF)             → rank combination
  CrossEncoderReranker           → precision re-ranking

Supports both the legacy CSV dataset path and the new StaQC HuggingFace
dataset (pass language='python'|'sql' and max_records=N).
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .bm25_retriever import BM25Retriever
from .cross_encoder_reranker import CrossEncoderReranker
from .dataset_loader import DataProcessor
from .hybrid_fusion import HybridFusion
from .semantic_retriever import SemanticRetriever
from .utils import (
    format_search_results,
    save_json,
    setup_logger,
    timing_decorator,
    validate_search_input,
)

logger = setup_logger(__name__)


class HybridSearchEngine:
    """
    Production-grade Hybrid Search Engine.

    Parameters
    ----------
    data_path          : CSV path (legacy) OR any non-CSV string to trigger
                         StaQC HuggingFace download.
    model_dir          : Directory for index persistence.
    bi_encoder_model   : sentence-transformers model name.
    cross_encoder_model: cross-encoder model name.
    device             : 'cpu' or 'cuda'.
    language           : StaQC language split ('python' | 'sql').
    max_records        : Maximum documents to load from StaQC (0 = no cap).
    index_type         : FAISS index type override ('flat'|'ivf'|'ivfpq'|None).
    nlist              : IVF nlist override (None = auto).
    nprobe             : IVF nprobe override (None = auto).
    """

    def __init__(
        self,
        data_path: str = "staqc",
        model_dir: str = "./models",
        bi_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        language: str = "python",
        max_records: int = 100_000,
        index_type: Optional[str] = None,
        nlist: Optional[int] = None,
        nprobe: Optional[int] = None,
    ):
        logger.info("=" * 70)
        logger.info("Initialising Hybrid Search Engine")
        logger.info("=" * 70)

        self.data_path = data_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Component initialisation
        self.data_processor = DataProcessor(
            data_path=data_path,
            language=language,
            max_records=max_records,
            cache_dir=str(self.model_dir / "data_cache"),
        )
        self.bm25_retriever = BM25Retriever()
        self.semantic_retriever = SemanticRetriever(
            model_name=bi_encoder_model,
            device=device,
            index_type=index_type,
            nlist=nlist,
            nprobe=nprobe,
        )
        self.fusion = HybridFusion()
        self.reranker = CrossEncoderReranker(
            model_name=cross_encoder_model,
            device=device,
        )

        self.corpus: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.is_built: bool = False

        logger.info(f"data_path={data_path}, language={language}, max_records={max_records}")
        logger.info(f"model_dir={model_dir}, device={device}")

    # ------------------------------------------------------------------
    # Index building & loading
    # ------------------------------------------------------------------

    @timing_decorator
    def build_indices(self, force_rebuild: bool = True) -> None:
        """
        Build (or reload) all search indices.

        If indices already exist on disk and *force_rebuild* is False,
        they are loaded instead of rebuilt.
        """
        bm25_path = self.model_dir / "bm25_index.pkl"
        faiss_path = self.model_dir / "faiss_index.bin"

        if not force_rebuild and bm25_path.exists() and faiss_path.exists():
            logger.info("Indices found on disk – loading …")
            self.load_indices()
            return

        logger.info("Building indices from scratch …")

        logger.info("Step 1/4  Loading & preprocessing data …")
        self.corpus, self.metadata = self.data_processor.preprocess_data()
        logger.info(f"  → {len(self.corpus):,} documents")

        logger.info("Step 2/4  Building BM25 index …")
        self.bm25_retriever.build_index(self.corpus)

        logger.info("Step 3/4  Building FAISS semantic index …")
        self.semantic_retriever.build_index(self.corpus)

        logger.info("Step 4/4  Saving indices …")
        self.save_indices()

        self.is_built = True
        self._log_statistics()

    def save_indices(self) -> None:
        self.bm25_retriever.save(str(self.model_dir / "bm25_index.pkl"))
        self.semantic_retriever.save(str(self.model_dir))
        logger.info(f"Indices saved → {self.model_dir}")

    def load_indices(self) -> None:
        self.corpus, self.metadata = self.data_processor.preprocess_data()
        bm25_path = self.model_dir / "bm25_index.pkl"
        if bm25_path.exists():
            self.bm25_retriever.load(str(bm25_path))
        else:
            logger.warning(f"BM25 index not found at {bm25_path}")

        faiss_path = self.model_dir / "faiss_index.bin"
        if faiss_path.exists():
            self.semantic_retriever.load(str(self.model_dir))
        else:
            logger.warning(f"FAISS index not found at {faiss_path}")

        self.is_built = True
        logger.info("Indices loaded")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @timing_decorator
    def search(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "hybrid",
        retrieve_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Execute a search query.

        Parameters
        ----------
        query       : free-text query string
        top_k       : number of results to return
        search_type : 'bm25' | 'semantic' | 'hybrid'
        retrieve_k  : candidates retrieved per retriever before re-ranking

        Returns
        -------
        List of result dicts with keys: id, title, body, category,
        difficulty, tags, score, rank
        """
        validate_search_input(query, top_k)
        if not self.is_built:
            raise RuntimeError("Indices not built. Call build_indices() first.")

        if search_type == "bm25":
            return self._search_bm25(query, top_k)
        if search_type == "semantic":
            return self._search_semantic(query, top_k)
        if search_type == "hybrid":
            return self._search_hybrid(query, top_k, retrieve_k)
        raise ValueError(f"Unknown search_type: '{search_type}'")

    # ------------------------------------------------------------------
    # Private search implementations
    # ------------------------------------------------------------------

    def _search_bm25(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return self._format(self.bm25_retriever.search(query, top_k=top_k))

    def _search_semantic(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return self._format(self.semantic_retriever.search(query, top_k=top_k))

    def _search_hybrid(
        self, query: str, top_k: int, retrieve_k: int
    ) -> List[Dict[str, Any]]:
        bm25_res = self.bm25_retriever.search(query, top_k=retrieve_k)
        sem_res = self.semantic_retriever.search(query, top_k=retrieve_k)
        fused = self.fusion.fuse(bm25_res, sem_res, top_k=retrieve_k)

        doc_ids = [d for d, _ in fused]
        documents = [self.corpus[d] for d in doc_ids]
        reranked = self.reranker.rerank(query, documents, doc_ids, top_k=top_k)
        return self._format(reranked)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def compare_retrievers(
        self,
        query: str,
        top_k: int = 5,
        retrieve_k: int = 50,
    ) -> Dict[str, Any]:
        """Run all three search modes and return side-by-side results."""
        bm25_r = self._search_bm25(query, top_k)
        sem_r = self._search_semantic(query, top_k)
        hyb_r = self._search_hybrid(query, top_k, retrieve_k)

        bm25_ids = {r["id"] for r in bm25_r}
        sem_ids = {r["id"] for r in sem_r}
        hyb_ids = {r["id"] for r in hyb_r}

        return {
            "query": query,
            "bm25": bm25_r,
            "semantic": sem_r,
            "hybrid": hyb_r,
            "overlap_analysis": {
                "bm25_semantic_overlap": len(bm25_ids & sem_ids),
                "bm25_hybrid_overlap": len(bm25_ids & hyb_ids),
                "semantic_hybrid_overlap": len(sem_ids & hyb_ids),
                "all_three_overlap": len(bm25_ids & sem_ids & hyb_ids),
            },
        }

    def evaluate_on_queries(
        self,
        test_queries: List[str],
        save_path: Optional[str] = None,
        search_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Legacy evaluation helper (no ground-truth required).
        For full IR metrics use EvaluationSuite from src.evaluation.
        """
        search_types = search_types or ["bm25", "semantic", "hybrid"]
        results: Dict[str, Any] = {
            "queries": [],
            "statistics": {"total_queries": len(test_queries), "avg_search_time": 0.0},
        }
        total_t = 0.0

        for query in test_queries:
            t0 = time.time()
            comparison = self.compare_retrievers(query)
            elapsed = time.time() - t0
            total_t += elapsed
            results["queries"].append(
                {
                    "query": query,
                    "results": comparison,
                    "search_time": elapsed,
                }
            )

        results["statistics"]["avg_search_time"] = total_t / max(len(test_queries), 1)

        if save_path:
            save_json(results, save_path)
            logger.info(f"Evaluation results saved → {save_path}")

        return results

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format(self, results: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
        out = []
        for rank, (doc_id, score) in enumerate(results, start=1):
            meta = dict(self.metadata[doc_id])
            meta["score"] = score
            meta["rank"] = rank
            out.append(meta)
        return out

    # ------------------------------------------------------------------
    # Info / diagnostics
    # ------------------------------------------------------------------

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "data": self.data_processor.get_statistics(),
            "bm25": self.bm25_retriever.get_statistics(),
            "semantic": self.semantic_retriever.get_statistics(),
            "is_built": self.is_built,
            "model_directory": str(self.model_dir),
        }

    def _log_statistics(self) -> None:
        info = self.get_system_info()
        logger.info("=" * 70)
        logger.info("Search Engine Statistics")
        logger.info("=" * 70)
        logger.info(f"Documents      : {info['data'].get('total_documents', '?'):,}")
        logger.info(f"BM25 vocab     : {info['bm25'].get('vocabulary_size', '?'):,}")
        logger.info(f"FAISS type     : {info['semantic'].get('index_type', '?')}")
        logger.info(f"FAISS vectors  : {info['semantic'].get('total_vectors', '?'):,}")
        logger.info("=" * 70)

    def __repr__(self) -> str:
        status = "ready" if self.is_built else "not_built"
        return f"HybridSearchEngine(status={status}, n={len(self.corpus):,})"
