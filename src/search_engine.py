"""
Hybrid Search Engine Orchestrator
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
from .utils import save_json, setup_logger, timing_decorator, validate_search_input

logger = setup_logger(__name__)


class HybridSearchEngine:

    def __init__(
        self,
        data_path: str = "staqc",
        model_dir: str = "./models",
        bi_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        language: str = "mca_python",
        max_records: int = 100_000,
        index_type: Optional[str] = None,
        nlist: Optional[int] = None,
        nprobe: Optional[int] = None,
    ):
        logger.info("=" * 70)
        logger.info("Initialising Hybrid Search Engine")
        logger.info("=" * 70)

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

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
        self.fusion   = HybridFusion()
        self.reranker = CrossEncoderReranker(model_name=cross_encoder_model, device=device)

        self.corpus:   List[str]            = []
        self.metadata: List[Dict[str, Any]] = []
        self.is_built: bool = False

        logger.info(f"data_path={data_path}, language={language}, max_records={max_records}")
        logger.info(f"model_dir={model_dir}, device={device}")

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_indices(self, force_rebuild: bool = False) -> None:
        bm25_path  = self.model_dir / "bm25_index.pkl"
        faiss_path = self.model_dir / "faiss_index.bin"

        if not force_rebuild and bm25_path.exists() and faiss_path.exists():
            logger.info("Indices exist on disk → loading (use force_rebuild=True to rebuild)")
            self._load_from_disk()
            return

        # ── Full build ──────────────────────────────────────────────
        logger.info("=" * 50)
        logger.info("BUILDING INDICES FROM SCRATCH")
        logger.info("=" * 50)

        # Step 1 — data
        logger.info("Step 1/4  Loading & preprocessing data …")
        self.corpus, self.metadata = self.data_processor.preprocess_data()
        n = len(self.corpus)
        logger.info(f"Step 1/4  ✓ {n:,} documents loaded")
        print(f"\n  ✓ Step 1/4 complete — {n:,} documents loaded")

        if n == 0:
            raise RuntimeError("No documents loaded. Check dataset config and network.")

        # Step 2 — BM25
        logger.info("Step 2/4  Building BM25 index …")
        print("  ⏳ Step 2/4  Building BM25 index …")
        self.bm25_retriever.build_index(self.corpus)
        logger.info("Step 2/4  ✓ BM25 index built")
        print("  ✓ Step 2/4  BM25 index built")

        # Step 3 — FAISS
        logger.info("Step 3/4  Building FAISS semantic index …")
        print("  ⏳ Step 3/4  Building FAISS semantic index (encoding ~4K docs) …")
        self.semantic_retriever.build_index(self.corpus)
        logger.info("Step 3/4  ✓ FAISS index built")
        print("  ✓ Step 3/4  FAISS index built")

        # Step 4 — Save
        logger.info("Step 4/4  Saving indices …")
        print("  ⏳ Step 4/4  Saving indices to disk …")
        self.bm25_retriever.save(str(self.model_dir / "bm25_index.pkl"))
        self.semantic_retriever.save(str(self.model_dir))
        logger.info(f"Step 4/4  ✓ Indices saved → {self.model_dir}")
        print(f"  ✓ Step 4/4  Indices saved → {self.model_dir}")

        self.is_built = True
        self._log_stats()
        print("\n  ✅ All indices built and saved successfully!\n")

    def _load_from_disk(self) -> None:
        logger.info("Loading corpus …")
        print("\n  Loading corpus from cache …")
        self.corpus, self.metadata = self.data_processor.preprocess_data()
        print(f"  ✓ {len(self.corpus):,} documents")

        logger.info("Loading BM25 index …")
        print("  Loading BM25 index …")
        self.bm25_retriever.load(str(self.model_dir / "bm25_index.pkl"))
        print("  ✓ BM25 loaded")

        logger.info("Loading FAISS index …")
        print("  Loading FAISS index …")
        self.semantic_retriever.load(str(self.model_dir))
        print("  ✓ FAISS loaded")

        self.is_built = True
        print("  ✅ All indices loaded from disk\n")

    # kept for backwards-compat
    def load_indices(self)  -> None: self._load_from_disk()
    def save_indices(self)  -> None:
        self.bm25_retriever.save(str(self.model_dir / "bm25_index.pkl"))
        self.semantic_retriever.save(str(self.model_dir))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "hybrid",
        retrieve_k: int = 50,
    ) -> List[Dict[str, Any]]:
        validate_search_input(query, top_k)
        if not self.is_built:
            raise RuntimeError("Call build_indices() first.")

        if search_type == "bm25":
            return self._fmt(self.bm25_retriever.search(query, top_k=top_k))
        if search_type == "semantic":
            return self._fmt(self.semantic_retriever.search(query, top_k=top_k))
        if search_type == "hybrid":
            bm25_r = self.bm25_retriever.search(query, top_k=retrieve_k)
            sem_r  = self.semantic_retriever.search(query, top_k=retrieve_k)
            fused  = self.fusion.fuse(bm25_r, sem_r, top_k=retrieve_k)
            ids    = [d for d, _ in fused]
            docs   = [self.corpus[d] for d in ids]
            return self._fmt(self.reranker.rerank(query, docs, ids, top_k=top_k))
        raise ValueError(f"Unknown search_type: '{search_type}'")

    def compare_retrievers(self, query: str, top_k: int = 5, retrieve_k: int = 50) -> Dict[str, Any]:
        bm25_r = self._fmt(self.bm25_retriever.search(query, top_k=top_k))
        sem_r  = self._fmt(self.semantic_retriever.search(query, top_k=top_k))
        hyb_r  = self.search(query, top_k=top_k, search_type="hybrid", retrieve_k=retrieve_k)
        b, s, h = {r["id"] for r in bm25_r}, {r["id"] for r in sem_r}, {r["id"] for r in hyb_r}
        return {
            "query": query, "bm25": bm25_r, "semantic": sem_r, "hybrid": hyb_r,
            "overlap_analysis": {
                "bm25_semantic_overlap":   len(b & s),
                "bm25_hybrid_overlap":     len(b & h),
                "semantic_hybrid_overlap": len(s & h),
                "all_three_overlap":       len(b & s & h),
            },
        }

    def evaluate_on_queries(
        self, test_queries: List[str], save_path: Optional[str] = None, **_
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "queries": [],
            "statistics": {"total_queries": len(test_queries), "avg_search_time": 0.0},
        }
        total_t = 0.0
        for q in test_queries:
            t0 = time.time()
            comp = self.compare_retrievers(q)
            elapsed = time.time() - t0
            total_t += elapsed
            results["queries"].append({"query": q, "results": comp, "search_time": elapsed})
        results["statistics"]["avg_search_time"] = total_t / max(len(test_queries), 1)
        if save_path:
            save_json(results, save_path)
        return results

    def _fmt(self, results: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
        out = []
        for rank, (doc_id, score) in enumerate(results, start=1):
            m = dict(self.metadata[doc_id])
            m["score"] = score
            m["rank"]  = rank
            out.append(m)
        return out

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "data":     self.data_processor.get_statistics(),
            "bm25":     self.bm25_retriever.get_statistics(),
            "semantic": self.semantic_retriever.get_statistics(),
            "is_built": self.is_built,
            "model_directory": str(self.model_dir),
        }

    def _log_stats(self) -> None:
        info = self.get_system_info()
        logger.info("=" * 70)
        logger.info(f"Documents  : {info['data'].get('total_documents','?'):,}")
        logger.info(f"BM25 vocab : {info['bm25'].get('vocabulary_size','?'):,}")
        logger.info(f"FAISS type : {info['semantic'].get('index_type','?')}")
        logger.info(f"FAISS vecs : {info['semantic'].get('total_vectors','?'):,}")
        logger.info("=" * 70)

    def __repr__(self) -> str:
        return f"HybridSearchEngine(status={'ready' if self.is_built else 'not_built'}, n={len(self.corpus):,})"