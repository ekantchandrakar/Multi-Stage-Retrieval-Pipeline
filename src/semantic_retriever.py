"""
Semantic Retriever with Adaptive FAISS Indexing
------------------------------------------------
Supports both exact (IndexFlatIP) and approximate (IndexIVFFlat / IndexIVFPQ)
FAISS indices.  The index type is chosen automatically based on corpus size:

  corpus < IVF_THRESHOLD   → IndexFlatIP  (exact, fast for small sets)
  corpus >= IVF_THRESHOLD  → IndexIVFFlat (approximate, scalable to 100 K+)

For very large corpora (> 500 K) IndexIVFPQ is recommended and can be
enabled explicitly via index_type='ivfpq'.

References
----------
- FAISS wiki: https://github.com/facebookresearch/faiss/wiki
- IVF tuning:  nlist ≈ sqrt(N),  nprobe = max(1, nlist // 10)
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import setup_logger, timing_decorator

logger = setup_logger(__name__)

# Corpus-size threshold for switching from flat → IVF index
IVF_THRESHOLD = 10_000

# IVF hyper-parameters
_IVF_NLIST_FACTOR = 8          # nlist = max(64, N // factor)
_IVF_NPROBE_FRACTION = 0.05    # nprobe = max(1, nlist * fraction)
_IVFPQ_M = 64                  # number of sub-quantisers (must divide dim)
_IVFPQ_NBITS = 8               # bits per sub-quantiser


class SemanticRetriever:
    """
    Dense-vector retriever with adaptive FAISS index selection.

    Parameters
    ----------
    model_name : str
        A sentence-transformers model identifier.
    device : str
        'cpu' or 'cuda'.
    normalize_embeddings : bool
        Normalise to unit length (cosine similarity via inner product).
    index_type : str | None
        'flat'   – always use IndexFlatIP  (exact)
        'ivf'    – always use IndexIVFFlat (approximate)
        'ivfpq'  – always use IndexIVFPQ   (compressed, very large corpora)
        None     – auto-select based on corpus size  (default)
    nlist : int | None
        IVF number of Voronoi cells.  None = auto (≈ sqrt(N)).
    nprobe : int | None
        IVF cells visited per query.  None = auto (≈ nlist * 0.05).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        index_type: Optional[str] = None,   # None = auto
        nlist: Optional[int] = None,
        nprobe: Optional[int] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._requested_index_type = index_type
        self._nlist_override = nlist
        self._nprobe_override = nprobe

        logger.info(f"Loading bi-encoder: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dim: {self.embedding_dim}")

        # Set at build / load time
        self.index: Optional[faiss.Index] = None
        self.embeddings: Optional[np.ndarray] = None
        self.corpus_size: int = 0
        self._actual_index_type: str = "unknown"
        self._nlist: int = 0
        self._nprobe: int = 0

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    @timing_decorator
    def build_index(
        self,
        corpus: List[str],
        batch_size: int = 128,
        show_progress: bool = True,
    ) -> None:
        """
        Encode *corpus* and build a FAISS index.

        For IVF indices the index is first trained on a sample of the
        corpus, then all embeddings are added.
        """
        self.corpus_size = len(corpus)
        logger.info(f"Encoding {self.corpus_size:,} documents …")

        self.embeddings = self.model.encode(
            corpus,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        ).astype(np.float32)

        logger.info(f"Embeddings shape: {self.embeddings.shape}")

        self.index = self._build_faiss_index(self.embeddings)
        logger.info(
            f"FAISS index ready: type={self._actual_index_type}, "
            f"vectors={self.index.ntotal:,}, "
            f"nlist={self._nlist}, nprobe={self._nprobe}"
        )

    def _build_faiss_index(self, vecs: np.ndarray) -> faiss.Index:
        """Select, train, and populate the appropriate FAISS index."""
        n, d = vecs.shape
        chosen = self._choose_index_type(n)
        self._actual_index_type = chosen

        if chosen == "flat":
            index = faiss.IndexFlatIP(d)
            index.add(vecs)
            return index

        if chosen == "ivf":
            nlist = self._calc_nlist(n)
            nprobe = self._calc_nprobe(nlist)
            self._nlist = nlist
            self._nprobe = nprobe

            quantiser = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantiser, d, nlist, faiss.METRIC_INNER_PRODUCT)

            logger.info(f"Training IVFFlat (nlist={nlist}) on {n:,} vectors …")
            index.train(vecs)
            index.add(vecs)
            index.nprobe = nprobe
            return index

        if chosen == "ivfpq":
            nlist = self._calc_nlist(n)
            nprobe = self._calc_nprobe(nlist)
            self._nlist = nlist
            self._nprobe = nprobe

            m = self._calc_pq_m(d)
            quantiser = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFPQ(quantiser, d, nlist, m, _IVFPQ_NBITS)
            index.metric_type = faiss.METRIC_INNER_PRODUCT

            logger.info(f"Training IVFPQ (nlist={nlist}, m={m}) on {n:,} vectors …")
            index.train(vecs)
            index.add(vecs)
            index.nprobe = nprobe
            return index

        raise ValueError(f"Unknown index type: {chosen}")

    # ------------------------------------------------------------------
    # Index type & hyper-parameter selection
    # ------------------------------------------------------------------

    def _choose_index_type(self, n: int) -> str:
        if self._requested_index_type:
            return self._requested_index_type
        if n < IVF_THRESHOLD:
            return "flat"
        if n < 500_000:
            return "ivf"
        return "ivfpq"

    def _calc_nlist(self, n: int) -> int:
        if self._nlist_override:
            return self._nlist_override
        # Rule of thumb: nlist ≈ sqrt(N), bounded to [64, 65536]
        raw = max(64, int(np.sqrt(n)))
        # FAISS requires n_training_points ≥ nlist * 39 (k-means convergence)
        # so cap nlist to n // 39
        return min(raw, max(1, n // 39))

    def _calc_nprobe(self, nlist: int) -> int:
        if self._nprobe_override:
            return self._nprobe_override
        return max(1, int(nlist * _IVF_NPROBE_FRACTION))

    @staticmethod
    def _calc_pq_m(d: int) -> int:
        """Largest divisor of d that is ≤ _IVFPQ_M."""
        for m in range(_IVFPQ_M, 0, -1):
            if d % m == 0:
                return m
        return 1  # fallback

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @timing_decorator
    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Return top_k (doc_index, score) pairs for *query*."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        ).astype(np.float32)

        k = min(top_k, self.corpus_size)
        scores, indices = self.index.search(q_emb, k)

        return [(int(idx), float(sc)) for idx, sc in zip(indices[0], scores[0]) if idx >= 0]

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 50,
        batch_size: int = 64,
    ) -> List[List[Tuple[int, float]]]:
        """Batch search – much faster than calling search() in a loop."""
        if self.index is None:
            raise RuntimeError("Index not built.")

        all_results: List[List[Tuple[int, float]]] = []
        k = min(top_k, self.corpus_size)

        for start in range(0, len(queries), batch_size):
            batch = queries[start : start + batch_size]
            q_embs = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            ).astype(np.float32)

            scores, indices = self.index.search(q_embs, k)
            for row_scores, row_idxs in zip(scores, indices):
                all_results.append(
                    [(int(i), float(s)) for i, s in zip(row_idxs, row_scores) if i >= 0]
                )

        return all_results

    # ------------------------------------------------------------------
    # Tune nprobe at runtime (trade accuracy for speed)
    # ------------------------------------------------------------------

    def set_nprobe(self, nprobe: int) -> None:
        """Adjust nprobe for IVF / IVFPQ indices after building."""
        if self.index is not None and hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe
            self._nprobe = nprobe
            logger.info(f"nprobe updated → {nprobe}")
        else:
            logger.warning("set_nprobe() has no effect on a flat index.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, save_dir: str) -> None:
        if self.index is None or self.embeddings is None:
            raise RuntimeError("Nothing to save.")
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(p / "faiss_index.bin"))
        np.save(p / "embeddings.npy", self.embeddings)

        meta = {
            "model_name": self.model_name,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "embedding_dim": self.embedding_dim,
            "corpus_size": self.corpus_size,
            "actual_index_type": self._actual_index_type,
            "nlist": self._nlist,
            "nprobe": self._nprobe,
        }
        with open(p / "semantic_metadata.pkl", "wb") as f:
            pickle.dump(meta, f)

        logger.info(f"SemanticRetriever saved to {save_dir}")

    def load(self, save_dir: str) -> None:
        p = Path(save_dir)
        if not p.exists():
            raise FileNotFoundError(f"Directory not found: {save_dir}")

        with open(p / "semantic_metadata.pkl", "rb") as f:
            meta = pickle.load(f)

        self.index = faiss.read_index(str(p / "faiss_index.bin"))
        self.embeddings = np.load(p / "embeddings.npy")
        self.embedding_dim = meta["embedding_dim"]
        self.corpus_size = meta["corpus_size"]
        self.normalize_embeddings = meta["normalize_embeddings"]
        self._actual_index_type = meta.get("actual_index_type", "unknown")
        self._nlist = meta.get("nlist", 0)
        self._nprobe = meta.get("nprobe", 0)

        # Restore nprobe on loaded IVF index
        if hasattr(self.index, "nprobe") and self._nprobe > 0:
            self.index.nprobe = self._nprobe

        logger.info(
            f"SemanticRetriever loaded: {self.corpus_size:,} vectors, "
            f"type={self._actual_index_type}"
        )

    # ------------------------------------------------------------------
    # Stats / repr
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        if self.index is None:
            return {"status": "not_built"}
        s: Dict[str, Any] = {
            "status": "built",
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "corpus_size": self.corpus_size,
            "index_type": self._actual_index_type,
            "total_vectors": self.index.ntotal,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
        }
        if self._nlist:
            s["nlist"] = self._nlist
            s["nprobe"] = self._nprobe
        if self.embeddings is not None:
            s["embedding_memory_mb"] = round(self.embeddings.nbytes / 1024 / 1024, 2)
        return s

    def compute_similarity(self, text1: str, text2: str) -> float:
        e1 = self.model.encode([text1], normalize_embeddings=self.normalize_embeddings)[0]
        e2 = self.model.encode([text2], normalize_embeddings=self.normalize_embeddings)[0]
        return float(np.dot(e1, e2))

    def get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )[0]

    def __repr__(self) -> str:
        if self.index is None:
            return f"SemanticRetriever(model={self.model_name}, status=not_built)"
        return (
            f"SemanticRetriever(model={self.model_name}, "
            f"type={self._actual_index_type}, "
            f"n={self.corpus_size:,}, dim={self.embedding_dim})"
        )
