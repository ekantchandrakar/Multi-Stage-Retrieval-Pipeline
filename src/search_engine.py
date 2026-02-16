"""
Main Hybrid Search Engine
Orchestrates BM25, Semantic Search, RRF Fusion, and Cross-Encoder Re-ranking
"""
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .data_processor import DataProcessor
from .bm25_retriever import BM25Retriever
from .semantic_retriever import SemanticRetriever
from .hybrid_fusion import HybridFusion
from .cross_encoder_reranker import CrossEncoderReranker
from .utils import (
    setup_logger, 
    timing_decorator, 
    validate_search_input,
    format_search_results,
    save_json
)

logger = setup_logger(__name__)


class HybridSearchEngine:
    """
    Production-grade Hybrid Search Engine combining:
    - BM25 (Lexical Search)
    - Bi-Encoder (Semantic Search)
    - RRF (Reciprocal Rank Fusion)
    - Cross-Encoder (Re-ranking)
    """
    
    def __init__(self,
                 data_path: str,
                 model_dir: str = "./models",
                 bi_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str = "cpu"):
        """
        Initialize Hybrid Search Engine.
        
        Args:
            data_path: Path to dataset CSV
            model_dir: Directory for saving/loading models
            bi_encoder_model: Bi-encoder model name
            cross_encoder_model: Cross-encoder model name
            device: Device to use ('cpu' or 'cuda')
        """
        logger.info("="*80)
        logger.info("Initializing Hybrid Search Engine")
        logger.info("="*80)
        
        self.data_path = data_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_processor = DataProcessor(data_path)
        self.bm25_retriever = BM25Retriever()
        self.semantic_retriever = SemanticRetriever(
            model_name=bi_encoder_model,
            device=device
        )
        self.fusion = HybridFusion()
        self.reranker = CrossEncoderReranker(
            model_name=cross_encoder_model,
            device=device
        )
        
        # Data storage
        self.corpus = []
        self.metadata = []
        self.is_built = False
        
        logger.info(f"Data path: {data_path}")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Bi-Encoder: {bi_encoder_model}")
        logger.info(f"Cross-Encoder: {cross_encoder_model}")
        logger.info(f"Device: {device}")
    
    @timing_decorator
    def build_indices(self, force_rebuild: bool = False) -> None:
        """
        Build all search indices.
        
        Args:
            force_rebuild: Force rebuild even if indices exist
        """
        logger.info("="*80)
        logger.info("Building Search Indices")
        logger.info("="*80)
        
        # Check if indices already exist
        bm25_path = self.model_dir / "bm25_index.pkl"
        semantic_dir = self.model_dir
        
        if not force_rebuild and bm25_path.exists():
            logger.info("Indices already exist. Loading from disk...")
            self.load_indices()
            return
        
        # Load and preprocess data
        logger.info("Step 1: Loading and preprocessing data...")
        self.corpus, self.metadata = self.data_processor.preprocess_data()
        logger.info(f"Loaded {len(self.corpus)} documents")
        
        # Build BM25 index
        logger.info("Step 2: Building BM25 index...")
        self.bm25_retriever.build_index(self.corpus)
        
        # Build Semantic index
        logger.info("Step 3: Building Semantic index (this may take a few minutes)...")
        self.semantic_retriever.build_index(self.corpus)
        
        # Save indices
        logger.info("Step 4: Saving indices...")
        self.save_indices()
        
        self.is_built = True
        logger.info("="*80)
        logger.info("Index building complete!")
        logger.info("="*80)
        
        # Print statistics
        self._print_statistics()
    
    @timing_decorator
    def search(self,
              query: str,
              top_k: int = 5,
              search_type: str = 'hybrid',
              retrieve_k: int = 50) -> List[Dict[str, Any]]:
        """
        Perform search.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            search_type: Type of search ('bm25', 'semantic', 'hybrid')
            retrieve_k: Number of candidates to retrieve before re-ranking
            
        Returns:
            List of result dictionaries with metadata
        """
        validate_search_input(query, top_k)
        
        if not self.is_built:
            raise ValueError("Indices not built. Call build_indices() first.")
        
        logger.info(f"Searching for: '{query}' (type={search_type}, top_k={top_k})")
        
        start_time = time.time()
        
        if search_type == 'bm25':
            results = self._search_bm25_only(query, top_k)
        elif search_type == 'semantic':
            results = self._search_semantic_only(query, top_k)
        elif search_type == 'hybrid':
            results = self._search_hybrid(query, top_k, retrieve_k)
        else:
            raise ValueError(f"Unknown search_type: {search_type}")
        
        elapsed = time.time() - start_time
        logger.info(f"Search completed in {elapsed:.3f}s")
        
        return results
    
    def _search_bm25_only(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """BM25-only search."""
        bm25_results = self.bm25_retriever.search(query, top_k=top_k)
        return self._format_results(bm25_results)
    
    def _search_semantic_only(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Semantic-only search."""
        semantic_results = self.semantic_retriever.search(query, top_k=top_k)
        return self._format_results(semantic_results)
    
    def _search_hybrid(self, query: str, top_k: int, retrieve_k: int) -> List[Dict[str, Any]]:
        """Full hybrid search with RRF fusion and cross-encoder re-ranking."""
        
        # Step 1: Retrieve candidates from both retrievers
        logger.debug(f"Retrieving top {retrieve_k} from BM25...")
        bm25_results = self.bm25_retriever.search(query, top_k=retrieve_k)
        
        logger.debug(f"Retrieving top {retrieve_k} from Semantic Search...")
        semantic_results = self.semantic_retriever.search(query, top_k=retrieve_k)
        
        # Step 2: Fuse using RRF
        logger.debug("Fusing results using RRF...")
        fused_results = self.fusion.fuse(bm25_results, semantic_results, top_k=retrieve_k)
        
        # Step 3: Re-rank using Cross-Encoder
        logger.debug(f"Re-ranking top {min(retrieve_k, len(fused_results))} with Cross-Encoder...")
        
        # Get documents for re-ranking
        doc_ids = [doc_id for doc_id, _ in fused_results]
        documents = [self.corpus[doc_id] for doc_id in doc_ids]
        
        # Re-rank
        reranked_results = self.reranker.rerank(
            query=query,
            documents=documents,
            doc_ids=doc_ids,
            top_k=top_k
        )
        
        return self._format_results(reranked_results)
    
    def _format_results(self, 
                       results: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
        """
        Format results with metadata.
        
        Args:
            results: List of (doc_id, score) tuples
            
        Returns:
            List of formatted result dictionaries
        """
        formatted = []
        
        for doc_id, score in results:
            meta = self.metadata[doc_id].copy()
            meta['score'] = score
            meta['rank'] = len(formatted) + 1
            formatted.append(meta)
        
        return formatted
    
    def compare_retrievers(self,
                          query: str,
                          top_k: int = 5,
                          retrieve_k: int = 50) -> Dict[str, Any]:
        """
        Compare results from all three retrieval methods.
        
        Args:
            query: Search query
            top_k: Number of results to return
            retrieve_k: Number of candidates for hybrid approach
            
        Returns:
            Dictionary with results from each method
        """
        logger.info(f"Comparing retrievers for query: '{query}'")
        
        comparison = {
            'query': query,
            'bm25': self._search_bm25_only(query, top_k),
            'semantic': self._search_semantic_only(query, top_k),
            'hybrid': self._search_hybrid(query, top_k, retrieve_k)
        }
        
        # Analyze overlap
        bm25_ids = set(r['id'] for r in comparison['bm25'])
        semantic_ids = set(r['id'] for r in comparison['semantic'])
        hybrid_ids = set(r['id'] for r in comparison['hybrid'])
        
        comparison['overlap_analysis'] = {
            'bm25_semantic_overlap': len(bm25_ids & semantic_ids),
            'bm25_hybrid_overlap': len(bm25_ids & hybrid_ids),
            'semantic_hybrid_overlap': len(semantic_ids & hybrid_ids),
            'all_three_overlap': len(bm25_ids & semantic_ids & hybrid_ids)
        }
        
        return comparison
    
    def evaluate_on_queries(self,
                           test_queries: List[str],
                           save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate search engine on multiple test queries.
        
        Args:
            test_queries: List of test queries
            save_path: Optional path to save results
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating on {len(test_queries)} queries...")
        
        results = {
            'queries': [],
            'statistics': {
                'total_queries': len(test_queries),
                'avg_search_time': 0.0
            }
        }
        
        total_time = 0.0
        
        for query in test_queries:
            start = time.time()
            comparison = self.compare_retrievers(query)
            elapsed = time.time() - start
            total_time += elapsed
            
            results['queries'].append({
                'query': query,
                'results': comparison,
                'search_time': elapsed
            })
        
        results['statistics']['avg_search_time'] = total_time / len(test_queries)
        
        if save_path:
            save_json(results, save_path)
            logger.info(f"Results saved to {save_path}")
        
        return results
    
    def save_indices(self) -> None:
        """Save all indices to disk."""
        logger.info("Saving indices...")
        
        # Save BM25
        bm25_path = self.model_dir / "bm25_index.pkl"
        self.bm25_retriever.save(str(bm25_path))
        
        # Save Semantic
        self.semantic_retriever.save(str(self.model_dir))
        
        logger.info(f"Indices saved to {self.model_dir}")
    
    def load_indices(self) -> None:
        """Load all indices from disk."""
        logger.info("Loading indices...")
        
        # Load data first
        self.corpus, self.metadata = self.data_processor.preprocess_data()
        
        # Load BM25
        bm25_path = self.model_dir / "bm25_index.pkl"
        if bm25_path.exists():
            self.bm25_retriever.load(str(bm25_path))
        else:
            logger.warning(f"BM25 index not found at {bm25_path}")
        
        # Load Semantic
        if (self.model_dir / "faiss_index.bin").exists():
            self.semantic_retriever.load(str(self.model_dir))
        else:
            logger.warning(f"Semantic index not found in {self.model_dir}")
        
        self.is_built = True
        logger.info("Indices loaded successfully")
    
    def _print_statistics(self) -> None:
        """Print engine statistics."""
        logger.info("="*80)
        logger.info("Search Engine Statistics")
        logger.info("="*80)
        
        # Data statistics
        data_stats = self.data_processor.get_statistics()
        logger.info(f"Total Documents: {data_stats['total_documents']}")
        logger.info(f"Categories: {len(data_stats['categories'])}")
        logger.info(f"Difficulty Levels: {len(data_stats['difficulty_levels'])}")
        
        # BM25 statistics
        bm25_stats = self.bm25_retriever.get_statistics()
        logger.info(f"BM25 Vocabulary Size: {bm25_stats.get('vocabulary_size', 'N/A')}")
        
        # Semantic statistics
        semantic_stats = self.semantic_retriever.get_statistics()
        logger.info(f"Semantic Embedding Dim: {semantic_stats.get('embedding_dim', 'N/A')}")
        logger.info(f"Semantic Index Type: {semantic_stats.get('index_type', 'N/A')}")
        
        logger.info("="*80)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            Dictionary with system information
        """
        return {
            'data': self.data_processor.get_statistics(),
            'bm25': self.bm25_retriever.get_statistics(),
            'semantic': self.semantic_retriever.get_statistics(),
            'is_built': self.is_built,
            'model_directory': str(self.model_dir)
        }
    
    def __repr__(self) -> str:
        """String representation."""
        status = "ready" if self.is_built else "not_built"
        return f"HybridSearchEngine(status={status}, docs={len(self.corpus)})"
