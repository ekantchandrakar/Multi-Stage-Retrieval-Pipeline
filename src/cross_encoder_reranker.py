"""
Cross-Encoder Re-ranker for precision optimization
Provides fine-grained relevance scoring for final ranking
"""
from typing import List, Tuple, Dict, Any
import numpy as np

from sentence_transformers import CrossEncoder

from .utils import setup_logger, timing_decorator, batch_iterator

logger = setup_logger(__name__)


class CrossEncoderReranker:
    """
    Cross-Encoder based re-ranker for fine-grained relevance scoring.
    
    Unlike Bi-Encoders that encode query and document separately,
    Cross-Encoders process (query, document) pairs together, allowing
    for more nuanced relevance judgments at the cost of slower inference.
    
    Typical use: Re-rank top-k candidates from initial retrieval.
    """
    
    def __init__(self,
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str = "cpu",
                 max_length: int = 512):
        """
        Initialize Cross-Encoder Reranker.
        
        Args:
            model_name: Name of the cross-encoder model
            device: Device to use ('cpu' or 'cuda')
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        # Load model
        logger.info(f"Loading Cross-Encoder model: {model_name}")
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device
        )
        
        logger.info(f"Initialized CrossEncoderReranker with model {model_name}")
    
    @timing_decorator
    def rerank(self,
              query: str,
              documents: List[str],
              doc_ids: List[int],
              top_k: int = 5,
              batch_size: int = 16) -> List[Tuple[int, float]]:
        """
        Re-rank documents using Cross-Encoder.
        
        Args:
            query: Search query
            documents: List of document texts
            doc_ids: List of document IDs (same order as documents)
            top_k: Number of top results to return
            batch_size: Batch size for inference
            
        Returns:
            Re-ranked list of (doc_id, relevance_score) tuples
        """
        if len(documents) != len(doc_ids):
            raise ValueError("documents and doc_ids must have same length")
        
        if not documents:
            return []
        
        logger.debug(f"Re-ranking {len(documents)} documents for query: '{query}'")
        
        # Create (query, document) pairs
        pairs = [[query, doc] for doc in documents]
        
        # Predict relevance scores
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=False
        )
        
        # Combine doc_ids with scores and sort
        results = list(zip(doc_ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        top_results = results[:top_k]
        
        # Convert scores to float
        top_results = [(int(doc_id), float(score)) for doc_id, score in top_results]
        
        logger.debug(f"Re-ranking complete. Top score: {top_results[0][1]:.4f}")
        
        return top_results
    
    def batch_rerank(self,
                    queries: List[str],
                    documents_list: List[List[str]],
                    doc_ids_list: List[List[int]],
                    top_k: int = 5,
                    batch_size: int = 16) -> List[List[Tuple[int, float]]]:
        """
        Batch re-ranking for multiple queries.
        
        Args:
            queries: List of queries
            documents_list: List of document lists (one per query)
            doc_ids_list: List of doc_id lists (one per query)
            top_k: Number of results per query
            batch_size: Batch size for inference
            
        Returns:
            List of re-ranked results for each query
        """
        logger.info(f"Batch re-ranking for {len(queries)} queries")
        
        results = []
        for query, documents, doc_ids in zip(queries, documents_list, doc_ids_list):
            query_results = self.rerank(query, documents, doc_ids, top_k, batch_size)
            results.append(query_results)
        
        return results
    
    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single (query, document) pair.
        
        Args:
            query: Search query
            document: Document text
            
        Returns:
            Relevance score
        """
        score = self.model.predict([[query, document]])[0]
        return float(score)
    
    def score_pairs(self,
                   pairs: List[Tuple[str, str]],
                   batch_size: int = 16) -> List[float]:
        """
        Score multiple (query, document) pairs.
        
        Args:
            pairs: List of (query, document) tuples
            batch_size: Batch size for inference
            
        Returns:
            List of relevance scores
        """
        if not pairs:
            return []
        
        # Convert to list of lists
        pair_lists = [[query, doc] for query, doc in pairs]
        
        # Predict scores
        scores = self.model.predict(
            pair_lists,
            batch_size=batch_size,
            show_progress_bar=False
        )
        
        return [float(score) for score in scores]
    
    def explain_score(self,
                     query: str,
                     document: str,
                     show_tokens: bool = False) -> Dict[str, Any]:
        """
        Explain the relevance score for a (query, document) pair.
        
        Note: This is a basic explanation. True token-level attribution
        would require additional techniques like attention visualization.
        
        Args:
            query: Search query
            document: Document text
            show_tokens: Whether to show tokenization (for debugging)
            
        Returns:
            Dictionary with score explanation
        """
        score = self.score_pair(query, document)
        
        explanation = {
            'query': query,
            'document': document[:500] + "..." if len(document) > 500 else document,
            'relevance_score': score,
            'model': self.model_name,
            'interpretation': self._interpret_score(score)
        }
        
        if show_tokens:
            # This is approximate - actual tokenization depends on model
            explanation['query_length'] = len(query.split())
            explanation['document_length'] = len(document.split())
            explanation['total_length'] = explanation['query_length'] + explanation['document_length']
            explanation['exceeds_max_length'] = explanation['total_length'] > self.max_length
        
        return explanation
    
    def _interpret_score(self, score: float) -> str:
        """
        Provide human-readable interpretation of relevance score.
        
        Args:
            score: Relevance score
            
        Returns:
            Interpretation string
        """
        # These thresholds are approximate and model-dependent
        if score > 5.0:
            return "Highly relevant"
        elif score > 2.0:
            return "Very relevant"
        elif score > 0.0:
            return "Relevant"
        elif score > -2.0:
            return "Somewhat relevant"
        else:
            return "Not relevant"
    
    def compare_documents(self,
                         query: str,
                         doc1: str,
                         doc2: str) -> Dict[str, Any]:
        """
        Compare relevance of two documents for a query.
        
        Args:
            query: Search query
            doc1: First document
            doc2: Second document
            
        Returns:
            Comparison results
        """
        score1 = self.score_pair(query, doc1)
        score2 = self.score_pair(query, doc2)
        
        comparison = {
            'query': query,
            'document_1_score': score1,
            'document_2_score': score2,
            'score_difference': score1 - score2,
            'more_relevant': 'Document 1' if score1 > score2 else 'Document 2',
            'confidence': abs(score1 - score2)
        }
        
        return comparison
    
    def get_score_distribution(self,
                              query: str,
                              documents: List[str]) -> Dict[str, Any]:
        """
        Get distribution statistics for document scores.
        
        Args:
            query: Search query
            documents: List of documents
            
        Returns:
            Distribution statistics
        """
        if not documents:
            return {}
        
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        distribution = {
            'num_documents': len(documents),
            'mean_score': float(np.mean(scores)),
            'median_score': float(np.median(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'score_range': float(np.max(scores) - np.min(scores))
        }
        
        # Score percentiles
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            distribution[f'percentile_{p}'] = float(np.percentile(scores, p))
        
        return distribution
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get Cross-Encoder statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'model_name': self.model_name,
            'device': self.device,
            'max_length': self.max_length,
            'model_type': 'CrossEncoder'
        }
        
        return stats
    
    def __repr__(self) -> str:
        """String representation."""
        return f"CrossEncoderReranker(model={self.model_name}, device={self.device})"
