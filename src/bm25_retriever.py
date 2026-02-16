"""
BM25 Retriever for Lexical Search
Implements Best Matching 25 algorithm for keyword-based retrieval
"""
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

from rank_bm25 import BM25Okapi

from .utils import setup_logger, tokenize_text, timing_decorator

logger = setup_logger(__name__)


class BM25Retriever:
    """
    BM25-based lexical retriever for keyword matching.
    
    BM25 (Best Matching 25) is a probabilistic ranking function that:
    - Considers term frequency (TF)
    - Applies inverse document frequency (IDF)
    - Includes document length normalization
    """
    
    def __init__(self, 
                 k1: float = 1.5,
                 b: float = 0.75,
                 epsilon: float = 0.25):
        """
        Initialize BM25 Retriever.
        
        Args:
            k1: Term frequency saturation parameter (default: 1.5)
                - Higher values increase the impact of term frequency
                - Typical range: 1.2 to 2.0
            b: Length normalization parameter (default: 0.75)
                - Controls how much document length affects ranking
                - Range: 0 (no normalization) to 1 (full normalization)
            epsilon: Floor value for IDF (default: 0.25)
                - Prevents zero IDF for very common terms
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.bm25 = None
        self.tokenized_corpus = []
        self.corpus_size = 0
        
        logger.info(f"Initialized BM25Retriever with k1={k1}, b={b}, epsilon={epsilon}")
    
    @timing_decorator
    def build_index(self, corpus: List[str]) -> None:
        """
        Build BM25 index from corpus.
        
        Args:
            corpus: List of document texts
        """
        logger.info(f"Building BM25 index for {len(corpus)} documents")
        
        # Tokenize corpus
        self.tokenized_corpus = [tokenize_text(doc) for doc in corpus]
        self.corpus_size = len(corpus)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon
        )
        
        logger.info(f"BM25 index built successfully with {self.corpus_size} documents")
    
    @timing_decorator
    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Search using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document_index, score) tuples sorted by score
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")
        
        # Tokenize query
        query_tokens = tokenize_text(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        
        results = [(int(idx), float(score)) for idx, score in zip(top_indices, top_scores)]
        
        logger.debug(f"BM25 search returned {len(results)} results for query: '{query}'")
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = 50) -> List[List[Tuple[int, float]]]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            
        Returns:
            List of search results for each query
        """
        logger.info(f"Performing batch BM25 search for {len(queries)} queries")
        
        results = []
        for query in queries:
            query_results = self.search(query, top_k)
            results.append(query_results)
        
        return results
    
    def get_term_frequencies(self, query: str) -> Dict[str, int]:
        """
        Get term frequencies in the corpus for query terms.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary mapping terms to their document frequencies
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built.")
        
        query_tokens = tokenize_text(query)
        term_freqs = {}
        
        for term in query_tokens:
            # Count documents containing the term
            doc_freq = sum(1 for doc_tokens in self.tokenized_corpus if term in doc_tokens)
            term_freqs[term] = doc_freq
        
        return term_freqs
    
    def explain_score(self, query: str, doc_index: int) -> Dict[str, Any]:
        """
        Explain BM25 score for a specific document.
        
        Args:
            query: Search query
            doc_index: Document index
            
        Returns:
            Dictionary with score explanation
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built.")
        
        query_tokens = tokenize_text(query)
        doc_tokens = self.tokenized_corpus[doc_index]
        
        # Calculate score components
        score = self.bm25.get_scores(query_tokens)[doc_index]
        
        # Get term-level details
        term_details = {}
        for term in query_tokens:
            term_freq = doc_tokens.count(term)
            doc_freq = sum(1 for doc in self.tokenized_corpus if term in doc)
            
            term_details[term] = {
                'term_frequency': term_freq,
                'document_frequency': doc_freq,
                'in_document': term in doc_tokens
            }
        
        explanation = {
            'document_index': doc_index,
            'total_score': float(score),
            'document_length': len(doc_tokens),
            'query_terms': query_tokens,
            'term_details': term_details
        }
        
        return explanation
    
    def save(self, filepath: str) -> None:
        """
        Save BM25 index to file.
        
        Args:
            filepath: Path to save the index
        """
        if self.bm25 is None:
            raise ValueError("No index to save. Build index first.")
        
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'bm25': self.bm25,
            'tokenized_corpus': self.tokenized_corpus,
            'corpus_size': self.corpus_size,
            'k1': self.k1,
            'b': self.b,
            'epsilon': self.epsilon
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"BM25 index saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load BM25 index from file.
        
        Args:
            filepath: Path to load the index from
        """
        load_path = Path(filepath)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")
        
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.bm25 = save_data['bm25']
        self.tokenized_corpus = save_data['tokenized_corpus']
        self.corpus_size = save_data['corpus_size']
        self.k1 = save_data['k1']
        self.b = save_data['b']
        self.epsilon = save_data['epsilon']
        
        logger.info(f"BM25 index loaded from {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get BM25 index statistics.
        
        Returns:
            Dictionary of statistics
        """
        if self.bm25 is None:
            return {'status': 'not_built'}
        
        vocab_size = len(set(token for doc in self.tokenized_corpus for token in doc))
        avg_doc_length = np.mean([len(doc) for doc in self.tokenized_corpus])
        
        stats = {
            'status': 'built',
            'corpus_size': self.corpus_size,
            'vocabulary_size': vocab_size,
            'avg_document_length': float(avg_doc_length),
            'parameters': {
                'k1': self.k1,
                'b': self.b,
                'epsilon': self.epsilon
            }
        }
        
        return stats
    
    def __repr__(self) -> str:
        """String representation."""
        if self.bm25 is None:
            return "BM25Retriever(status=not_built)"
        return f"BM25Retriever(corpus_size={self.corpus_size}, k1={self.k1}, b={self.b})"
