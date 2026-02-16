"""
Semantic Retriever using Bi-Encoder and FAISS
Implements dense vector search for semantic similarity
"""
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss

from .utils import setup_logger, timing_decorator, batch_iterator

logger = setup_logger(__name__)


class SemanticRetriever:
    """
    Semantic retriever using dense embeddings and FAISS for efficient similarity search.
    
    Uses a Bi-Encoder model to create document embeddings and FAISS for
    fast approximate nearest neighbor search.
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu",
                 normalize_embeddings: bool = True):
        """
        Initialize Semantic Retriever.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cpu' or 'cuda')
            normalize_embeddings: Whether to normalize embeddings (for cosine similarity)
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # FAISS index
        self.index = None
        self.embeddings = None
        self.corpus_size = 0
        
        logger.info(f"Initialized SemanticRetriever with model {model_name}")
    
    @timing_decorator
    def build_index(self, 
                   corpus: List[str], 
                   batch_size: int = 32,
                   show_progress: bool = True) -> None:
        """
        Build FAISS index from corpus.
        
        Args:
            corpus: List of document texts
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
        """
        logger.info(f"Building semantic index for {len(corpus)} documents")
        
        self.corpus_size = len(corpus)
        
        # Encode corpus
        logger.info("Encoding corpus...")
        self.embeddings = self.model.encode(
            corpus,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        logger.info(f"Encoded {len(self.embeddings)} documents to {self.embeddings.shape[1]}-dim vectors")
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        
        if self.normalize_embeddings:
            # Use Inner Product for normalized vectors (equivalent to cosine similarity)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            # Use L2 distance
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add embeddings to index
        self.index.add(self.embeddings.astype(np.float32))
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
    
    @timing_decorator
    def search(self, 
              query: str, 
              top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Search using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document_index, similarity_score) tuples sorted by score
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        # Search
        scores, indices = self.index.search(
            query_embedding.astype(np.float32),
            min(top_k, self.corpus_size)
        )
        
        # Convert to list of tuples
        results = [
            (int(idx), float(score)) 
            for idx, score in zip(indices[0], scores[0])
        ]
        
        logger.debug(f"Semantic search returned {len(results)} results for query: '{query}'")
        
        return results
    
    def batch_search(self, 
                    queries: List[str], 
                    top_k: int = 50,
                    batch_size: int = 32) -> List[List[Tuple[int, float]]]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            batch_size: Batch size for encoding queries
            
        Returns:
            List of search results for each query
        """
        logger.info(f"Performing batch semantic search for {len(queries)} queries")
        
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode all queries
        query_embeddings = self.model.encode(
            queries,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        # Search
        scores, indices = self.index.search(
            query_embeddings.astype(np.float32),
            min(top_k, self.corpus_size)
        )
        
        # Convert to list of results
        results = []
        for query_indices, query_scores in zip(indices, scores):
            query_results = [
                (int(idx), float(score)) 
                for idx, score in zip(query_indices, query_scores)
            ]
            results.append(query_results)
        
        return results
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        return embedding[0]
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        if self.normalize_embeddings:
            # Cosine similarity (dot product of normalized vectors)
            similarity = np.dot(emb1, emb2)
        else:
            # L2 distance converted to similarity
            distance = np.linalg.norm(emb1 - emb2)
            similarity = 1.0 / (1.0 + distance)
        
        return float(similarity)
    
    def get_nearest_neighbors(self, 
                             doc_index: int, 
                             top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find nearest neighbors for a document.
        
        Args:
            doc_index: Index of the document
            top_k: Number of neighbors to return
            
        Returns:
            List of (neighbor_index, similarity_score) tuples
        """
        if self.index is None or self.embeddings is None:
            raise ValueError("Index not built.")
        
        if not (0 <= doc_index < self.corpus_size):
            raise ValueError(f"Document index {doc_index} out of range")
        
        # Get embedding
        query_embedding = self.embeddings[doc_index:doc_index+1]
        
        # Search (top_k + 1 to exclude the document itself)
        scores, indices = self.index.search(
            query_embedding.astype(np.float32),
            top_k + 1
        )
        
        # Filter out the document itself and convert to list
        results = [
            (int(idx), float(score)) 
            for idx, score in zip(indices[0], scores[0])
            if idx != doc_index
        ][:top_k]
        
        return results
    
    def save(self, save_dir: str) -> None:
        """
        Save index and embeddings.
        
        Args:
            save_dir: Directory to save files
        """
        if self.index is None or self.embeddings is None:
            raise ValueError("No index to save. Build index first.")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = save_path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        logger.info(f"FAISS index saved to {index_path}")
        
        # Save embeddings
        embeddings_path = save_path / "embeddings.npy"
        np.save(embeddings_path, self.embeddings)
        logger.info(f"Embeddings saved to {embeddings_path}")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'device': self.device,
            'normalize_embeddings': self.normalize_embeddings,
            'embedding_dim': self.embedding_dim,
            'corpus_size': self.corpus_size
        }
        metadata_path = save_path / "semantic_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load(self, save_dir: str) -> None:
        """
        Load index and embeddings.
        
        Args:
            save_dir: Directory containing saved files
        """
        load_path = Path(save_dir)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Save directory not found: {save_dir}")
        
        # Load metadata
        metadata_path = load_path / "semantic_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Verify model compatibility
        if metadata['model_name'] != self.model_name:
            logger.warning(
                f"Loaded metadata model ({metadata['model_name']}) "
                f"differs from current model ({self.model_name})"
            )
        
        # Load FAISS index
        index_path = load_path / "faiss_index.bin"
        self.index = faiss.read_index(str(index_path))
        logger.info(f"FAISS index loaded from {index_path}")
        
        # Load embeddings
        embeddings_path = load_path / "embeddings.npy"
        self.embeddings = np.load(embeddings_path)
        logger.info(f"Embeddings loaded from {embeddings_path}")
        
        # Set attributes
        self.embedding_dim = metadata['embedding_dim']
        self.corpus_size = metadata['corpus_size']
        self.normalize_embeddings = metadata['normalize_embeddings']
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get semantic retriever statistics.
        
        Returns:
            Dictionary of statistics
        """
        if self.index is None:
            return {'status': 'not_built'}
        
        stats = {
            'status': 'built',
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'corpus_size': self.corpus_size,
            'normalize_embeddings': self.normalize_embeddings,
            'device': self.device,
            'index_type': type(self.index).__name__,
            'total_vectors': self.index.ntotal if self.index else 0
        }
        
        if self.embeddings is not None:
            stats['embedding_memory_mb'] = self.embeddings.nbytes / (1024 * 1024)
        
        return stats
    
    def __repr__(self) -> str:
        """String representation."""
        if self.index is None:
            return f"SemanticRetriever(model={self.model_name}, status=not_built)"
        return (f"SemanticRetriever(model={self.model_name}, "
                f"corpus_size={self.corpus_size}, dim={self.embedding_dim})")
