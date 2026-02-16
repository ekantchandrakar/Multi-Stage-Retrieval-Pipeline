"""
Hybrid Fusion using Reciprocal Rank Fusion (RRF)
Combines rankings from multiple retrievers
"""
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import numpy as np

from .utils import setup_logger, timing_decorator

logger = setup_logger(__name__)


class HybridFusion:
    """
    Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.
    
    RRF Formula:
        RRF(d) = Σ_r 1/(k + rank_r(d))
    
    where:
        - d is a document
        - r iterates over all retrievers
        - rank_r(d) is the rank of document d in retriever r's results
        - k is a constant (typically 60)
    
    Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
    and individual Rank Learning Methods" (SIGIR 2009)
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize Hybrid Fusion.
        
        Args:
            k: Constant for RRF formula (default: 60)
               - Higher k reduces the impact of lower-ranked documents
               - Standard value from literature: 60
        """
        self.k = k
        logger.info(f"Initialized HybridFusion with k={k}")
    
    @timing_decorator
    def fuse(self, 
            *ranked_lists: List[Tuple[int, float]],
            top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Fuse multiple ranked lists using RRF.
        
        Args:
            *ranked_lists: Variable number of ranked lists
                          Each list contains (doc_id, score) tuples
            top_k: Number of top results to return
            
        Returns:
            Fused ranked list of (doc_id, rrf_score) tuples
        """
        if not ranked_lists:
            return []
        
        logger.debug(f"Fusing {len(ranked_lists)} ranked lists")
        
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        
        for ranked_list in ranked_lists:
            for rank, (doc_id, _) in enumerate(ranked_list, start=1):
                # RRF formula: 1 / (k + rank)
                rrf_scores[doc_id] += 1.0 / (self.k + rank)
        
        # Sort by RRF score
        fused_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        logger.debug(f"Fused results: {len(fused_results)} documents")
        
        return fused_results
    
    @timing_decorator
    def weighted_fuse(self,
                     ranked_lists: List[List[Tuple[int, float]]],
                     weights: List[float],
                     top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Fuse multiple ranked lists with custom weights.
        
        Args:
            ranked_lists: List of ranked lists
            weights: Weight for each ranked list (should sum to 1.0)
            top_k: Number of top results to return
            
        Returns:
            Fused ranked list of (doc_id, weighted_rrf_score) tuples
        """
        if len(ranked_lists) != len(weights):
            raise ValueError("Number of ranked lists must match number of weights")
        
        if not np.isclose(sum(weights), 1.0):
            logger.warning(f"Weights sum to {sum(weights)}, normalizing...")
            weights = [w / sum(weights) for w in weights]
        
        logger.debug(f"Weighted fusion with {len(ranked_lists)} lists and weights: {weights}")
        
        # Calculate weighted RRF scores
        rrf_scores = defaultdict(float)
        
        for ranked_list, weight in zip(ranked_lists, weights):
            for rank, (doc_id, _) in enumerate(ranked_list, start=1):
                rrf_scores[doc_id] += weight * (1.0 / (self.k + rank))
        
        # Sort by weighted RRF score
        fused_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return fused_results
    
    def fuse_with_scores(self,
                        *ranked_lists: List[Tuple[int, float]],
                        top_k: int = 50,
                        score_weight: float = 0.0) -> List[Tuple[int, float]]:
        """
        Fuse ranked lists considering both ranks and original scores.
        
        Args:
            *ranked_lists: Variable number of ranked lists
            top_k: Number of top results to return
            score_weight: Weight for original scores (0.0 to 1.0)
                         - 0.0: Pure RRF (rank-based only)
                         - 1.0: Pure score-based
                         - 0.5: Equal contribution
            
        Returns:
            Fused ranked list
        """
        if not (0.0 <= score_weight <= 1.0):
            raise ValueError("score_weight must be between 0.0 and 1.0")
        
        logger.debug(f"Fusing with score_weight={score_weight}")
        
        # Calculate combined scores
        combined_scores = defaultdict(float)
        score_contributions = defaultdict(list)
        
        for ranked_list in ranked_lists:
            # Normalize scores in this list
            if ranked_list:
                scores = [score for _, score in ranked_list]
                max_score = max(scores) if scores else 1.0
                min_score = min(scores) if scores else 0.0
                score_range = max_score - min_score if max_score > min_score else 1.0
            
            for rank, (doc_id, score) in enumerate(ranked_list, start=1):
                # RRF component
                rrf_component = (1 - score_weight) * (1.0 / (self.k + rank))
                
                # Score component (normalized)
                if ranked_list:
                    normalized_score = (score - min_score) / score_range
                    score_component = score_weight * normalized_score
                else:
                    score_component = 0.0
                
                combined_scores[doc_id] += rrf_component + score_component
                score_contributions[doc_id].append(score)
        
        # Sort by combined score
        fused_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return fused_results
    
    def explain_fusion(self,
                      doc_id: int,
                      *ranked_lists: List[Tuple[int, float]]) -> Dict[str, Any]:
        """
        Explain how a document's RRF score was calculated.
        
        Args:
            doc_id: Document ID to explain
            *ranked_lists: Ranked lists used in fusion
            
        Returns:
            Dictionary with fusion explanation
        """
        explanation = {
            'document_id': doc_id,
            'k': self.k,
            'retrievers': []
        }
        
        total_rrf = 0.0
        
        for i, ranked_list in enumerate(ranked_lists, start=1):
            # Find document in this ranked list
            rank = None
            original_score = None
            
            for r, (d_id, score) in enumerate(ranked_list, start=1):
                if d_id == doc_id:
                    rank = r
                    original_score = score
                    break
            
            if rank is not None:
                rrf_contribution = 1.0 / (self.k + rank)
                total_rrf += rrf_contribution
                
                explanation['retrievers'].append({
                    'retriever': f'Retriever_{i}',
                    'rank': rank,
                    'original_score': float(original_score),
                    'rrf_contribution': float(rrf_contribution)
                })
            else:
                explanation['retrievers'].append({
                    'retriever': f'Retriever_{i}',
                    'rank': None,
                    'original_score': None,
                    'rrf_contribution': 0.0
                })
        
        explanation['total_rrf_score'] = float(total_rrf)
        
        return explanation
    
    def get_overlap(self, *ranked_lists: List[Tuple[int, float]],
                   top_k: int = 50) -> Dict[str, Any]:
        """
        Analyze overlap between different ranked lists.
        
        Args:
            *ranked_lists: Ranked lists to analyze
            top_k: Consider top-k results from each list
            
        Returns:
            Dictionary with overlap statistics
        """
        if not ranked_lists:
            return {}
        
        # Get top-k doc IDs from each list
        doc_sets = []
        for ranked_list in ranked_lists:
            doc_ids = set(doc_id for doc_id, _ in ranked_list[:top_k])
            doc_sets.append(doc_ids)
        
        # Calculate overlap
        all_docs = set.union(*doc_sets) if doc_sets else set()
        common_docs = set.intersection(*doc_sets) if doc_sets else set()
        
        analysis = {
            'num_retrievers': len(ranked_lists),
            'top_k': top_k,
            'total_unique_documents': len(all_docs),
            'documents_in_all_retrievers': len(common_docs),
            'overlap_percentage': (len(common_docs) / len(all_docs) * 100) if all_docs else 0.0
        }
        
        # Pairwise overlap
        if len(doc_sets) == 2:
            set1, set2 = doc_sets
            analysis['pairwise_overlap'] = len(set1 & set2)
            analysis['only_in_retriever_1'] = len(set1 - set2)
            analysis['only_in_retriever_2'] = len(set2 - set1)
        
        return analysis
    
    def __repr__(self) -> str:
        """String representation."""
        return f"HybridFusion(k={self.k})"
