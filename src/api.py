"""
FastAPI REST API for Hybrid Search System
Provides endpoints for Lexical Search, Semantic Search, Hybrid Search, and Statistics
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
from pathlib import Path
import logging

from .search_engine import HybridSearchEngine
from .utils import setup_logger

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid Search API",
    description="REST API for Lexical, Semantic, and Hybrid Search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global search engine instance
search_engine: Optional[HybridSearchEngine] = None


# ==================== Request/Response Models ====================

class SearchRequest(BaseModel):
    """Base search request model."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(5, ge=1, le=100, description="Number of results to return")


class LexicalSearchRequest(SearchRequest):
    """Lexical (BM25) search request."""
    pass


class SemanticSearchRequest(SearchRequest):
    """Semantic search request."""
    pass


class HybridSearchRequest(SearchRequest):
    """Hybrid search request."""
    retrieve_k: int = Field(50, ge=10, le=200, description="Number of candidates to retrieve before re-ranking")


class SearchResult(BaseModel):
    """Individual search result."""
    rank: int
    id: str
    score: float
    title: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None
    content_preview: Optional[str] = None


class SearchResponse(BaseModel):
    """General search response."""
    query: str
    search_type: str
    search_time_ms: float
    total_results: int
    results: List[SearchResult]
    max_score: Optional[float] = None
    min_score: Optional[float] = None


class SystemStats(BaseModel):
    """System statistics."""
    total_documents: int
    vocabulary_size: Optional[int] = None
    embedding_dimension: Optional[int] = None
    categories: List[str]
    difficulty_levels: List[str]
    avg_document_length: Optional[float] = None


class IndexStats(BaseModel):
    """Index statistics."""
    bm25_status: str
    bm25_corpus_size: int
    bm25_vocabulary: int
    bm25_avg_doc_length: float
    semantic_status: str
    semantic_corpus_size: int
    semantic_embedding_dim: int
    semantic_index_type: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    is_ready: bool
    message: str


class ComparissonAnalysis(BaseModel):
    """Comparison analysis between search methods."""
    bm25_semantic_overlap: int
    bm25_hybrid_overlap: int
    semantic_hybrid_overlap: int
    all_three_overlap: int


class ComparisonResponse(BaseModel):
    """Comparison response with results from all methods."""
    query: str
    comparison_time_ms: float
    bm25_results: List[SearchResult]
    semantic_results: List[SearchResult]
    hybrid_results: List[SearchResult]
    overlap_analysis: ComparissonAnalysis


class StatisticsResponse(BaseModel):
    """Complete statistics response."""
    system_stats: SystemStats
    index_stats: IndexStats
    engine_status: str


# ==================== Initialization ====================

@app.on_event("startup")
async def startup_event():
    """Initialize search engine on startup."""
    global search_engine
    
    try:
        logger.info("Initializing Hybrid Search Engine...")
        
        DATA_PATH = "data/semantic_search_dataset_2000.csv"
        MODEL_DIR = "models"
        
        search_engine = HybridSearchEngine(
            data_path=DATA_PATH,
            model_dir=MODEL_DIR,
            bi_encoder_model="sentence-transformers/all-MiniLM-L6-v2",
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu"
        )
        
        # Build indices if not already built
        logger.info("Building/Loading search indices...")
        search_engine.build_indices(force_rebuild=False)
        
        logger.info("✅ Search Engine initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize search engine: {str(e)}")
        raise


# ==================== Health Check ====================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        Health status of the API
    """
    is_ready = search_engine is not None and search_engine.is_built
    
    return HealthResponse(
        status="healthy" if is_ready else "unavailable",
        is_ready=is_ready,
        message="API is ready for searches" if is_ready else "API is initializing..."
    )


# ==================== API Endpoints ====================

@app.post("/api/v1/search/lexical", response_model=SearchResponse, tags=["Search"])
async def lexical_search(request: LexicalSearchRequest) -> SearchResponse:
    """
    Lexical Search using BM25 algorithm.
    
    Performs keyword-based search considering:
    - Term frequency (TF)
    - Inverse document frequency (IDF)
    - Document length normalization
    
    Args:
        request: Search request with query and top_k
        
    Returns:
        Search results with BM25 scores
    """
    if search_engine is None or not search_engine.is_built:
        raise HTTPException(
            status_code=503,
            detail="Search engine not ready. Please try again later."
        )
    
    try:
        start_time = time.time()
        
        results = search_engine.search(
            query=request.query,
            top_k=request.top_k,
            search_type='bm25'
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Format results
        formatted_results = [
            SearchResult(
                rank=result['rank'],
                id=result['id'],
                score=result['score'],
                title=result.get('title'),
                category=result.get('category'),
                difficulty=result.get('difficulty'),
                content_preview=result.get('content', '')[:200] + '...'
            )
            for result in results
        ]
        
        scores = [r.score for r in formatted_results] if formatted_results else [0]
        
        return SearchResponse(
            query=request.query,
            search_type='lexical',
            search_time_ms=elapsed_ms,
            total_results=len(formatted_results),
            results=formatted_results,
            max_score=max(scores),
            min_score=min(scores)
        )
        
    except Exception as e:
        logger.error(f"Error in lexical search: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Search failed: {str(e)}"
        )


@app.post("/api/v1/search/semantic", response_model=SearchResponse, tags=["Search"])
async def semantic_search(request: SemanticSearchRequest) -> SearchResponse:
    """
    Semantic Search using Dense Vector Embeddings.
    
    Performs semantic similarity search using:
    - Bi-Encoder model for dense embeddings
    - FAISS for fast approximate nearest neighbor search
    - Cosine similarity scoring
    
    Args:
        request: Search request with query and top_k
        
    Returns:
        Search results with semantic similarity scores
    """
    if search_engine is None or not search_engine.is_built:
        raise HTTPException(
            status_code=503,
            detail="Search engine not ready. Please try again later."
        )
    
    try:
        start_time = time.time()
        
        results = search_engine.search(
            query=request.query,
            top_k=request.top_k,
            search_type='semantic'
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Format results
        formatted_results = [
            SearchResult(
                rank=result['rank'],
                id=result['id'],
                score=result['score'],
                title=result.get('title'),
                category=result.get('category'),
                difficulty=result.get('difficulty'),
                content_preview=result.get('content', '')[:200] + '...'
            )
            for result in results
        ]
        
        scores = [r.score for r in formatted_results] if formatted_results else [0]
        
        return SearchResponse(
            query=request.query,
            search_type='semantic',
            search_time_ms=elapsed_ms,
            total_results=len(formatted_results),
            results=formatted_results,
            max_score=max(scores),
            min_score=min(scores)
        )
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Search failed: {str(e)}"
        )


@app.post("/api/v1/search/hybrid", response_model=SearchResponse, tags=["Search"])
async def hybrid_search(request: HybridSearchRequest) -> SearchResponse:
    """
    Hybrid Search combining Lexical and Semantic approaches.
    
    Uses a multi-stage pipeline:
    1. **Candidate Retrieval**: BM25 + Semantic search (top K results)
    2. **Fusion**: Reciprocal Rank Fusion (RRF) combines rankings
    3. **Re-ranking**: Cross-Encoder model re-ranks final results
    
    This approach leverages strengths of both methods:
    - BM25: Exact keyword matching
    - Semantic: Meaning-based similarity
    - RRF: Balanced fusion
    - Cross-Encoder: Fine-grained relevance scoring
    
    Args:
        request: Search request with query, top_k, and retrieve_k
        
    Returns:
        Optimized search results combining both methods
    """
    if search_engine is None or not search_engine.is_built:
        raise HTTPException(
            status_code=503,
            detail="Search engine not ready. Please try again later."
        )
    
    try:
        start_time = time.time()
        
        results = search_engine.search(
            query=request.query,
            top_k=request.top_k,
            search_type='hybrid',
            retrieve_k=request.retrieve_k
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Format results
        formatted_results = [
            SearchResult(
                rank=result['rank'],
                id=result['id'],
                score=result['score'],
                title=result.get('title'),
                category=result.get('category'),
                difficulty=result.get('difficulty'),
                content_preview=result.get('content', '')[:200] + '...'
            )
            for result in results
        ]
        
        scores = [r.score for r in formatted_results] if formatted_results else [0]
        
        return SearchResponse(
            query=request.query,
            search_type='hybrid',
            search_time_ms=elapsed_ms,
            total_results=len(formatted_results),
            results=formatted_results,
            max_score=max(scores),
            min_score=min(scores)
        )
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Search failed: {str(e)}"
        )


# ==================== Statistics Endpoint ====================

@app.get("/api/v1/statistics", response_model=StatisticsResponse, tags=["Statistics"])
async def get_statistics() -> StatisticsResponse:
    """
    Get comprehensive system statistics.
    
    Returns detailed information about:
    - **System Stats**: Documents, categories, difficulty levels, vocabulary
    - **Index Stats**: BM25 and Semantic index information
    - **Engine Status**: Overall system health
    
    Returns:
        Complete statistics about the search system
    """
    if search_engine is None or not search_engine.is_built:
        raise HTTPException(
            status_code=503,
            detail="Search engine not ready. Please try again later."
        )
    
    try:
        # Get system info
        sys_info = search_engine.get_system_info()
        
        # Extract statistics
        data_stats = sys_info['data']
        bm25_stats = sys_info['bm25']
        semantic_stats = sys_info['semantic']
        
        # Build response
        system_stats = SystemStats(
            total_documents=data_stats.get('total_documents', 0),
            vocabulary_size=bm25_stats.get('vocabulary_size'),
            embedding_dimension=semantic_stats.get('embedding_dim'),
            categories=data_stats.get('categories', []),
            difficulty_levels=data_stats.get('difficulty_levels', []),
            avg_document_length=bm25_stats.get('avg_document_length')
        )
        
        index_stats = IndexStats(
            bm25_status=bm25_stats.get('status', 'unknown'),
            bm25_corpus_size=bm25_stats.get('corpus_size', 0),
            bm25_vocabulary=bm25_stats.get('vocabulary_size', 0),
            bm25_avg_doc_length=bm25_stats.get('avg_document_length', 0.0),
            semantic_status=semantic_stats.get('status', 'unknown'),
            semantic_corpus_size=semantic_stats.get('corpus_size', 0),
            semantic_embedding_dim=semantic_stats.get('embedding_dim', 0),
            semantic_index_type=semantic_stats.get('index_type', 'unknown')
        )
        
        return StatisticsResponse(
            system_stats=system_stats,
            index_stats=index_stats,
            engine_status="ready" if sys_info['is_built'] else "not_ready"
        )
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@app.post("/api/v1/compare", response_model=ComparisonResponse, tags=["Statistics"])
async def compare_search_methods(request: SearchRequest) -> ComparisonResponse:
    """
    Compare results from all three search methods.
    
    This endpoint runs all three search approaches on the same query and:
    - Returns results from BM25, Semantic, and Hybrid methods
    - Analyzes overlap between different methods
    - Helps understand method strengths and weaknesses
    
    Args:
        request: Search request with query and top_k
        
    Returns:
        Comparison results showing all three methods side-by-side
    """
    if search_engine is None or not search_engine.is_built:
        raise HTTPException(
            status_code=503,
            detail="Search engine not ready. Please try again later."
        )
    
    try:
        start_time = time.time()
        
        comparison = search_engine.compare_retrievers(
            query=request.query,
            top_k=request.top_k
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        def format_results(results_list):
            return [
                SearchResult(
                    rank=result['rank'],
                    id=result['id'],
                    score=result['score'],
                    title=result.get('title'),
                    category=result.get('category'),
                    difficulty=result.get('difficulty'),
                    content_preview=result.get('content', '')[:200] + '...'
                )
                for result in results_list
            ]
        
        overlap = comparison['overlap_analysis']
        overlap_analysis = ComparissonAnalysis(
            bm25_semantic_overlap=overlap['bm25_semantic_overlap'],
            bm25_hybrid_overlap=overlap['bm25_hybrid_overlap'],
            semantic_hybrid_overlap=overlap['semantic_hybrid_overlap'],
            all_three_overlap=overlap['all_three_overlap']
        )
        
        return ComparisonResponse(
            query=request.query,
            comparison_time_ms=elapsed_ms,
            bm25_results=format_results(comparison['bm25']),
            semantic_results=format_results(comparison['semantic']),
            hybrid_results=format_results(comparison['hybrid']),
            overlap_analysis=overlap_analysis
        )
        
    except Exception as e:
        logger.error(f"Error in comparison: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Comparison failed: {str(e)}"
        )


# ==================== Root Endpoint ====================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Hybrid Search API",
        "version": "1.0.0",
        "description": "REST API for Lexical, Semantic, and Hybrid Search",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "lexical_search": "/api/v1/search/lexical",
            "semantic_search": "/api/v1/search/semantic",
            "hybrid_search": "/api/v1/search/hybrid",
            "statistics": "/api/v1/statistics",
            "compare": "/api/v1/compare"
        }
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("🚀 Starting Hybrid Search API Server")
    print("="*80)
    print("\n📖 API Documentation: http://localhost:8000/docs")
    print("\n✅ Server is running... Press Ctrl+C to stop\n")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)