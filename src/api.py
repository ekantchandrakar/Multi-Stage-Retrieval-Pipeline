"""
FastAPI REST API for Hybrid Search System
Provides endpoints for Lexical, Semantic, Hybrid Search, Statistics, and Admin Metrics
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
import json
from pathlib import Path
import logging

from .search_engine import HybridSearchEngine
from .utils import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Hybrid Search API",
    description="REST API for Lexical, Semantic, and Hybrid Search with IR Metrics",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_engine: Optional[HybridSearchEngine] = None


# ── Request / Response models ──────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=100)

class HybridSearchRequest(SearchRequest):
    retrieve_k: int = Field(50, ge=10, le=200)

class SearchResult(BaseModel):
    rank: int
    id: str
    score: float
    title: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None
    content_preview: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    search_type: str
    search_time_ms: float
    total_results: int
    results: List[SearchResult]
    max_score: Optional[float] = None
    min_score: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    is_ready: bool
    message: str

class OverlapAnalysis(BaseModel):
    bm25_semantic_overlap: int
    bm25_hybrid_overlap: int
    semantic_hybrid_overlap: int
    all_three_overlap: int

class ComparisonResponse(BaseModel):
    query: str
    comparison_time_ms: float
    bm25_results: List[SearchResult]
    semantic_results: List[SearchResult]
    hybrid_results: List[SearchResult]
    overlap_analysis: OverlapAnalysis

class SystemStatsResponse(BaseModel):
    total_documents: int
    categories: Dict[str, int]
    difficulty_levels: Dict[str, int]
    bm25_vocabulary_size: Optional[int]
    faiss_index_type: Optional[str]
    faiss_nlist: Optional[int]
    faiss_nprobe: Optional[int]
    embedding_dim: Optional[int]
    engine_status: str


# ── Startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global search_engine
    try:
        logger.info("Initialising Hybrid Search Engine …")
        search_engine = HybridSearchEngine(
            data_path="staqc",
            model_dir="models",
            language="python",
            max_records=100_000,
            device="cpu",
        )
        search_engine.build_indices(force_rebuild=False)
        logger.info("Engine ready")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


# ── Helpers ────────────────────────────────────────────────────────────────

def _fmt(results: List[Dict[str, Any]]) -> List[SearchResult]:
    return [
        SearchResult(
            rank=r["rank"],
            id=str(r["id"]),
            score=r["score"],
            title=r.get("title"),
            category=r.get("category"),
            difficulty=r.get("difficulty"),
            content_preview=(r.get("body") or "")[:200] + "…",
        )
        for r in results
    ]

def _ready():
    if search_engine is None or not search_engine.is_built:
        raise HTTPException(status_code=503, detail="Search engine not ready.")


# ── Health ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    ready = search_engine is not None and search_engine.is_built
    return HealthResponse(
        status="healthy" if ready else "unavailable",
        is_ready=ready,
        message="Ready" if ready else "Initialising …",
    )


# ── Search endpoints ───────────────────────────────────────────────────────

@app.post("/api/v1/search/lexical", response_model=SearchResponse, tags=["Search"])
async def lexical_search(request: SearchRequest):
    """BM25 keyword search."""
    _ready()
    t0 = time.time()
    results = search_engine.search(request.query, top_k=request.top_k, search_type="bm25")
    ms = (time.time() - t0) * 1000
    fmt = _fmt(results)
    scores = [r.score for r in fmt] or [0.0]
    return SearchResponse(query=request.query, search_type="lexical",
                          search_time_ms=ms, total_results=len(fmt), results=fmt,
                          max_score=max(scores), min_score=min(scores))

@app.post("/api/v1/search/semantic", response_model=SearchResponse, tags=["Search"])
async def semantic_search(request: SearchRequest):
    """Dense bi-encoder + FAISS (IVF) search."""
    _ready()
    t0 = time.time()
    results = search_engine.search(request.query, top_k=request.top_k, search_type="semantic")
    ms = (time.time() - t0) * 1000
    fmt = _fmt(results)
    scores = [r.score for r in fmt] or [0.0]
    return SearchResponse(query=request.query, search_type="semantic",
                          search_time_ms=ms, total_results=len(fmt), results=fmt,
                          max_score=max(scores), min_score=min(scores))

@app.post("/api/v1/search/hybrid", response_model=SearchResponse, tags=["Search"])
async def hybrid_search(request: HybridSearchRequest):
    """BM25 + Semantic + RRF fusion + Cross-Encoder re-ranking."""
    _ready()
    t0 = time.time()
    results = search_engine.search(
        request.query, top_k=request.top_k,
        search_type="hybrid", retrieve_k=request.retrieve_k,
    )
    ms = (time.time() - t0) * 1000
    fmt = _fmt(results)
    scores = [r.score for r in fmt] or [0.0]
    return SearchResponse(query=request.query, search_type="hybrid",
                          search_time_ms=ms, total_results=len(fmt), results=fmt,
                          max_score=max(scores), min_score=min(scores))

@app.post("/api/v1/compare", response_model=ComparisonResponse, tags=["Search"])
async def compare_methods(request: SearchRequest):
    """Run all three methods side-by-side."""
    _ready()
    t0 = time.time()
    comp = search_engine.compare_retrievers(request.query, top_k=request.top_k)
    ms = (time.time() - t0) * 1000
    ov = comp["overlap_analysis"]
    return ComparisonResponse(
        query=request.query,
        comparison_time_ms=ms,
        bm25_results=_fmt(comp["bm25"]),
        semantic_results=_fmt(comp["semantic"]),
        hybrid_results=_fmt(comp["hybrid"]),
        overlap_analysis=OverlapAnalysis(**ov),
    )


# ── Statistics ─────────────────────────────────────────────────────────────

@app.get("/api/v1/statistics", response_model=SystemStatsResponse, tags=["Statistics"])
async def get_statistics():
    """Corpus and index statistics."""
    _ready()
    info = search_engine.get_system_info()
    d = info["data"]
    b = info["bm25"]
    s = info["semantic"]
    return SystemStatsResponse(
        total_documents=d.get("total_documents", 0),
        categories=d.get("categories", {}),
        difficulty_levels=d.get("difficulty_levels", {}),
        bm25_vocabulary_size=b.get("vocabulary_size"),
        faiss_index_type=s.get("index_type"),
        faiss_nlist=s.get("nlist"),
        faiss_nprobe=s.get("nprobe"),
        embedding_dim=s.get("embedding_dim"),
        engine_status="ready" if info["is_built"] else "not_ready",
    )


# ── Admin: pre-computed evaluation metrics ────────────────────────────────

@app.get("/api/v1/admin/metrics", tags=["Admin"])
async def get_evaluation_metrics():
    """
    Returns the last-run evaluation report (NDCG@10, MRR, Recall@10).

    This is a PRE-COMPUTED static JSON produced by running evaluate.py.
    It is never recomputed per-request.

    Run offline:
        python evaluate.py
    Then this endpoint serves the result.
    """
    summary_path = Path("outputs/evaluation_summary.json")
    report_path  = Path("outputs/evaluation_report.json")

    if summary_path.exists():
        with open(summary_path) as f:
            return JSONResponse(content=json.load(f))
    if report_path.exists():
        with open(report_path) as f:
            data = json.load(f)
            # Return only the aggregate section to keep response small
            return JSONResponse(content={
                "aggregate": data.get("aggregate", {}),
                "config":    data.get("config", {}),
                "note": "Run evaluate.py to refresh these metrics.",
            })

    return JSONResponse(
        status_code=404,
        content={
            "detail": "No evaluation report found.",
            "hint":   "Run:  python evaluate.py  to generate metrics.",
        },
    )


# ── Root ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Hybrid Search API v2",
        "endpoints": {
            "health":          "/health",
            "lexical_search":  "/api/v1/search/lexical",
            "semantic_search": "/api/v1/search/semantic",
            "hybrid_search":   "/api/v1/search/hybrid",
            "compare":         "/api/v1/compare",
            "statistics":      "/api/v1/statistics",
            "admin_metrics":   "/api/v1/admin/metrics  (pre-computed, run evaluate.py first)",
        },
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    print("\n📖  API docs: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
