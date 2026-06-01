# 🚀 Hybrid Semantic Search System v2

A production-grade hybrid retrieval system combining lexical (BM25) and semantic (Transformer + IVF-FAISS) search with cross-encoder re-ranking and full IR evaluation.

## What's New in v2

| Feature            | Detail                                                                                       |
| ------------------ | -------------------------------------------------------------------------------------------- |
| **StaQC Dataset**  | Replaces the 2 K CSV — loads up to 100 K+ records from `koutch/staqc` on HuggingFace         |
| **Adaptive FAISS** | Auto-selects `IndexFlatIP` (< 10 K docs), `IndexIVFFlat` (< 500 K), or `IndexIVFPQ` (500 K+) |
| **IR Evaluation**  | `NDCG@10`, `MRR`, `Recall@10`, `Precision@10`, `MAP` via `EvaluationSuite`                   |

## Architecture

```
Query
  │
  ├─► BM25Retriever          (lexical, top-50)
  │
  ├─► SemanticRetriever      (IVF-FAISS, top-50)
  │         └─ auto: Flat / IVFFlat / IVFPQ
  │
  ├─► HybridFusion (RRF)     (fuse → top-50)
  │
  └─► CrossEncoderReranker   (re-rank → top-5)
```

## Project Structure

```
hybrid_search_system/
├── main.py                      # Full demo (dataset + IVF + metrics)
├── evaluate.py                  # Standalone IR evaluation runner  ← NEW
├── requirements.txt
├── setup.py
│
├── configs/
│   ├── model_config.yaml        # IVF nlist/nprobe settings added
│   └── search_config.yaml       # Dataset + evaluation config added
│
├── src/
│   ├── __init__.py              # Exports all public classes
│   ├── dataset_loader.py        # NEW – StaQC HuggingFace loader + DataProcessor
│   ├── bm25_retriever.py        # Updated – camelCase tokeniser
│   ├── semantic_retriever.py    # Updated – adaptive FAISS (Flat/IVF/IVFPQ)
│   ├── hybrid_fusion.py         # Updated – cleaner weighted fuse
│   ├── cross_encoder_reranker.py# Unchanged interface
│   ├── search_engine.py         # Updated – StaQC + IVF wired in
│   ├── evaluation.py            # NEW – NDCG, MRR, Recall, MAP, EvaluationSuite
│   ├── utils.py                 # Updated – backwards-compatible
│   └── api.py                   # Updated – /admin/metrics endpoint added
│
└── tests/
    └── test_hybrid_search.py    # Updated – covers all 3 new features
```

## Quick Start

```bash
# 1. Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run full demo (downloads StaQC, builds IVF index, prints metrics)
python main.py

# 3. Run evaluation benchmark
python evaluate.py --queries 10 --k 10

# 4. Start API server
uvicorn src.api:app --reload --port 8000
```

## Feature 1 – StaQC Dataset (2 K → 100 K)

```python
from src.search_engine import HybridSearchEngine

engine = HybridSearchEngine(
    data_path="staqc",      # triggers HuggingFace download
    language="python",      # or "sql"
    max_records=50_000,     # cap records; 0 = no cap
    model_dir="models",
)
engine.build_indices()
```

The dataset is cached locally as Parquet after the first download — subsequent starts are instant.

## Feature 2 – IR Evaluation Metrics

### One-liner per query

```python
from src.evaluation import IRMetrics

metrics = IRMetrics.evaluate_query(
    retrieved=["doc_42", "doc_7", "doc_13"],
    relevant={"doc_7", "doc_42"},
    k=10,
)
# → {'ndcg@10': 0.93, 'mrr': 1.0, 'recall@10': 1.0, 'precision@10': 0.3, 'ap': 1.0}
```

### Full benchmark

```python
from src.evaluation import EvaluationSuite, GroundTruthBuilder

# Build pseudo-relevance labels (or supply your own)
gtb = GroundTruthBuilder(engine.metadata)
ground_truth = gtb.build_from_keywords(queries, min_relevance=2)

suite = EvaluationSuite(engine, k=10)
report = suite.run(queries, ground_truth, search_types=["bm25","semantic","hybrid"])
suite.print_report(report)
suite.save_report(report, "outputs/eval.json")
```

**When to run:** offline, after changing models or corpus — never per API request.

## Feature 3 – Adaptive FAISS Index

| Corpus size  | Index type     | Notes                                |
| ------------ | -------------- | ------------------------------------ |
| < 10 K       | `IndexFlatIP`  | Exact, fastest for small sets        |
| 10 K – 500 K | `IndexIVFFlat` | Approximate, auto-tuned nlist/nprobe |
| ≥ 500 K      | `IndexIVFPQ`   | Compressed, memory-efficient         |

Override manually:

```python
engine = HybridSearchEngine(
    index_type="ivf",   # force IVFFlat regardless of corpus size
    nlist=256,          # override auto nlist
    nprobe=16,          # override auto nprobe
    ...
)
```

Tune nprobe at runtime (trade accuracy for speed):

```python
engine.semantic_retriever.set_nprobe(32)
```

## API Endpoints

| Method | Path                      | Description                                          |
| ------ | ------------------------- | ---------------------------------------------------- |
| POST   | `/api/v1/search/lexical`  | BM25 search                                          |
| POST   | `/api/v1/search/semantic` | IVF-FAISS semantic search                            |
| POST   | `/api/v1/search/hybrid`   | Full hybrid pipeline                                 |
| POST   | `/api/v1/compare`         | All three side-by-side                               |
| GET    | `/api/v1/statistics`      | Corpus + index stats                                 |
| GET    | `/api/v1/admin/metrics`   | Pre-computed NDCG/MRR/Recall (run evaluate.py first) |

## Evaluation Metrics — Interpretation

| Metric    | Formula                  | Good   | Excellent |
| --------- | ------------------------ | ------ | --------- |
| NDCG@10   | DCG / IDCG               | > 0.70 | > 0.85    |
| MRR       | 1 / rank(first relevant) | > 0.70 | > 0.85    |
| Recall@10 | hits / total relevant    | > 0.50 | > 0.75    |
| MAP       | mean AP over queries     | > 0.50 | > 0.70    |

## Running Tests

```bash
pytest tests/ -v --tb=short
pytest tests/ -v -k "TestSemanticRetrieverIVF"   # IVF tests only
pytest tests/ -v -k "TestNDCG or TestMRR"         # metric tests only
```

## Performance

| Component                     | Latency         |
| ----------------------------- | --------------- |
| BM25 search                   | ~10–20 ms       |
| Semantic (IVFFlat, 50 K docs) | ~15–30 ms       |
| RRF fusion                    | ~1–2 ms         |
| Cross-encoder (50 docs)       | ~100–200 ms     |
| **Total hybrid**              | **~130–250 ms** |

---

Built with ❤️ — BM25 + IVF-FAISS + Cross-Encoder + NDCG/MRR/Recall
