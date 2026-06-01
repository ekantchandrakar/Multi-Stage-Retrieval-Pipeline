# System Architecture — Hybrid Semantic Search System v2

## Executive Summary

A production-ready hybrid retrieval system combining:

- **BM25** – lexical/keyword search
- **Bi-Encoder + Adaptive FAISS** – semantic/dense search (Flat / IVFFlat / IVFPQ)
- **RRF Fusion** – rank combination
- **Cross-Encoder** – precision re-ranking
- **IR Evaluation Suite** – NDCG@10, MRR, Recall@10, MAP

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                        USER QUERY                        │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│              HybridSearchEngine (Orchestrator)           │
└────────────────────────┬─────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌──────────────────┐       ┌────────────────────────────┐
│  BM25Retriever   │       │   SemanticRetriever        │
│  (Lexical)       │       │   (Bi-Encoder + FAISS)     │
│                  │       │                            │
│  camelCase-aware │       │  Auto index selection:     │
│  tokeniser       │       │  • < 10K  → IndexFlatIP    │
│  BM25Okapi       │       │  • < 500K → IndexIVFFlat   │
│                  │       │  • ≥ 500K → IndexIVFPQ     │
│  Top 50 docs     │       │  Top 50 docs               │
└────────┬─────────┘       └────────────┬───────────────┘
         │                              │
         └──────────────┬───────────────┘
                        ▼
┌──────────────────────────────────────────────────────────┐
│                HybridFusion (RRF)                        │
│   RRF(d) = Σ 1/(k + rank(d))   k=60                    │
│   Output: fused top-50 candidates                       │
└────────────────────────┬─────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│            CrossEncoderReranker                          │
│   model: ms-marco-MiniLM-L-6-v2                         │
│   Input: (query, document) pairs                        │
│   Output: final top-K results                           │
└────────────────────────┬─────────────────────────────────┘
                         ▼
                   Ranked Results
```

---

## New Components (v2)

### dataset_loader.py — StaQC HuggingFace Integration

```
HuggingFace koutch/staqc
        │
        ▼
StaQCDatasetLoader.load()
        │  - streams python or sql split
        │  - caps at max_records
        │  - normalises columns → id, title, body, tags, category, difficulty
        │  - caches as Parquet (instant on re-run)
        ▼
DataProcessor.preprocess_data()
        │  - also accepts legacy CSV paths
        ▼
corpus: List[str], metadata: List[Dict]
```

**Column mapping**

| StaQC raw      | Canonical                        |
| -------------- | -------------------------------- |
| `question_id`  | `id`                             |
| `question`     | `title`                          |
| `code_snippet` | `body`                           |
| `tags`         | `tags`                           |
| derived        | `category` (from tags heuristic) |
| derived        | `difficulty` (from body length)  |

---

### semantic_retriever.py — Adaptive FAISS

```
corpus size N
      │
      ├── N < 10,000  →  IndexFlatIP         (exact inner product)
      │
      ├── N < 500,000 →  IndexIVFFlat        (approximate)
      │                    nlist = min(sqrt(N), N//39)
      │                    nprobe = max(1, nlist * 0.05)
      │
      └── N ≥ 500,000 →  IndexIVFPQ          (compressed + approximate)
                           nlist = auto
                           m (sub-quantisers) = largest divisor of dim ≤ 64
```

**IVF training flow**

```python
quantiser = IndexFlatIP(dim)
index = IndexIVFFlat(quantiser, dim, nlist, METRIC_INNER_PRODUCT)
index.train(embeddings)   # k-means clustering
index.add(embeddings)
index.nprobe = nprobe
```

**Runtime tuning**

```python
retriever.set_nprobe(32)  # more cells → higher recall, slower
```

---

### evaluation.py — IR Metrics Suite

#### Metric functions (stateless)

| Function            | Formula                                 |
| ------------------- | --------------------------------------- |
| `ndcg_at_k`         | DCG@k / IDCG@k                          |
| `reciprocal_rank`   | 1 / rank(first relevant)                |
| `recall_at_k`       | \|retrieved ∩ relevant\| / \|relevant\| |
| `precision_at_k`    | \|retrieved ∩ relevant\| / k            |
| `average_precision` | area under P-R curve                    |

#### EvaluationSuite — orchestrates a full benchmark

```
EvaluationSuite.run(queries, ground_truth, search_types)
      │
      ├── for each query × search_type:
      │     engine.search(query, top_k=k, search_type=…)
      │     compute QueryMetrics (ndcg, mrr, recall, precision, ap)
      │
      └── aggregate → AggregateMetrics (mean + std per metric)
```

#### Ground truth options

```python
# Option A: keyword matching (no labels needed)
gtb = GroundTruthBuilder(metadata)
gt  = gtb.build_from_keywords(queries, min_relevance=2)

# Option B: use engine top-1 as pseudo-relevant
gt  = gtb.build_from_engine_top1(engine, queries)

# Option C: supply your own labels
gt  = {"how to sort list": {"doc_42", "doc_77"}, …}
```

#### When to run evaluation

```
┌──────────────────────────────────────────────┐
│  OFFLINE (developer)                         │
│  python evaluate.py → outputs/eval.json      │
│                                              │
│  Trigger: new model, corpus change,          │
│           parameter tuning                   │
└──────────────────────────────────────────────┘
         │  static JSON
         ▼
┌──────────────────────────────────────────────┐
│  API: GET /api/v1/admin/metrics              │
│  Returns pre-computed JSON — zero latency    │
│  Never recomputed per request                │
└──────────────────────────────────────────────┘
```

---

## Data Flow

### Offline phase (index building)

```
StaQC HuggingFace → DataProcessor → corpus + metadata
                                          │
                          ┌───────────────┤
                          ▼               ▼
                    BM25 index      FAISS index
                   (bm25_index.pkl) (faiss_index.bin + embeddings.npy)
```

### Online phase (query)

```
Query → BM25 (10-20ms) ──────┐
      → FAISS IVF (15-30ms) ─┤→ RRF (1ms) → CrossEncoder (100-200ms) → Results
```

### Evaluation phase (offline)

```
evaluate.py → EvaluationSuite.run() → per_query metrics
                                    → aggregate (NDCG, MRR, Recall)
                                    → outputs/evaluation_report.json
```

---

## Scalability

| Corpus size | FAISS type   | Indexing time | Query latency |
| ----------- | ------------ | ------------- | ------------- |
| 2 K (old)   | IndexFlatIP  | ~30 s         | ~5 ms         |
| 20 K        | IndexFlatIP  | ~3 min        | ~15 ms        |
| 100 K       | IndexIVFFlat | ~8 min        | ~25 ms        |
| 500 K       | IndexIVFFlat | ~35 min       | ~40 ms        |
| 1 M+        | IndexIVFPQ   | ~60 min       | ~30 ms        |

---

## File Change Summary

| File                            | Change type | Reason                          |
| ------------------------------- | ----------- | ------------------------------- |
| `src/dataset_loader.py`         | **NEW**     | StaQC + DataProcessor wrapper   |
| `src/evaluation.py`             | **NEW**     | NDCG/MRR/Recall/EvaluationSuite |
| `src/semantic_retriever.py`     | **UPDATED** | Adaptive FAISS (Flat/IVF/IVFPQ) |
| `src/search_engine.py`          | **UPDATED** | Wire in StaQC + IVF params      |
| `src/bm25_retriever.py`         | **UPDATED** | camelCase-aware tokeniser       |
| `src/hybrid_fusion.py`          | **UPDATED** | Minor cleanup, same interface   |
| `src/cross_encoder_reranker.py` | **UPDATED** | Larger default batch size       |
| `src/utils.py`                  | **UPDATED** | Backwards-compatible            |
| `src/__init__.py`               | **UPDATED** | Export new classes              |
| `src/api.py`                    | **UPDATED** | `/admin/metrics` endpoint       |
| `main.py`                       | **UPDATED** | Demo all 3 features             |
| `evaluate.py`                   | **NEW**     | Standalone evaluation runner    |
| `requirements.txt`              | **UPDATED** | `datasets`, `pyarrow` added     |
| `configs/model_config.yaml`     | **UPDATED** | IVF settings                    |
| `configs/search_config.yaml`    | **UPDATED** | Dataset + eval config           |
| `tests/test_hybrid_search.py`   | **UPDATED** | Tests for all 3 features        |

---

## References

- BM25: Robertson & Zaragoza (2009)
- RRF: Cormack et al., SIGIR 2009
- Sentence-BERT: Reimers & Gurevych, EMNLP 2019
- FAISS: Johnson et al., IEEE TPAMI 2019
- StaQC: Yao et al., WWW 2018
