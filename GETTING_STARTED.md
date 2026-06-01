# Getting Started — Hybrid Search System v2

## Prerequisites

- Python 3.8+
- pip
- 4 GB RAM minimum (8 GB recommended for 100 K corpus)
- 3 GB free disk (models + FAISS index)
- Internet connection (first run downloads StaQC + model weights)

---

## 1. Install

```bash
# Clone / extract the project
cd hybrid_search_system

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Run the Demo

```bash
python main.py
```

What happens on first run:

1. Downloads `koutch/staqc` (python split, capped at 20 K by default)
2. Downloads sentence-transformer model weights (~90 MB)
3. Builds BM25 index (~30 s for 20 K docs)
4. Builds FAISS index — auto-selects IVFFlat for 20 K docs (~2 min)
5. Runs sample searches and prints comparison
6. Runs IR evaluation (NDCG@10, MRR, Recall@10)

Subsequent runs: indices are loaded from disk in seconds.

---

## 3. Run the Evaluation Benchmark

```bash
# Full benchmark (15 queries, NDCG@10 / MRR / Recall@10)
python evaluate.py

# Quick test with 5 queries
python evaluate.py --queries 5

# Benchmark only BM25
python evaluate.py --search-type bm25

# Use SQL dataset instead
python evaluate.py --language sql

# Larger corpus
python evaluate.py --max-records 50000
```

Output files:

- `outputs/evaluation_report.json` — full per-query + aggregate results
- `outputs/evaluation_summary.json` — compact summary for the API

---

## 4. Start the API Server

```bash
uvicorn src.api:app --reload --port 8000
```

API docs: http://localhost:8000/docs

Key endpoints:

```bash
# Hybrid search
curl -X POST http://localhost:8000/api/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{"query": "reverse linked list Python", "top_k": 5}'

# Pre-computed evaluation metrics (run evaluate.py first)
curl http://localhost:8000/api/v1/admin/metrics

# Corpus statistics + FAISS index info
curl http://localhost:8000/api/v1/statistics
```

---

## 5. Use in Your Own Code

### Basic search

```python
from src.search_engine import HybridSearchEngine

engine = HybridSearchEngine(
    data_path="staqc",
    language="python",
    max_records=20_000,
    model_dir="models",
)
engine.build_indices()

results = engine.search("implement binary search tree", top_k=5)
for r in results:
    print(f"[{r['rank']}] {r['title']}  (score={r['score']:.4f})")
```

### Control FAISS index type

```python
engine = HybridSearchEngine(
    data_path="staqc",
    max_records=100_000,
    index_type="ivf",   # force IVFFlat
    nlist=512,
    nprobe=32,
)
```

### Evaluate search quality

```python
from src.evaluation import EvaluationSuite, GroundTruthBuilder, IRMetrics

# Single query
m = IRMetrics.evaluate_query(
    retrieved=[r["id"] for r in engine.search("sort list python", top_k=10)],
    relevant={"doc_42", "doc_77"},
    k=10,
)
print(m)  # {'ndcg@10': 0.86, 'mrr': 1.0, 'recall@10': 0.5, ...}

# Full benchmark
gtb = GroundTruthBuilder(engine.metadata)
gt  = gtb.build_from_keywords(my_queries, min_relevance=2)
suite = EvaluationSuite(engine, k=10)
report = suite.run(my_queries, gt, search_types=["bm25","semantic","hybrid"])
suite.print_report(report)
suite.save_report(report, "outputs/my_eval.json")
```

---

## 6. Changing Dataset Size

Edit `main.py` or pass arguments to `evaluate.py`:

| max_records | FAISS type (auto) | Build time | RAM   |
| ----------- | ----------------- | ---------- | ----- |
| 2 000       | IndexFlatIP       | ~1 min     | ~1 GB |
| 20 000      | IndexIVFFlat      | ~3 min     | ~2 GB |
| 100 000     | IndexIVFFlat      | ~15 min    | ~4 GB |

To use full StaQC (148 K python questions):

```python
engine = HybridSearchEngine(data_path="staqc", max_records=0)  # 0 = no cap
```

---

## 7. Run Tests

```bash
# All tests
pytest tests/ -v

# Only new-feature tests
pytest tests/ -v -k "IVF or NDCG or MRR or Recall or StaQC"

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Troubleshooting

**HuggingFace download fails**

```bash
pip install -U datasets huggingface_hub
# Or set HF_ENDPOINT if behind a proxy
export HF_ENDPOINT=https://hf-mirror.com
```

**Out of memory during FAISS build**

```python
# Reduce batch size
engine.semantic_retriever.build_index(corpus, batch_size=32)
```

**FAISS index accuracy too low (IVF)**

```python
engine.semantic_retriever.set_nprobe(64)  # default ~nlist*0.05
```

**Force rebuild after corpus change**

```python
engine.build_indices(force_rebuild=True)
```
