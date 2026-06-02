# Hybrid Semantic Search System

A production-grade search system that combines keyword matching, semantic understanding, and neural re-ranking into a single pipeline. Built on top of real Stack Overflow Python Q&A data from the [StaQC dataset](https://huggingface.co/datasets/koutch/staqc).

---

## What it does

Most search systems pick one approach — either exact keyword matching or semantic similarity. This one uses both, fuses the results, then re-ranks with a neural model. In practice that means:

- A query like **"reverse a linked list in Python"** through BM25 alone surfaces docs that literally contain those words ("using dictionaries and linked list python") — close but not quite right
- The semantic retriever finds **"What is the pythonic way to reverse a defaultdict(list)?"** — semantically correct but keyword-mismatched
- The hybrid pipeline combines both and surfaces the right answers at the top

That difference is what this project demonstrates.

---

## Architecture

```
User query
    │
    ├─► BM25Retriever        lexical match, ~10–20 ms
    ├─► SemanticRetriever    FAISS dense vectors, ~15–30 ms
    │         └─ auto: IndexFlatIP (<10K) / IVFFlat (<500K) / IVFPQ (500K+)
    │
    ├─► HybridFusion (RRF)   combines both rankings
    └─► CrossEncoderReranker neural re-ranking, final top-5
```

The index builds once and loads from disk on every subsequent run — typically under 10 seconds.

---

## Results (tested locally on 3,700 docs)

These numbers came from a run on 3,700 records of `mca_python` StaQC data on a CPU-only machine. The dataset was rate-limited mid-download at 3,700/4,000 rows, which is why it's not a round number.

| Method     | NDCG@10  | MRR      | Precision@10 | Avg latency |
| ---------- | -------- | -------- | ------------ | ----------- |
| BM25       | 1.00     | 1.00     | 1.00         | 57 ms       |
| Semantic   | 0.96     | 1.00     | 0.95         | 22 ms       |
| **Hybrid** | **0.99** | **1.00** | **0.99**     | 6,114 ms    |

A few honest notes about these numbers:

**The metrics look suspiciously good.** NDCG and MRR near 1.0 is because ground-truth was built using keyword matching (`min_relevance=2`) — roughly 2,600+ documents matched each query as "relevant", so it's almost impossible to retrieve a non-relevant result. These numbers reflect internal consistency, not real-world relevance quality. For production use you'd want human-labelled relevance judgements.

**Recall@10 is intentionally low (~0.014).** With 2,600 relevant documents per query and only 10 retrieved, you're covering about 1.4% of what exists. That's correct math, not a bug.

**Hybrid latency is high at 6 seconds.** This is the cross-encoder running on CPU re-ranking 50 candidate pairs. On a machine with a GPU that drops to ~200–400 ms. For local/demo use it's fine; for production you'd run inference on GPU or reduce the candidate pool.

---

## Dataset

Uses `koutch/staqc` from HuggingFace, loaded via the [Datasets Server REST API](https://huggingface.co/docs/datasets-server) — no `datasets` library required. The dataset contains Stack Overflow questions with accepted code answers.

| Config       | Content                      | Size      |
| ------------ | ---------------------------- | --------- |
| `mca_python` | Multi-code-answer Python Q&A | ~40K rows |
| `man_python` | Manually curated Python Q&A  | ~2K rows  |
| `man_sql`    | Manually curated SQL Q&A     | ~2K rows  |

For local development, the project defaults to 4,000 rows of `mca_python`. For production, raise `MAX_RECORDS` and let it run — the full 40K takes about 15 minutes to encode on CPU.

Data is cached locally as CSV after the first download so subsequent runs don't re-fetch anything.

---

## Getting started

```bash
# clone and install
cd hybrid_search_system
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# build indices (downloads ~4K rows, encodes with sentence-transformers)
python main.py
```

First run takes 5–10 minutes depending on your connection and CPU. You'll see each step print as it completes:

```
✓ Step 1/4 complete — 3,700 documents loaded
✓ Step 2/4  BM25 index built
✓ Step 3/4  FAISS index built
✓ Step 4/4  Indices saved → models
✅ All indices built and saved successfully!
```

Second run loads everything from disk and reaches the search results in about 10 seconds.

---

## Run the evaluation

```bash
python evaluate.py
```

This builds pseudo-relevance ground truth, runs all three search types across 15 queries, and prints NDCG/MRR/Recall side by side. The full report saves to `outputs/evaluation_report.json`.

To test with fewer queries or a single search type:

```bash
python evaluate.py --queries 5
python evaluate.py --search-type hybrid
python evaluate.py --min-relevance 4   # stricter ground truth
```

---

## Start the API

```bash
uvicorn src.api:app --reload --port 8000
```

Swagger docs at `http://localhost:8000/docs`. Key endpoints:

|      | Path                      | What it does                                    |
| ---- | ------------------------- | ----------------------------------------------- |
| POST | `/api/v1/search/hybrid`   | Full pipeline search                            |
| POST | `/api/v1/search/lexical`  | BM25 only                                       |
| POST | `/api/v1/search/semantic` | FAISS only                                      |
| POST | `/api/v1/compare`         | All three side by side                          |
| GET  | `/api/v1/statistics`      | Corpus + index info                             |
| GET  | `/api/v1/admin/metrics`   | Pre-computed NDCG/MRR (run `evaluate.py` first) |

---

## Project structure

```
hybrid_search_system/
├── main.py                    entry point — builds indices, runs demo
├── evaluate.py                offline IR evaluation (NDCG / MRR / Recall)
├── requirements.txt
│
├── src/
│   ├── dataset_loader.py      StaQC download (REST API) + DataProcessor
│   ├── bm25_retriever.py      BM25Okapi with camelCase tokeniser
│   ├── semantic_retriever.py  sentence-transformers + adaptive FAISS
│   ├── hybrid_fusion.py       Reciprocal Rank Fusion
│   ├── cross_encoder_reranker.py  ms-marco-MiniLM-L-6-v2 re-ranking
│   ├── search_engine.py       orchestrator (build / search / compare)
│   ├── evaluation.py          NDCG, MRR, Recall, MAP, EvaluationSuite
│   └── api.py                 FastAPI endpoints
│
├── models/                    saved indices (git-ignored)
│   ├── bm25_index.pkl
│   ├── faiss_index.bin
│   ├── embeddings.npy
│   └── data_cache/staqc_*.csv
│
└── outputs/
    ├── evaluation_report.json
    └── batch_results.json
```

---

## Scaling up

The system was tested at 3,700 documents. To run at larger scales:

```python
# in main.py
MAX_RECORDS = 20_000   # ~15 min build on CPU
MAX_RECORDS = 0        # full 40K mca_python corpus
```

FAISS index type switches automatically:

- Under 10K → exact flat index (fast, precise)
- 10K–500K → IVFFlat (approximate, much faster at query time)
- Over 500K → IVFPQ (compressed, for very large sets)

To rebuild after changing the corpus:

```bash
python main.py --force-rebuild
# or delete models/ and re-run
```

---

## Known limitations

- **HuggingFace rate limiting** — the Datasets Server API allows ~3,700–4,000 rows per session unauthenticated. Set `HF_TOKEN` in your environment to get higher limits. The local CSV cache means this only affects the first download.
- **CPU cross-encoder latency** — 6 seconds per query on CPU is too slow for production. A GPU or a smaller candidate pool (`retrieve_k=20` instead of 50) brings it down significantly.
- **Pseudo-relevance ground truth** — the evaluation numbers are only as good as the keyword-matching ground truth. Real relevance labels would give more meaningful NDCG/Recall scores.

---

## Tech stack

| Component        | Library                                                     |
| ---------------- | ----------------------------------------------------------- |
| Keyword search   | rank-bm25                                                   |
| Dense embeddings | sentence-transformers (all-MiniLM-L6-v2)                    |
| Vector index     | FAISS (faiss-cpu)                                           |
| Re-ranking       | sentence-transformers CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| API              | FastAPI + uvicorn                                           |
| Dataset          | HuggingFace Datasets Server API                             |
