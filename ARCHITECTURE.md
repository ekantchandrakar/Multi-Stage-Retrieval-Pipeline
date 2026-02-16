# System Architecture Documentation

## 📋 Project Summary: Hybrid Semantic Search System

## Executive Summary

A **production-ready hybrid retrieval system** that combines:

- **BM25** (lexical/keyword search)
- **Bi-Encoder** (semantic/dense embeddings)
- **RRF Fusion** (rank combination)
- **Cross-Encoder** (precision re-ranking)

Built for the `semantic_search_dataset_2000.csv` dataset containing 2,000 technical Q&A documents.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SEARCH ENGINE ORCHESTRATOR                   │
│                    (HybridSearchEngine)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│   BM25 RETRIEVER        │   │  SEMANTIC RETRIEVER     │
│   (Lexical Search)      │   │  (Bi-Encoder + FAISS)   │
│                         │   │                         │
│  • Tokenization         │   │  • Text Encoding        │
│  • TF-IDF Scoring       │   │  • Vector Embedding     │
│  • BM25 Ranking         │   │  • Similarity Search    │
│                         │   │                         │
│  Output: Top 50 docs    │   │  Output: Top 50 docs    │
└────────┬────────────────┘   └────────┬────────────────┘
         │                             │
         │        ┌────────────────────┘
         │        │
         ▼        ▼
┌─────────────────────────────────────────────────────────────────┐
│               RECIPROCAL RANK FUSION (RRF)                      │
│                                                                  │
│  • Combines rankings from both retrievers                       │
│  • Formula: RRF(d) = Σ 1/(k + rank(d))                         │
│  • k = 60 (standard parameter)                                  │
│                                                                  │
│  Output: Fused Top 50 candidates                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           CROSS-ENCODER RE-RANKER                                │
│                                                                  │
│  • Processes (query, document) pairs jointly                    │
│  • Fine-grained relevance scoring                               │
│  • Model: cross-encoder/ms-marco-MiniLM-L-6-v2                 │
│                                                                  │
│  Output: Final Top K results                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RANKED RESULTS                              │
│              (Documents with metadata + scores)                  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Processor (`data_processor.py`)

**Purpose**: Load, clean, and preprocess the document corpus.

**Key Functions**:

- `load_data()`: Load CSV dataset into pandas DataFrame
- `preprocess_data()`: Combine title + body, clean text
- `get_document_by_index()`: Retrieve document metadata
- `get_statistics()`: Dataset statistics

**Data Flow**:

```
CSV File → DataFrame → Text Cleaning → Corpus + Metadata
```

**Output**:

- `corpus`: List of cleaned document texts
- `metadata`: List of document metadata dictionaries

---

### 2. BM25 Retriever (`bm25_retriever.py`)

**Purpose**: Lexical (keyword-based) retrieval using BM25 algorithm.

**Algorithm**: BM25 (Best Matching 25)

```
score(D,Q) = Σ IDF(qi) × [f(qi,D) × (k1 + 1)] / [f(qi,D) + k1 × (1 - b + b × |D|/avgdl)]
```

Where:

- `qi`: Query terms
- `f(qi,D)`: Term frequency in document D
- `k1`: Term frequency saturation (default: 1.5)
- `b`: Length normalization (default: 0.75)
- `|D|`: Document length
- `avgdl`: Average document length

**Implementation**:

```python
# Tokenization
tokenized_corpus = [tokenize(doc) for doc in corpus]

# Build BM25 index
bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

# Search
scores = bm25.get_scores(tokenize(query))
```

**Strengths**:

- ✅ Fast (~10-20ms)
- ✅ Exact term matching
- ✅ Good for technical keywords

**Limitations**:

- ❌ No semantic understanding
- ❌ Vocabulary mismatch problem
- ❌ Sensitive to exact wording

---

### 3. Semantic Retriever (`semantic_retriever.py`)

**Purpose**: Semantic retrieval using dense embeddings and vector similarity.

**Model**: `sentence-transformers/all-MiniLM-L6-v2`

- **Type**: Bi-Encoder
- **Embedding Dimension**: 384
- **Training**: Trained on 1B+ sentence pairs

**Process**:

1. **Encoding**:

   ```python
   embeddings = model.encode(corpus)  # Shape: (N, 384)
   ```

2. **Indexing** (FAISS):

   ```python
   index = faiss.IndexFlatIP(384)  # Inner Product
   index.add(normalize(embeddings))
   ```

3. **Search**:
   ```python
   query_embedding = model.encode([query])
   scores, indices = index.search(query_embedding, k=50)
   ```

**FAISS Index Types**:

- `IndexFlatIP`: Exact inner product search (used here)
- `IndexFlatL2`: Exact L2 distance search
- `IndexIVFFlat`: Approximate search (for large datasets)

**Strengths**:

- ✅ Semantic understanding
- ✅ Handles paraphrasing
- ✅ Cross-lingual capability (with multilingual models)

**Limitations**:

- ❌ May miss exact term matches
- ❌ Higher memory usage
- ❌ Model-dependent quality

---

### 4. Hybrid Fusion (`hybrid_fusion.py`)

**Purpose**: Combine rankings from multiple retrievers.

**Algorithm**: Reciprocal Rank Fusion (RRF)

**Formula**:

```
RRF(d) = Σ [1 / (k + rank_r(d))]
         r∈retrievers

Where:
- d: Document
- r: Retriever
- rank_r(d): Rank of document d in retriever r
- k: Constant (typically 60)
```

**Example**:

```
Document A:
  BM25 rank: 1    → RRF contribution: 1/(60+1) = 0.0164
  Semantic rank: 3 → RRF contribution: 1/(60+3) = 0.0159
  Total RRF: 0.0323

Document B:
  BM25 rank: 10   → RRF contribution: 1/(60+10) = 0.0143
  Semantic rank: 1 → RRF contribution: 1/(60+1) = 0.0164
  Total RRF: 0.0307

Winner: Document A (higher RRF score)
```

**Why RRF?**:

- Doesn't require score normalization
- Robust to score distribution differences
- Simple and effective
- Well-studied in IR literature

**Alternatives**:

- Linear combination (requires score normalization)
- Borda count (rank-based voting)
- CombSUM/CombMNZ

---

### 5. Cross-Encoder Re-ranker (`cross_encoder_reranker.py`)

**Purpose**: Fine-grained relevance scoring for final ranking.

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

- **Type**: Cross-Encoder
- **Training**: MS MARCO passage ranking dataset
- **Input**: (query, document) pairs
- **Output**: Relevance score

**Bi-Encoder vs Cross-Encoder**:

```
Bi-Encoder (Semantic Retriever):
  encode(query) → q_vec
  encode(doc)   → d_vec
  similarity = dot(q_vec, d_vec)

  ✅ Fast (can pre-compute document vectors)
  ❌ Lower precision (independent encoding)

Cross-Encoder (Re-ranker):
  encode([query, doc]) → relevance_score

  ✅ Higher precision (joint encoding)
  ❌ Slower (must encode each pair)
```

**Usage Pattern**:

1. Bi-Encoder: Retrieve top 50 candidates (fast, broad)
2. Cross-Encoder: Re-rank to top 5 (slow, precise)

**Implementation**:

```python
# Create (query, doc) pairs
pairs = [[query, doc] for doc in candidate_docs]

# Predict relevance scores
scores = cross_encoder.predict(pairs)

# Sort and return top-k
results = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)[:k]
```

---

### 6. Search Engine Orchestrator (`search_engine.py`)

**Purpose**: Coordinate all components and manage search pipeline.

**Main Methods**:

1. **`build_indices()`**:

   ```python
   # Load data
   corpus, metadata = data_processor.preprocess_data()

   # Build BM25
   bm25_retriever.build_index(corpus)

   # Build Semantic
   semantic_retriever.build_index(corpus)

   # Save to disk
   save_indices()
   ```

2. **`search(query, top_k, search_type)`**:

   ```python
   if search_type == 'hybrid':
       # Step 1: Retrieve candidates
       bm25_results = bm25_retriever.search(query, k=50)
       semantic_results = semantic_retriever.search(query, k=50)

       # Step 2: Fuse
       fused = fusion.fuse(bm25_results, semantic_results, k=50)

       # Step 3: Re-rank
       final = reranker.rerank(query, fused_docs, top_k=5)

       return final
   ```

---

## Data Flow

### Offline Phase (Index Building)

```
1. Load Dataset
   ↓
2. Preprocess Text (clean, combine fields)
   ↓
3. Build BM25 Index (tokenize, compute statistics)
   ↓
4. Build Semantic Index (encode, create FAISS index)
   ↓
5. Save Indices to Disk
```

**Time Complexity**:

- BM25 indexing: O(N × L) where N = docs, L = avg length
- Semantic indexing: O(N × L × D) where D = model complexity
- Total: ~3-5 minutes for 2000 documents

### Online Phase (Query Processing)

```
1. Receive Query
   ↓
2. Parallel Retrieval:
   ├─ BM25: O(V) where V = vocabulary size
   └─ Semantic: O(log N) with FAISS indexing
   ↓
3. RRF Fusion: O(K) where K = candidates
   ↓
4. Cross-Encoder Re-rank: O(K × M) where M = model complexity
   ↓
5. Return Top Results
```

**Time Complexity**:

- BM25 search: ~10-20ms
- Semantic search: ~5-15ms
- RRF fusion: ~1-2ms
- Cross-encoder (50 docs): ~100-200ms
- **Total: ~120-240ms**

---

## Storage

### Memory Requirements

**Runtime Memory**:

```
Component               Size (approx)
────────────────────────────────────
BM25 Index             50-100 MB
FAISS Index            3-5 MB
Embeddings (2K docs)   3 MB
Model Weights          ~400 MB
Working Memory         ~500 MB
────────────────────────────────────
Total                  ~1 GB
```

**Disk Storage**:

```
Component               Size
────────────────────────────────────
Dataset                1.1 MB
BM25 Index (pkl)       ~50 MB
FAISS Index            ~3 MB
Embeddings (npy)       ~3 MB
Models (cached)        ~400 MB
────────────────────────────────────
Total                  ~460 MB
```

---

## Scalability

### Current System (2,000 docs)

- ✅ In-memory processing
- ✅ Exact search with FAISS
- ✅ Sub-second latency

### Medium Scale (10K - 100K docs)

- Use FAISS IVF index for approximate search
- Implement batching for re-ranking
- Consider distributed BM25 (Elasticsearch)

### Large Scale (1M+ docs)

- Distributed FAISS with GPU
- Elasticsearch/Solr for BM25
- Two-stage retrieval (coarse → fine)
- Caching layer for popular queries

---

## Performance Optimization

### 1. Index Optimization

```python
# Use approximate search for large datasets
index = faiss.IndexIVFFlat(quantizer, dim, nlist=100)
index.nprobe = 10  # Trade accuracy for speed
```

### 2. Batch Processing

```python
# Process multiple queries in parallel
results = engine.batch_search(queries, batch_size=32)
```

### 3. GPU Acceleration

```python
# Use GPU for encoding
engine = HybridSearchEngine(device='cuda')
```

### 4. Caching

```python
# Cache frequent queries
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_search(query):
    return engine.search(query)
```

---

## Extension Points

### 1. Add New Retrievers

```python
class CustomRetriever:
    def search(self, query, top_k):
        # Your retrieval logic
        return [(doc_id, score), ...]

# Add to fusion
fusion.fuse(bm25_results, semantic_results, custom_results)
```

### 2. Custom Fusion Strategy

```python
class CustomFusion(HybridFusion):
    def fuse(self, *ranked_lists):
        # Your fusion algorithm
        return fused_results
```

### 3. Query Expansion

```python
def expand_query(query):
    # Add synonyms, related terms
    expanded = query + " " + get_synonyms(query)
    return expanded
```

### 4. Filtering

```python
# Filter by category
results = engine.search(query)
filtered = [r for r in results if r['category'] == 'Algorithms']
```

---

## Testing Strategy

### Unit Tests

- Individual component testing
- Mock data for fast execution
- Coverage: >80%

### Integration Tests

- End-to-end search pipeline
- Real data with small subset
- Performance benchmarks

### Evaluation

- Compare against baseline
- Measure: Precision@K, Recall@K, NDCG
- Human evaluation on sample queries

---

## Deployment Considerations

### API Wrapper

```python
from fastapi import FastAPI

app = FastAPI()
engine = HybridSearchEngine(...)
engine.load_indices()

@app.get("/search")
def search(q: str, k: int = 5):
    return engine.search(q, top_k=k)
```

### Docker Container

```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "api.py"]
```

### Monitoring

- Log query latency
- Track cache hit rate
- Monitor memory usage
- Alert on errors

---

## Future Enhancements

1. **Query Understanding**
   - Intent classification
   - Named entity recognition
   - Query reformulation

2. **Learning to Rank**
   - Train custom re-ranker
   - Use click-through data
   - A/B testing framework

3. **Multilingual Support**
   - Use multilingual models
   - Language detection
   - Cross-lingual retrieval

4. **Real-time Updates**
   - Incremental indexing
   - Document versioning
   - Stale result detection

---

## References

- BM25: Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009)
- RRF: Cormack et al., "Reciprocal Rank Fusion" (SIGIR 2009)
- Sentence-BERT: Reimers & Gurevych, "Sentence-BERT" (EMNLP 2019)
- MS MARCO: Nguyen et al., "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset" (2016)
