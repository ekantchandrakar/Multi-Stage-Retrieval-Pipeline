# 🚀 Hybrid Semantic Search System

A production-grade hybrid retrieval system combining lexical (BM25) and semantic (Transformer) search with advanced re-ranking capabilities.

## 🎯 Project Overview

This system implements a state-of-the-art hybrid search architecture that:
- **Combines** lexical (BM25) and semantic (Bi-Encoder) retrieval strategies
- **Fuses** rankings using Reciprocal Rank Fusion (RRF)
- **Re-ranks** results using Cross-Encoder for precision optimization
- **Demonstrates** real-world retrieval trade-offs and performance comparisons

## 🏗️ System Architecture

```
                Query
                   │
         ┌─────────┴─────────┐
         │                   │
      BM25               Bi-Encoder
   (Lexical)            (Semantic)
         │                   │
         └─────────┬─────────┘
                   │
                RRF Fusion
                   │
            Top 50 Candidates
                   │
         Cross-Encoder Rerank
                   │
              Final Top 5
```

## 📁 Project Structure

```
hybrid_search_system/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
├── .gitignore                        # Git ignore rules
│
├── configs/                          # Configuration files
│   ├── model_config.yaml            # Model configurations
│   └── search_config.yaml           # Search parameters
│
├── data/                             # Data directory
│   └── semantic_search_dataset_2000.csv
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── data_processor.py            # Data loading and preprocessing
│   ├── bm25_retriever.py            # BM25 lexical search
│   ├── semantic_retriever.py        # Bi-Encoder semantic search
│   ├── hybrid_fusion.py             # RRF fusion implementation
│   ├── cross_encoder_reranker.py    # Cross-Encoder re-ranking
│   ├── search_engine.py             # Main search engine orchestrator
│   └── utils.py                     # Utility functions
│
├── models/                           # Saved models and indices
│   ├── bm25_index.pkl
│   ├── faiss_index.bin
│   └── embeddings.npy
│
├── outputs/                          # Output results
│   ├── evaluation_results.json
│   └── example_queries.json
│
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_bm25.py
│   ├── test_semantic.py
│   └── test_hybrid.py
│
└── notebooks/                        # Jupyter notebooks
    ├── 01_data_exploration.ipynb
    ├── 02_retrieval_comparison.ipynb
    └── 03_evaluation_analysis.ipynb
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
cd hybrid_search_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Build the Search System

```python
from src.search_engine import HybridSearchEngine

# Initialize the search engine
engine = HybridSearchEngine(
    data_path='data/semantic_search_dataset_2000.csv',
    model_dir='models/'
)

# Build indices (one-time setup)
engine.build_indices()
```

### 2. Perform Search

```python
# Search with hybrid approach
results = engine.search(
    query="reverse singly linked list",
    top_k=5,
    search_type='hybrid'  # Options: 'bm25', 'semantic', 'hybrid'
)

# Display results
for i, result in enumerate(results, 1):
    print(f"{i}. {result['title']}")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Category: {result['category']} | Difficulty: {result['difficulty']}")
    print()
```

### 3. Compare Retrieval Methods

```python
# Compare all three approaches
comparison = engine.compare_retrievers(
    query="optimize SQL join performance"
)

print("BM25 Results:", comparison['bm25'][:3])
print("Semantic Results:", comparison['semantic'][:3])
print("Hybrid Results:", comparison['hybrid'][:3])
```

## 📊 Dataset

**File**: `semantic_search_dataset_2000.csv`

**Fields**:
- `id`: Unique identifier
- `category`: Topic category (e.g., Data Structures, Algorithms, Databases)
- `difficulty`: Beginner, Intermediate, Advanced
- `title`: Question title
- `body`: Detailed question description
- `tags`: Comma-separated tags

**Statistics**:
- Total documents: 2,000
- Categories: 6 (Data Structures, Algorithms, Databases, DevOps & Cloud, Backend Systems, etc.)
- Difficulty levels: 3 (Beginner, Intermediate, Advanced)

## 🔬 Components

### 1. BM25 Retriever (Lexical Search)
- **Algorithm**: BM25 (Best Matching 25)
- **Strengths**: Exact term matching, keyword search
- **Use Case**: Finding documents with specific technical terms

### 2. Semantic Retriever (Bi-Encoder)
- **Model**: `all-MiniLM-L6-v2`
- **Index**: FAISS (Facebook AI Similarity Search)
- **Strengths**: Understanding semantic similarity, handling synonyms
- **Use Case**: Conceptual searches, paraphrased queries

### 3. Reciprocal Rank Fusion (RRF)
- **Formula**: `RRF(d) = Σ 1/(k + rank(d))` where k=60
- **Purpose**: Merge rankings from multiple retrievers
- **Benefit**: Combines strengths of both lexical and semantic approaches

### 4. Cross-Encoder Re-ranker
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Purpose**: Fine-grained relevance scoring
- **Benefit**: Highest precision for final top-k results

## 🎯 Evaluation

### Test Queries

The system includes hard test queries to demonstrate retrieval quality:

1. **"reverse singly linked list"** - Tests exact term matching
2. **"optimize SQL join performance"** - Tests database expertise
3. **"deploy microservice on kubernetes"** - Tests DevOps knowledge
4. **"reduce API latency in production"** - Tests backend systems
5. **"implement binary search tree"** - Tests data structures
6. **"explain ACID properties in databases"** - Tests conceptual understanding

### Metrics

```python
from src.utils import evaluate_search_quality

metrics = evaluate_search_quality(
    engine=engine,
    test_queries=['query1', 'query2', ...],
    ground_truth=None  # Optional if available
)
```

## 🔧 Configuration

### Model Configuration (`configs/model_config.yaml`)

```yaml
bi_encoder:
  model_name: "all-MiniLM-L6-v2"
  device: "cpu"
  batch_size: 32

cross_encoder:
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  device: "cpu"

faiss:
  index_type: "IndexFlatIP"  # Inner product for normalized vectors
  normalize: true
```

### Search Configuration (`configs/search_config.yaml`)

```yaml
bm25:
  top_k: 50
  k1: 1.5
  b: 0.75

semantic:
  top_k: 50
  normalize: true

rrf:
  k: 60
  top_k: 50

cross_encoder:
  top_k: 5
  batch_size: 16
```

## 📈 Performance Considerations

### Memory Usage
- **BM25 Index**: ~50-100 MB for 2,000 documents
- **FAISS Index**: ~3-5 MB (384-dim embeddings)
- **Embeddings**: ~3 MB (2000 × 384 × 4 bytes)

### Speed Benchmarks (approximate)
- **BM25 Search**: ~10-20 ms
- **Semantic Search**: ~5-15 ms (FAISS)
- **RRF Fusion**: ~1-2 ms
- **Cross-Encoder Re-ranking** (50 docs): ~100-200 ms
- **Total Hybrid Search**: ~120-240 ms

### Scalability
- Handles up to 100K documents efficiently
- For larger datasets, consider:
  - Approximate nearest neighbor search (FAISS IVF)
  - Distributed BM25 (Elasticsearch)
  - GPU acceleration for embeddings

## 🧪 Testing

Run unit tests:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_hybrid.py -v

# Run with coverage
pytest --cov=src tests/
```

## 📚 API Reference

### HybridSearchEngine

```python
class HybridSearchEngine:
    def __init__(self, data_path: str, model_dir: str)
    def build_indices(self) -> None
    def search(self, query: str, top_k: int = 5, search_type: str = 'hybrid') -> List[Dict]
    def compare_retrievers(self, query: str, top_k: int = 5) -> Dict
    def save_models(self, path: str) -> None
    def load_models(self, path: str) -> None
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License

## 🙏 Acknowledgments

- **sentence-transformers**: Hugging Face team
- **rank-bm25**: dorianbrown
- **FAISS**: Facebook Research
- **Dataset**: Custom generated technical Q&A dataset

## 📞 Contact

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ for production-grade semantic search**
