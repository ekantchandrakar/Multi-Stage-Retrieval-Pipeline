# 🚀 Getting Started with Hybrid Search System

Welcome! This guide will walk you through setting up and using the Hybrid Semantic Search System.

## 📋 Prerequisites Check

Before starting, ensure you have:

- ✅ Python 3.8 or higher (`python --version`)
- ✅ pip package manager (`pip --version`)
- ✅ 4GB RAM minimum
- ✅ 2GB free disk space

## 🎯 5-Minute Quick Start

### Step 1: Navigate to Project Directory

```bash
cd hybrid_search_system
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install packages
pip install -r requirements.txt
```

### Step 3: Run the Demo

```bash
python main.py
```

That's it! The system will:

1. Build search indices (~3-5 minutes first time)
2. Run example searches
3. Display results comparison
4. Generate evaluation report

## 📖 Detailed Setup Guide

### Installation Options

#### Option A: Basic Installation

```bash
pip install -r requirements.txt
```

#### Option B: Development Installation

```bash
pip install -e .  # Installs package in editable mode
pip install -r requirements.txt
```

#### Option C: With Development Tools

```bash
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy jupyter
```

## 🔍 Understanding the System

### The Three Search Modes

1. **BM25 (Lexical)**
   - Best for: Exact keyword matching
   - Example: "SQL JOIN optimization"
   - Speed: Very fast (~10-20ms)

2. **Semantic (Bi-Encoder)**
   - Best for: Conceptual similarity
   - Example: "How to make database faster?"
   - Speed: Very fast (~5-15ms)

3. **Hybrid (RRF + Cross-Encoder)**
   - Best for: Optimal results
   - Combines: BM25 + Semantic + Re-ranking
   - Speed: Medium (~120-240ms)

### Architecture Overview

```
Query → BM25 → ┐
                ├→ RRF Fusion → Cross-Encoder → Top 5 Results
Query → Semantic →┘
```

## 💻 Code Examples

### Example 1: Simple Search

```python
from src.search_engine import HybridSearchEngine

# Initialize
engine = HybridSearchEngine('data/semantic_search_dataset_2000.csv')

# Build indices (first time only)
engine.build_indices()

# Search!
results = engine.search("reverse linked list", top_k=5)

# Display
for i, result in enumerate(results, 1):
    print(f"{i}. {result['title']}")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Category: {result['category']}\n")
```

### Example 2: Compare All Methods

```python
# Compare BM25, Semantic, and Hybrid
comparison = engine.compare_retrievers("optimize SQL queries", top_k=3)

print("BM25 Top Result:")
print(f"  {comparison['bm25'][0]['title']}")

print("\nSemantic Top Result:")
print(f"  {comparison['semantic'][0]['title']}")

print("\nHybrid Top Result:")
print(f"  {comparison['hybrid'][0]['title']}")
```

### Example 3: Batch Evaluation

```python
# Test multiple queries
test_queries = [
    "implement binary search tree",
    "deploy microservice kubernetes",
    "database indexing strategies"
]

results = engine.evaluate_on_queries(
    test_queries,
    save_path='outputs/my_evaluation.json'
)

print(f"Average search time: {results['statistics']['avg_search_time']:.3f}s")
```

### Example 4: Using Specific Search Mode

```python
# BM25 only
bm25_results = engine.search(
    "Python linked list",
    search_type='bm25',
    top_k=5
)

# Semantic only
semantic_results = engine.search(
    "data structure traversal",
    search_type='semantic',
    top_k=5
)

# Hybrid (default)
hybrid_results = engine.search(
    "optimize database performance",
    search_type='hybrid',
    top_k=5
)
```

## 🎓 Interactive Learning

### Jupyter Notebook Tutorial

```bash
# Start Jupyter
jupyter notebook

# Open the analysis notebook
# Navigate to: notebooks/evaluation_analysis.ipynb
```

The notebook includes:

- Step-by-step walkthrough
- Visual comparisons
- Performance analysis
- Interactive examples

## 🧪 Testing Your Installation

### Run Unit Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_hybrid_search.py

# Run with coverage
pytest --cov=src tests/
```

### Quick Validation

```python
# Quick test to ensure everything works
from src.search_engine import HybridSearchEngine

engine = HybridSearchEngine('data/semantic_search_dataset_2000.csv')
print("✅ Engine initialized successfully!")

# Check dataset
print(f"✅ Dataset loaded: {len(engine.data_processor)} documents")
```

## 📊 Understanding the Dataset

The dataset contains 2,000 technical Q&A documents:

```python
from src.data_processor import DataProcessor

processor = DataProcessor('data/semantic_search_dataset_2000.csv')
processor.load_data()
stats = processor.get_statistics()

print(f"Total Documents: {stats['total_documents']}")
print(f"Categories: {list(stats['categories'].keys())}")
print(f"Difficulty Levels: {list(stats['difficulty_levels'].keys())}")
```

## 🎯 Common Use Cases

### Use Case 1: Technical Documentation Search

```python
# Good for: Finding specific code examples
results = engine.search(
    "implement JWT authentication in Node.js",
    search_type='hybrid',
    top_k=5
)
```

### Use Case 2: Conceptual Queries

```python
# Good for: Understanding concepts
results = engine.search(
    "What are ACID properties and why are they important?",
    search_type='semantic',
    top_k=5
)
```

### Use Case 3: Exact Keyword Matching

```python
# Good for: Finding documents with specific terms
results = engine.search(
    "Kubernetes StatefulSet deployment",
    search_type='bm25',
    top_k=5
)
```

## ⚙️ Configuration

### Customize Model Settings

Edit `configs/model_config.yaml`:

```yaml
bi_encoder:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu" # Change to "cuda" for GPU

cross_encoder:
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  device: "cpu"
```

### Customize Search Settings

Edit `configs/search_config.yaml`:

```yaml
rrf_fusion:
  k: 60 # RRF constant

cross_encoder_rerank:
  top_k: 5 # Number of final results
```

### Programmatic Configuration

```python
# Custom BM25 parameters
from src.bm25_retriever import BM25Retriever
bm25 = BM25Retriever(k1=1.2, b=0.75)

# Custom RRF parameter
from src.hybrid_fusion import HybridFusion
fusion = HybridFusion(k=40)
```

## 🐛 Troubleshooting

### Problem: "Index not built" error

**Solution:**

```python
engine.build_indices()  # Build indices first
```

### Problem: Slow first run

**Why:** Downloading models and building indices
**Solution:** This is normal. Subsequent runs are much faster.

### Problem: Out of memory

**Solution:** Reduce batch size

```python
engine.semantic_retriever.build_index(
    corpus,
    batch_size=16  # Reduce from default 32
)
```

### Problem: Import errors

**Solution:** Ensure all dependencies installed

```bash
pip install -r requirements.txt --upgrade
```

## 📈 Performance Optimization

### Use GPU (if available)

```python
engine = HybridSearchEngine(
    data_path='data/semantic_search_dataset_2000.csv',
    device='cuda'
)
```

### Reduce Re-ranking Candidates

```python
# Faster but potentially lower quality
results = engine.search(
    query,
    top_k=5,
    retrieve_k=20  # Default: 50
)
```

### Load Pre-built Indices

```python
# Skip building if indices already exist
engine = HybridSearchEngine('data/semantic_search_dataset_2000.csv')
engine.load_indices()  # Fast!
results = engine.search("your query")
```

## 📚 Next Steps

### Beginner Path

1. ✅ Run `main.py`
2. ✅ Try example code above
3. ✅ Open Jupyter notebook
4. ✅ Experiment with different queries

### Intermediate Path

1. ✅ Read `ARCHITECTURE.md`
2. ✅ Modify configuration files
3. ✅ Run unit tests
4. ✅ Extend with custom retrievers

### Advanced Path

1. ✅ Study source code
2. ✅ Implement custom fusion
3. ✅ Fine-tune models
4. ✅ Deploy as API service

## 🎓 Learning Resources

### Included Documentation

- `README.md` - Complete overview
- `QUICKSTART.md` - Fast setup
- `ARCHITECTURE.md` - System design
- `PROJECT_SUMMARY.md` - Project overview

### Code Examples

- `main.py` - Comprehensive demo
- `tests/test_hybrid_search.py` - Usage examples
- `notebooks/evaluation_analysis.ipynb` - Interactive tutorial

## 🤝 Getting Help

### Check Documentation

1. Start with `README.md`
2. Review `QUICKSTART.md`
3. Deep dive with `ARCHITECTURE.md`

### Code Examples

Look at:

- `main.py` for complete workflows
- Test files for specific features
- Jupyter notebook for analysis

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run your code with detailed logs
```

## ✅ Verification Checklist

Before starting, verify:

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip list`)
- [ ] Dataset present (`data/semantic_search_dataset_2000.csv`)
- [ ] Sufficient disk space (~2GB)
- [ ] Virtual environment activated (recommended)

## 🎉 You're Ready!

Your hybrid search system is ready to use. Start with:

```python
from src.search_engine import HybridSearchEngine

engine = HybridSearchEngine('data/semantic_search_dataset_2000.csv')
engine.build_indices()
results = engine.search("your first query!", top_k=5)

for r in results:
    print(f"✨ {r['title']}")
```

Happy searching! 🔍
