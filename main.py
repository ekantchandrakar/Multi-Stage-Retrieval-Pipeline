"""
Main demo script for Hybrid Search System
Demonstrates building indices and performing searches
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.search_engine import HybridSearchEngine
from src.utils import setup_logger, format_search_results, save_json

logger = setup_logger(__name__)


def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print("HYBRID SEMANTIC SEARCH SYSTEM - DEMO")
    print("="*80 + "\n")
    
    # Configuration
    DATA_PATH = "data/semantic_search_dataset_2000.csv"
    MODEL_DIR = "models"
    OUTPUT_DIR = "outputs"
    
    # Test queries
    TEST_QUERIES = [
        "reverse singly linked list",
        "optimize SQL join performance",
        "deploy microservice on kubernetes",
        "reduce API latency in production",
        "implement binary search tree",
        "explain ACID properties in databases",
        "design scalable REST API",
        "kubernetes pod autoscaling",
        "database indexing strategies",
        "implement heap data structure"
    ]
    
    # Initialize search engine
    print("🚀 Initializing Hybrid Search Engine...\n")
    engine = HybridSearchEngine(
        data_path=DATA_PATH,
        model_dir=MODEL_DIR,
        device="cpu"
    )
    
    # Build indices
    print("\n📊 Building search indices...")
    print("This may take a few minutes on first run...\n")
    engine.build_indices()
    
    # Perform single search demonstration
    print("\n" + "="*80)
    print("SINGLE SEARCH DEMONSTRATION")
    print("="*80 + "\n")
    
    demo_query = "reverse singly linked list"
    print(f"Query: '{demo_query}'\n")
    
    print("Comparing all three retrieval methods:\n")
    comparison = engine.compare_retrievers(demo_query, top_k=5)
    
    print("\n--- BM25 Results (Lexical) ---")
    print(format_search_results(comparison['bm25']))
    
    print("\n--- Semantic Results (Bi-Encoder) ---")
    print(format_search_results(comparison['semantic']))
    
    print("\n--- Hybrid Results (RRF + Cross-Encoder) ---")
    print(format_search_results(comparison['hybrid']))
    
    print("\n--- Overlap Analysis ---")
    overlap = comparison['overlap_analysis']
    print(f"BM25 ∩ Semantic: {overlap['bm25_semantic_overlap']} documents")
    print(f"BM25 ∩ Hybrid: {overlap['bm25_hybrid_overlap']} documents")
    print(f"Semantic ∩ Hybrid: {overlap['semantic_hybrid_overlap']} documents")
    print(f"All Three: {overlap['all_three_overlap']} documents")
    
    # Batch evaluation
    print("\n" + "="*80)
    print("BATCH EVALUATION ON TEST QUERIES")
    print("="*80 + "\n")
    
    print(f"Evaluating on {len(TEST_QUERIES)} test queries...\n")
    
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    results = engine.evaluate_on_queries(
        TEST_QUERIES,
        save_path=f"{OUTPUT_DIR}/evaluation_results.json"
    )
    
    print(f"\n✅ Evaluation complete!")
    print(f"Average search time: {results['statistics']['avg_search_time']:.3f}s")
    print(f"Results saved to: {OUTPUT_DIR}/evaluation_results.json")
    
    # Display sample results for a few queries
    print("\n" + "="*80)
    print("SAMPLE RESULTS")
    print("="*80)
    
    for i, query_result in enumerate(results['queries'][:3], 1):
        query = query_result['query']
        hybrid_results = query_result['results']['hybrid']
        
        print(f"\n{i}. Query: '{query}'")
        print(f"   Search time: {query_result['search_time']:.3f}s")
        print(f"   Top result: {hybrid_results[0]['title']}")
        print(f"   Score: {hybrid_results[0]['score']:.4f}")
    
    # System statistics
    print("\n" + "="*80)
    print("SYSTEM STATISTICS")
    print("="*80 + "\n")
    
    system_info = engine.get_system_info()
    
    print(f"Total Documents: {system_info['data']['total_documents']}")
    print(f"Categories: {list(system_info['data']['categories'].keys())}")
    print(f"BM25 Vocabulary: {system_info['bm25']['vocabulary_size']} terms")
    print(f"Semantic Embedding Dim: {system_info['semantic']['embedding_dim']}")
    print(f"FAISS Index Vectors: {system_info['semantic']['total_vectors']}")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80 + "\n")
    
    print("Next steps:")
    print("1. Check outputs/evaluation_results.json for detailed results")
    print("2. Explore the Jupyter notebooks in notebooks/ directory")
    print("3. Try your own queries using the search engine")
    print("\nExample usage:")
    print(">>> from src.search_engine import HybridSearchEngine")
    print(">>> engine = HybridSearchEngine('data/semantic_search_dataset_2000.csv')")
    print(">>> engine.load_indices()  # Load pre-built indices")
    print(">>> results = engine.search('your query here', top_k=5)")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
