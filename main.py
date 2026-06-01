"""
main.py – Hybrid Search System Demo
=====================================
Demonstrates:
  1. StaQC dataset loading (replaces old CSV)
  2. IVF-FAISS auto-selection based on corpus size
  3. Full IR evaluation: NDCG@10, MRR, Recall@10
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.search_engine import HybridSearchEngine
from src.evaluation import EvaluationSuite, GroundTruthBuilder
from src.utils import setup_logger, format_search_results, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration  ← edit these as needed
# ---------------------------------------------------------------------------

DATA_LANGUAGE = "mca_python"        
MAX_RECORDS   = 4000        
MODEL_DIR     = "models"
OUTPUT_DIR    = "outputs"
DEVICE        = "cpu"

# Test queries (StaQC / StackOverflow flavour)
TEST_QUERIES = [
    # Python queries
    "reverse a linked list in Python",
    "implement binary search tree in Python",
    "Adding a simple value to a string",
    "how to use list comprehension in Python",
    "How can I sort a 2D list?"
    "In Python, can I call the variable from main function - use global variable?"
]

EVAL_K = 10   # NDCG@10, Recall@10, etc.


# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 70)
    print("  HYBRID SEMANTIC SEARCH SYSTEM  –  Production Demo")
    print("=" * 70 + "\n")

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    #  1.  Initialise engine (StaQC, IVF-FAISS auto-selected)             #
    # ------------------------------------------------------------------ #
    print(f"🔧  Initialising engine  (language={DATA_LANGUAGE}, max_records={MAX_RECORDS:,})")
    engine = HybridSearchEngine(
        data_path="staqc",          # triggers HuggingFace StaQC download
        model_dir=MODEL_DIR,
        language=DATA_LANGUAGE,  # "man_python" | "man_sql" | "mca_python"
        max_records=MAX_RECORDS,
        device=DEVICE,
        # index_type=None → auto: Flat if N<10K, IVFFlat if N<500K, IVFPQ otherwise
    )

    print("\n📦  Building / loading indices …\n")
    engine.build_indices()

    # Print FAISS index type chosen
    sinfo = engine.get_system_info()
    faiss_type = sinfo["semantic"].get("index_type", "?")
    print(f"\n✅  Index ready")
    print(f"   Documents : {sinfo['data'].get('total_documents', '?'):,}")
    print(f"   BM25 vocab: {sinfo['bm25'].get('vocabulary_size', '?'):,}")
    print(f"   FAISS type: {faiss_type}")
    if "nlist" in sinfo["semantic"]:
        print(f"   nlist={sinfo['semantic']['nlist']}, nprobe={sinfo['semantic']['nprobe']}")

    # ------------------------------------------------------------------ #
    #  2.  Single-query retriever comparison                               #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  SINGLE-QUERY RETRIEVER COMPARISON")
    print("=" * 70)

    demo_q = TEST_QUERIES[0]
    print(f"\nQuery: '{demo_q}'\n")
    comp = engine.compare_retrievers(demo_q, top_k=5)

    for method in ("bm25", "semantic", "hybrid"):
        print(f"\n── {method.upper()} ──")
        for r in comp[method]:
            print(f"  [{r['rank']}] {r['title'][:80]}  (score={r['score']:.4f})")

    ov = comp["overlap_analysis"]
    print(
        f"\nOverlap  BM25∩Sem={ov['bm25_semantic_overlap']}  "
        f"BM25∩Hyb={ov['bm25_hybrid_overlap']}  "
        f"Sem∩Hyb={ov['semantic_hybrid_overlap']}  "
        f"All3={ov['all_three_overlap']}"
    )

    # ------------------------------------------------------------------ #
    #  3.  IR Evaluation: NDCG@10, MRR, Recall@10                         #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  IR EVALUATION  (NDCG@10 · MRR · Recall@10)")
    print("=" * 70)

    # Build pseudo-relevance ground truth using keyword matching
    print("\n🔍  Building pseudo-relevance ground truth …")
    gtb = GroundTruthBuilder(engine.metadata)
    ground_truth = gtb.build_from_keywords(
        TEST_QUERIES,
        min_relevance=2,      # doc must contain ≥2 query tokens
    )

    n_relevant = {q: len(v) for q, v in ground_truth.items()}
    print(f"   Avg relevant docs per query: "
          f"{sum(n_relevant.values()) / len(n_relevant):.1f}")

    # Run evaluation
    suite = EvaluationSuite(engine, k=EVAL_K)
    report = suite.run(
        queries=TEST_QUERIES,
        ground_truth=ground_truth,
        search_types=["bm25", "semantic", "hybrid"],
        top_k=EVAL_K,
    )

    # Pretty-print
    suite.print_report(report)

    # Comparison table
    cmp_table = suite.compare_search_types(report)
    print("\n  Comparison table:")
    print(f"  {'Method':<12}", end="")
    first_row = cmp_table["comparison_table"][0] if cmp_table["comparison_table"] else {}
    headers = [k for k in first_row if k != "search_type"]
    for h in headers:
        print(f"  {h:<18}", end="")
    print()
    print("  " + "-" * (12 + 18 * len(headers)))
    for row in cmp_table["comparison_table"]:
        print(f"  {row['search_type']:<12}", end="")
        for h in headers:
            print(f"  {str(row.get(h,'')):<18}", end="")
        print()

    # Save full report
    report_path = f"{OUTPUT_DIR}/evaluation_report.json"
    suite.save_report(report, report_path)
    print(f"\n💾  Full report saved → {report_path}")

    # ------------------------------------------------------------------ #
    #  4.  Batch evaluation (legacy helper)                                #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  BATCH EVALUATION (all 3 search types, first 5 queries)")
    print("=" * 70 + "\n")

    batch = engine.evaluate_on_queries(
        TEST_QUERIES[:5],
        save_path=f"{OUTPUT_DIR}/batch_results.json",
    )
    print(f"   Avg search time: {batch['statistics']['avg_search_time']:.3f}s")

    # ------------------------------------------------------------------ #
    #  5.  Summary                                                         #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
    print(f"\n  Outputs written to ./{OUTPUT_DIR}/")
    print("  evaluation_report.json  – per-query + aggregate IR metrics")
    print("  batch_results.json      – full result sets for 5 queries\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)
