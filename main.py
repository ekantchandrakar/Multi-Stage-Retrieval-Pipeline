"""
main.py – Hybrid Search System Demo
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.search_engine import HybridSearchEngine
from src.evaluation import EvaluationSuite, GroundTruthBuilder
from src.utils import setup_logger, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_LANGUAGE = "mca_python"   # mca_python has ~40K rows
MAX_RECORDS   = 4_000          # start small; raise to 20_000 for production
MODEL_DIR     = "models"
OUTPUT_DIR    = "outputs"
DEVICE        = "cpu"

TEST_QUERIES = [
    "reverse a linked list in Python",
    "implement binary search tree in Python",
    "how to use list comprehension in Python",
    "how to sort a 2D list in Python",
    "read a file line by line in Python",
    "convert string to integer in Python",
    "find duplicates in a list Python",
    "merge two dictionaries Python",
    "how to use lambda function in Python",
    "exception handling try except Python",
]
EVAL_K = 10

# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 70)
    print("  HYBRID SEMANTIC SEARCH SYSTEM  –  Production Demo")
    print("=" * 70 + "\n")

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # ── 1. Engine ────────────────────────────────────────────────────────
    print(f"🔧  language={DATA_LANGUAGE}, max_records={MAX_RECORDS:,}\n")
    engine = HybridSearchEngine(
        data_path="staqc",
        model_dir=MODEL_DIR,
        language=DATA_LANGUAGE,
        max_records=MAX_RECORDS,
        device=DEVICE,
    )

    # ── 2. Build (or load) indices ───────────────────────────────────────
    print("📦  Building / loading indices …\n")
    engine.build_indices(force_rebuild=False)

    # ── 3. Info ──────────────────────────────────────────────────────────
    info = engine.get_system_info()
    print("=" * 70)
    print("  INDEX SUMMARY")
    print("=" * 70)
    print(f"  Documents : {info['data'].get('total_documents','?'):,}")
    print(f"  BM25 vocab: {info['bm25'].get('vocabulary_size','?'):,}")
    print(f"  FAISS type: {info['semantic'].get('index_type','?')}")
    print(f"  FAISS vecs: {info['semantic'].get('total_vectors','?'):,}")
    if info["semantic"].get("nlist"):
        print(f"  nlist={info['semantic']['nlist']}, nprobe={info['semantic']['nprobe']}")

    # ── 4. Single query comparison ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SINGLE-QUERY RETRIEVER COMPARISON")
    print("=" * 70)
    demo_q = TEST_QUERIES[0]
    print(f"\nQuery: '{demo_q}'\n")
    comp = engine.compare_retrievers(demo_q, top_k=5)
    for method in ("bm25", "semantic", "hybrid"):
        print(f"── {method.upper()} ──")
        for r in comp[method]:
            print(f"  [{r['rank']}] {r['title'][:75]}  (score={r['score']:.4f})")
        print()
    ov = comp["overlap_analysis"]
    print(f"Overlap  BM25∩Sem={ov['bm25_semantic_overlap']}  "
          f"BM25∩Hyb={ov['bm25_hybrid_overlap']}  "
          f"Sem∩Hyb={ov['semantic_hybrid_overlap']}  "
          f"All3={ov['all_three_overlap']}")

    # ── 5. IR Evaluation ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  IR EVALUATION  (NDCG@10 · MRR · Recall@10)")
    print("=" * 70 + "\n")

    gtb = GroundTruthBuilder(engine.metadata)
    gt  = gtb.build_from_keywords(TEST_QUERIES, min_relevance=2)

    labelled = sum(1 for v in gt.values() if v)
    if labelled == 0:
        print("  ⚠  min_relevance=2 found 0 relevant docs → retrying with 1 …")
        gt = gtb.build_from_keywords(TEST_QUERIES, min_relevance=1)
        labelled = sum(1 for v in gt.values() if v)

    avg_rel = sum(len(v) for v in gt.values()) / max(len(TEST_QUERIES), 1)
    print(f"  Queries with ≥1 relevant doc: {labelled}/{len(TEST_QUERIES)}")
    print(f"  Avg relevant docs per query : {avg_rel:.1f}\n")

    suite  = EvaluationSuite(engine, k=EVAL_K)
    report = suite.run(TEST_QUERIES, gt, search_types=["bm25","semantic","hybrid"], top_k=EVAL_K)
    suite.print_report(report)

    cmp  = suite.compare_search_types(report)
    rows = cmp["comparison_table"]
    if rows:
        hdrs = [k for k in rows[0] if k != "search_type"]
        print(f"  {'Method':<12}" + "".join(f"  {h:<18}" for h in hdrs))
        print("  " + "-" * (12 + 20 * len(hdrs)))
        for row in rows:
            print(f"  {row['search_type']:<12}" +
                  "".join(f"  {str(row.get(h,'')):<18}" for h in hdrs))

    suite.save_report(report, f"{OUTPUT_DIR}/evaluation_report.json")
    print(f"\n💾  Evaluation report → {OUTPUT_DIR}/evaluation_report.json")

    # ── 6. Batch evaluation ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BATCH EVALUATION (first 5 queries)")
    print("=" * 70 + "\n")
    batch = engine.evaluate_on_queries(
        TEST_QUERIES[:5],
        save_path=f"{OUTPUT_DIR}/batch_results.json",
    )
    print(f"  Avg search time: {batch['statistics']['avg_search_time']:.3f}s")

    # ── 7. Done ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DONE — files created:")
    print("=" * 70)
    print(f"  models/bm25_index.pkl")
    print(f"  models/faiss_index.bin")
    print(f"  models/embeddings.npy")
    print(f"  models/data_cache/staqc_{DATA_LANGUAGE}_{MAX_RECORDS}.csv")
    print(f"  outputs/evaluation_report.json")
    print(f"  outputs/batch_results.json")
    print("\n  Second run will load from disk — no re-download.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)