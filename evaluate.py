"""
evaluate.py  –  Best-practices guide for IR metrics
====================================================

This script is the RECOMMENDED entry point for measuring search quality.
Run it after main.py has built the indices.

Usage
-----
    python evaluate.py                     # full evaluation
    python evaluate.py --queries 5         # quick run on 5 queries
    python evaluate.py --search-type bm25  # single search type

What you will see
-----------------
  1. Corpus statistics
  2. NDCG@10  –  ranks quality of the entire result list
  3. MRR      –  "how quickly do I find the first good result?"
  4. Recall@10 – "how much of what exists do I surface?"
  5. Side-by-side comparison table: BM25 vs Semantic vs Hybrid
  6. Per-query breakdown (worst / best queries)
  7. JSON report saved to outputs/

Interpreting the numbers
-------------------------
  NDCG@10:  > 0.7 = good,  > 0.85 = excellent
  MRR:      > 0.7 = good,  > 0.85 = excellent
  Recall@10: > 0.5 = good  (highly corpus-dependent)

  Hybrid should beat both BM25 and Semantic on most queries.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.search_engine import HybridSearchEngine
from src.evaluation import EvaluationSuite, GroundTruthBuilder, IRMetrics

# ── Default config ──────────────────────────────────────────────────────────

LANGUAGE    = "man_python"
MAX_RECORDS = 20_000
MODEL_DIR   = "models"
OUTPUT_DIR  = "outputs"
EVAL_K      = 10

TEST_QUERIES = [
    "reverse a linked list in Python",
    "optimize SQL JOIN performance with index",
    "deploy Flask app to Kubernetes",
    "reduce API response latency",
    "implement binary search tree insert",
    "explain database ACID properties",
    "design scalable REST API with pagination",
    "kubernetes horizontal pod autoscaling",
    "database indexing strategies for large tables",
    "implement min-heap in Python",
    "how to use list comprehension in Python",
    "fix N+1 query problem in Django ORM",
    "implement LRU cache",
    "difference between process and thread",
    "SQL window functions explained",
]


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="IR Evaluation – NDCG / MRR / Recall")
    p.add_argument("--queries", type=int, default=len(TEST_QUERIES),
                   help="Number of test queries to use")
    p.add_argument("--k", type=int, default=EVAL_K,
                   help="Evaluation cut-off (NDCG@k, Recall@k …)")
    p.add_argument("--search-type", choices=["bm25", "semantic", "hybrid", "all"],
                   default="all", help="Which search type(s) to benchmark")
    p.add_argument("--min-relevance", type=int, default=2,
                   help="Min query tokens that must appear in a doc for it to be 'relevant'")
    p.add_argument("--model-dir", default=MODEL_DIR)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--language", choices=["man_python", "man_sql", "mca_python", "python", "sql"], default=LANGUAGE)
    p.add_argument("--max-records", type=int, default=MAX_RECORDS)
    return p.parse_args()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    queries = TEST_QUERIES[: args.queries]
    search_types = (
        ["bm25", "semantic", "hybrid"] if args.search_type == "all"
        else [args.search_type]
    )
    Path(args.output_dir).mkdir(exist_ok=True)

    # ── 1. Load / build engine ──────────────────────────────────────────
    print("\n🔧  Loading engine …")
    engine = HybridSearchEngine(
        data_path="staqc",
        model_dir=args.model_dir,
        language=args.language,
        max_records=args.max_records,
    )
    engine.build_indices()

    sinfo = engine.get_system_info()
    print(f"   Corpus  : {sinfo['data'].get('total_documents', '?'):,} docs")
    print(f"   FAISS   : {sinfo['semantic'].get('index_type', '?')}")

    # ── 2. Build pseudo-relevance ground truth ──────────────────────────
    print(f"\n🔍  Building ground truth (min_relevance={args.min_relevance}) …")
    gtb = GroundTruthBuilder(engine.metadata)
    ground_truth = gtb.build_from_keywords(
        queries, min_relevance=args.min_relevance
    )

    labelled = sum(1 for v in ground_truth.values() if v)
    total_rel = sum(len(v) for v in ground_truth.values())
    avg_rel   = total_rel / max(len(queries), 1)
    print(f"   Queries with ≥1 relevant doc : {labelled}/{len(queries)}")
    print(f"   Avg relevant docs per query  : {avg_rel:.1f}")

    if labelled == 0:
        print("\n⚠️  No relevant docs found with current min_relevance. "
              "Try --min-relevance 1")
        sys.exit(1)

    # ── 3. Quick single-query demo (best-practices usage of IRMetrics) ──
    print("\n" + "=" * 60)
    print("  QUICK DEMO  –  IRMetrics.evaluate_query()")
    print("=" * 60)

    demo_q = queries[0]
    demo_results = engine.search(demo_q, top_k=args.k, search_type="hybrid")
    demo_retrieved = [str(r["id"]) for r in demo_results]
    demo_relevant  = ground_truth.get(demo_q, set())

    demo_metrics = IRMetrics.evaluate_query(
        retrieved=demo_retrieved,
        relevant=demo_relevant,
        k=args.k,
    )

    print(f"\n  Query: '{demo_q}'")
    print(f"  Retrieved : {demo_retrieved[:5]} …")
    print(f"  Relevant  : {list(demo_relevant)[:5]} …  (total {len(demo_relevant)})")
    print()
    for metric, value in demo_metrics.items():
        print(f"  {metric:<15}: {value:.4f}")

    # ── 4. Full benchmark ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  FULL BENCHMARK  ({len(queries)} queries, k={args.k})")
    print("=" * 60)

    suite = EvaluationSuite(engine, k=args.k)
    report = suite.run(
        queries=queries,
        ground_truth=ground_truth,
        search_types=search_types,
        top_k=args.k,
    )
    suite.print_report(report)

    # ── 5. Comparison table ─────────────────────────────────────────────
    print("  Comparison table")
    print("  " + "-" * 56)
    cmp = suite.compare_search_types(report)
    rows = cmp["comparison_table"]
    if rows:
        hdrs = [k for k in rows[0] if k != "search_type"]
        hdr_line = f"  {'type':<12}" + "".join(f"  {h:<16}" for h in hdrs)
        print(hdr_line)
        print("  " + "-" * (len(hdr_line) - 2))
        for row in rows:
            vals = f"  {row['search_type']:<12}" + "".join(
                f"  {str(row.get(h,'')):<16}" for h in hdrs
            )
            print(vals)

    # ── 6. Worst / best queries for hybrid ─────────────────────────────
    hybrid_pq = [
        q for q in report["per_query"]
        if "hybrid" in q["results"]
    ]
    if hybrid_pq:
        ranked_by_ndcg = sorted(
            hybrid_pq,
            key=lambda q: q["results"]["hybrid"]["ndcg_at_k"],
        )
        print(f"\n  ✗ Hardest query  (hybrid, NDCG@{args.k})")
        worst = ranked_by_ndcg[0]
        print(f"    '{worst['query']}'  → NDCG={worst['results']['hybrid']['ndcg_at_k']:.4f}")

        print(f"\n  ✓ Easiest query  (hybrid, NDCG@{args.k})")
        best = ranked_by_ndcg[-1]
        print(f"    '{best['query']}'  → NDCG={best['results']['hybrid']['ndcg_at_k']:.4f}")

    # ── 7. Save outputs ─────────────────────────────────────────────────
    report_path = f"{args.output_dir}/evaluation_report.json"
    suite.save_report(report, report_path)

    # Also save a compact summary
    summary = {
        "config": report["config"],
        "aggregate": report["aggregate"],
        "corpus_size": sinfo["data"].get("total_documents"),
        "faiss_type": sinfo["semantic"].get("index_type"),
    }
    summary_path = f"{args.output_dir}/evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾  Saved: {report_path}")
    print(f"💾  Saved: {summary_path}")
    print()


if __name__ == "__main__":
    main()
