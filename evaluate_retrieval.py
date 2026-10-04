"""Evaluate retrieval quality against a hand-labeled set of test queries.

This is the part of a RAG pipeline that genuinely can be tested rigorously
without calling an LLM at all: given a question and a known-relevant
source document, does the retriever actually surface a chunk from that
document in its top-k results? Generation quality (whether the LLM's final
answer is good) is a separate concern this script doesn't address — that
requires the "anthropic" backend and your own judgment or an LLM-graded
eval, run in an environment with API access.

Usage:
    python src/evaluate_retrieval.py [--config config.yaml]
"""

import argparse
import os
import sys

import joblib

sys.path.append(os.path.dirname(__file__))
from utils import load_config, load_json


def precision_recall_at_k(retriever, eval_queries, k):
    precisions, recalls, reciprocal_ranks = [], [], []

    for item in eval_queries:
        query, relevant_sources = item["query"], set(item["relevant_sources"])
        retrieved = retriever.retrieve(query, top_k=k)
        retrieved_sources = [c["source"] for c in retrieved]

        # Precision@k: fraction of the k retrieved *chunks* that come from a
        # relevant document (a chunk-level count, correctly bounded by k).
        relevant_chunk_hits = sum(1 for s in retrieved_sources if s in relevant_sources)
        precisions.append(relevant_chunk_hits / k)

        # Recall@k: fraction of *distinct* relevant documents that were
        # found at all. This must count each relevant source at most once —
        # counting every matching chunk (like precision does) would let
        # recall exceed 1.0 whenever more than one retrieved chunk comes
        # from the same relevant document, which is invalid for a value
        # that's supposed to be a fraction. Caught exactly this by testing:
        # an earlier version reused the chunk-level hit count for recall
        # and produced recall@3 = 1.9.
        unique_relevant_found = len(set(retrieved_sources) & relevant_sources)
        recalls.append(unique_relevant_found / len(relevant_sources) if relevant_sources else 0.0)

        rr = 0.0
        for rank, source in enumerate(retrieved_sources, 1):
            if source in relevant_sources:
                rr = 1 / rank
                break
        reciprocal_ranks.append(rr)

    return {
        "precision_at_k": sum(precisions) / len(precisions),
        "recall_at_k": sum(recalls) / len(recalls),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    retriever = joblib.load(config["index_path"])
    eval_queries = load_json(config["eval_queries_path"])
    print(f"Loaded {len(eval_queries)} labeled test queries.\n")

    results = {}
    for k in config.get("eval_top_k", [1, 3, 5]):
        metrics = precision_recall_at_k(retriever, eval_queries, k)
        results[k] = metrics
        print(f"k={k}:  precision@k={metrics['precision_at_k']:.3f}  "
              f"recall@k={metrics['recall_at_k']:.3f}  MRR={metrics['mrr']:.3f}")

    # Show any queries where the correct source wasn't found at all in the
    # largest k tested — useful for spotting retrieval failures concretely.
    max_k = max(config.get("eval_top_k", [5]))
    print(f"\nQueries where no relevant source was retrieved in top {max_k}:")
    misses = 0
    for item in eval_queries:
        retrieved = retriever.retrieve(item["query"], top_k=max_k)
        retrieved_sources = {c["source"] for c in retrieved}
        if not (retrieved_sources & set(item["relevant_sources"])):
            print(f"  - \"{item['query']}\" (expected: {item['relevant_sources']})")
            misses += 1
    if misses == 0:
        print("  (none)")

    report_path = os.path.join(os.path.dirname(config["eval_queries_path"]), "..", "output", "retrieval_eval_report.md")
    report_path = os.path.normpath(report_path)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    lines = ["# Retrieval Evaluation Report", "", f"n = {len(eval_queries)} labeled test queries", ""]
    lines.append("| k | Precision@k | Recall@k | MRR |")
    lines.append("|---|---|---|---|")
    for k, m in results.items():
        lines.append(f"| {k} | {m['precision_at_k']:.3f} | {m['recall_at_k']:.3f} | {m['mrr']:.3f} |")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
