"""Ask the RAG pipeline a question.

Usage:
    python src/query.py --question "How do I reset my password?"
    python src/query.py --question "..." --backend anthropic   # needs ANTHROPIC_API_KEY
    python src/query.py --interactive
"""

import argparse
import os
import sys

import joblib

sys.path.append(os.path.dirname(__file__))
from utils import load_config
from generator import get_generator
from rag_pipeline import RAGPipeline


def load_retriever(index_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No index found at '{index_path}'. Run `python src/build_index.py` first.")
    return joblib.load(index_path)


def print_result(result):
    print(f"\nQuestion: {result['query']}")
    print("\nRetrieved context:")
    for chunk in result["retrieved_chunks"]:
        print(f"  [{chunk['source']}] score={chunk['score']}  \"{chunk['text'][:80]}...\"")
    print(f"\nAnswer:\n{result['answer']}\n")


def main():
    parser = argparse.ArgumentParser(description="Ask the RAG pipeline a question.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--question", help="A question to ask")
    parser.add_argument("--backend", choices=["extractive", "anthropic"], help="Override config.yaml's generator_backend")
    parser.add_argument("--top-k", type=int, help="Number of chunks to retrieve (overrides config)")
    parser.add_argument("--interactive", action="store_true", help="Ask questions interactively")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.backend:
        config["generator_backend"] = args.backend

    retriever = load_retriever(config["index_path"])
    generator = get_generator(config)
    pipeline = RAGPipeline(retriever, generator, top_k=config.get("top_k", 3))

    if args.question:
        result = pipeline.answer(args.question, top_k=args.top_k)
        print_result(result)
    elif args.interactive:
        print("Interactive mode. Type a question and press Enter (Ctrl+C to quit).\n")
        try:
            while True:
                question = input("> ").strip()
                if not question:
                    continue
                result = pipeline.answer(question, top_k=args.top_k)
                print_result(result)
        except KeyboardInterrupt:
            print("\nGoodbye.")
    else:
        print("Provide --question \"...\" or --interactive.")


if __name__ == "__main__":
    main()
