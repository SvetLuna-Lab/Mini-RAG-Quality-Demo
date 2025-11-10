"""
Small convenience script to try the retriever without command-line args.

Usage:
    python examples/quick_demo.py
"""

from pathlib import Path

from src.retriever import SimpleEmbeddingRetriever, load_txt_corpus_from_dir
from main import run_single_query  # reuse helper from main.py


def main() -> None:
    corpus_dir = Path("docs")
    docs = load_txt_corpus_from_dir(corpus_dir)

    retriever = SimpleEmbeddingRetriever()
    retriever.fit(docs)

    queries = [
        "liquid rocket engine",
        "retrieval augmented generation",
        "quality evaluation of RAG systems",
    ]

    for q in queries:
        run_single_query(retriever, q, top_k=2)


if __name__ == "__main__":
    main()
