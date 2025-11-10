"""
main.py

Mini RAG quality demo.

- Loads a tiny text corpus from the `docs/` directory.
- Builds an embedding-based retriever.
- Lets you ask queries and see the top-k most relevant documents.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.retriever import SimpleEmbeddingRetriever, load_txt_corpus_from_dir


def build_retriever(corpus_dir: Path) -> SimpleEmbeddingRetriever:
    """
    Load documents from `corpus_dir` and build a retriever on top of them.
    """
    print(f"Loading corpus from: {corpus_dir.resolve()}")
    docs = load_txt_corpus_from_dir(corpus_dir)

    print(f"Loaded {len(docs)} documents. Building index...")
    retriever = SimpleEmbeddingRetriever()
    retriever.fit(docs)

    print("Retriever is ready.\n")
    return retriever


def run_single_query(
    retriever: SimpleEmbeddingRetriever,
    query: str,
    top_k: int = 3,
) -> None:
    """
    Run a single query and print top-k results.
    """
    results = retriever.retrieve(query, top_k=top_k)

    print(f"\nTop {len(results)} results for query: {query!r}\n")
    for rank, (doc, score) in enumerate(results, start=1):
        # Make a short one-line snippet from the document text
        snippet = doc.text.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."

        print(f"[{rank}] score={score:.3f} | id={doc.doc_id}")
        print(f"    {snippet}\n")


def interactive_loop(
    retriever: SimpleEmbeddingRetriever,
    top_k: int = 3,
) -> None:
    """
    Simple REPL for querying the retriever from the terminal.
    """
    print("Mini RAG retrieval demo")
    print("Type a query and press Enter.")
    print("Press Ctrl+C or send an empty line to exit.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            print("Bye!")
            break

        run_single_query(retriever, query, top_k=top_k)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Mini RAG quality demo: tiny embedding-based retriever."
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="docs",
        help="Directory with .txt files to index (default: docs/).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of documents to retrieve (default: 3).",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="Optional single query. If not provided, runs in interactive mode.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    corpus_dir = Path(args.corpus_dir)
    retriever = build_retriever(corpus_dir)

    if args.query:
        run_single_query(retriever, args.query, top_k=args.top_k)
    else:
        interactive_loop(retriever, top_k=args.top_k)


if __name__ == "__main__":
    main()
