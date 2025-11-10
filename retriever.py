"""
retriever.py

Simple embedding-based retriever for a mini RAG demo.

- Uses sentence-transformers to obtain text embeddings.
- Keeps embeddings in memory (sufficient for small toy corpora).
- Returns top-k most similar documents by cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Document:
    """Text document representation used by the retriever."""
    doc_id: str
    text: str
    metadata: Dict[str, Any] | None = None


class SimpleEmbeddingRetriever:
    """
    Minimal retriever for RAG-style experiments.

    Public API:
    - fit(documents)         -> build the index
    - retrieve(query, k)     -> return top-k documents for a query
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        # Load the sentence-transformers model
        self.model = SentenceTransformer(model_name, device=device)

        # In-memory storage for documents and their embeddings
        self.documents: List[Document] = []
        self._embeddings: np.ndarray | None = None

    # --------- Internal helpers ---------

    def _encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into L2-normalized embeddings.

        Normalization makes cosine similarity equivalent to dot product.
        """
        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        return emb / norms

    # --------- Public API ---------

    def fit(self, documents: Iterable[Document]) -> None:
        """
        Build the in-memory index from an iterable of Document objects.

        This method overwrites any previously stored index.
        """
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("No documents passed to retriever.fit()")

        texts = [d.text for d in self.documents]
        self._embeddings = self._encode(texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k documents for the given query.

        Returns:
            List of (Document, score) pairs sorted by descending score.
        """
        if self._embeddings is None or not len(self.documents):
            raise RuntimeError("Retriever is not fitted. Call fit() first.")

        # Encode the query; shape: (dim,)
        query_emb = self._encode([query])[0]

        # Dot product between normalized vectors = cosine similarity
        scores = self._embeddings @ query_emb  # shape: (n_docs,)

        top_k = min(top_k, len(self.documents))
        # Get indices of top-k scores (unsorted)
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        # Sort the top-k indices by score in descending order
        idx = idx[np.argsort(-scores[idx])]

        return [(self.documents[i], float(scores[i])) for i in idx]


# --------- Utility for loading a small text corpus ---------


def load_txt_corpus_from_dir(root: str | Path) -> List[Document]:
    """
    Load all `.txt` files from a directory as a tiny corpus.

    Each file becomes a Document where:
    - doc_id  = file stem (name without extension)
    - text    = file contents
    - metadata["path"] = full file path as string
    """
    root = Path(root)
    docs: List[Document] = []

    for path in sorted(root.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        docs.append(
            Document(
                doc_id=path.stem,
                text=text,
                metadata={"path": str(path)},
            )
        )

    if not docs:
        raise ValueError(f"No .txt files found in {root!r}")

    return docs
