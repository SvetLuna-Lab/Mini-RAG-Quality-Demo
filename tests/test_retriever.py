import pathlib

from src.retriever import (
    SimpleEmbeddingRetriever,
    load_txt_corpus_from_dir,
)


def test_load_txt_corpus_from_dir_reads_txt_files(tmp_path: pathlib.Path) -> None:
    """
    Ensure that `load_txt_corpus_from_dir` correctly reads .txt files
    and wraps them into Document objects.
    """
    # create a couple of temporary .txt files
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "b.txt").write_text("rocket telemetry", encoding="utf-8")
    # and one non-txt file that must be ignored
    (tmp_path / "ignore.md").write_text("# not a txt", encoding="utf-8")

    docs = load_txt_corpus_from_dir(tmp_path)

    doc_ids = {d.doc_id for d in docs}
    texts = {d.text for d in docs}

    assert len(docs) == 2
    assert doc_ids == {"a.txt", "b.txt"}
    assert "hello world" in texts
    assert "rocket telemetry" in texts


def test_retriever_ranks_relevant_document_first() -> None:
    """
    Simple sanity check: document with the most relevant content
    should receive the highest score.
    """
    docs = [
        # intentionally tiny and simple
        type("Doc", (), {"doc_id": "cat.txt", "text": "a cute little cat"})(),
        type("Doc", (), {"doc_id": "rocket.txt", "text": "liquid rocket engine telemetry and thrust"})(),  # noqa: E501
    ]

    retriever = SimpleEmbeddingRetriever()
    retriever.fit(docs)

    query = "rocket engine"
    results = retriever.retrieve(query, top_k=2)

    # first hit should be the rocket-related document
    top_doc, top_score = results[0]
    second_doc, second_score = results[1]

    assert top_doc.doc_id == "rocket.txt"
    assert top_score >= second_score
    assert top_score > 0.0
