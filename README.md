# Mini-rag-quality-demo

Tiny, self-contained demo of a **Retrieval-Augmented Generation (RAG)** style retriever.

The goal of this repo is not to build a full RAG system, but to show a **minimal,
well-structured retrieval component**:

- loads a small text corpus from `docs/`
- builds a simple embedding-based retriever
- lets you ask queries and see the **top-k most relevant documents**
- contains **unit tests** and an **example script** so it’s easy to understand and extend

This is a good building block / portfolio project if you want to show basic
experience with *RAG quality, retrieval and evaluation*.

---

## Project structure

mini-rag-quality-demo/
├─ data/
│  └─ .gitkeep                # reserved for future data, currently empty
├─ docs/                      # tiny text corpus indexed by the retriever
│  ├─ .gitkeep                # keeps the folder in git even if you remove all txt files
│  ├─ doc1.txt
│  ├─ doc2.txt
│  ├─ doc3.txt
│  └─ doc4.txt
├─ examples/
│  ├─ quick_demo.py           # small non-interactive demo script
│  └─ .gitkeep                # keeps folder even if you remove all examples
├─ src/
│  ├─ __init__.py
│  └─ retriever.py            # core SimpleEmbeddingRetriever implementation
├─ tests/
│  ├─ __init__.py
│  └─ test_retriever.py       # unit tests for retriever + corpus loader
├─ main.py                    # CLI entry point (interactive or single-query mode)
├─ requirements.txt
├─ .gitignore
└─ README.md


Installation

Create and activate a virtual environment (optional, but recommended), then
install dependencies:

pip install -r requirements.txt

The project uses only lightweight dependencies (e.g. numpy, scikit-learn,
pytest, etc.), so it should run almost anywhere.


Usage
1. Prepare the corpus

All documents are simple .txt files in the docs/ directory.

You can edit / replace the existing ones:

doc1.txt, doc2.txt – general short texts

doc3.txt – about mini RAG systems & retrieval quality

doc4.txt – about rocket-engine telemetry & health monitoring

Or you can drop your own .txt files into docs/ – they will all be indexed.


2. Interactive demo (CLI)

Run:

python main.py


or, explicitly:

python main.py --corpus-dir docs --top-k 3


You’ll see something like:

Mini RAG retrieval demo
Type a query and press Enter.
Press Ctrl+C or send an empty line to exit.


Example queries:

liquid rocket engine

RAG quality evaluation

health monitoring telemetry

retrieval augmented generation

The script prints the top-k documents with their scores and short snippets.


3. Single query mode

If you prefer to run one query and exit:

python main.py --query "rocket engine telemetry" --top-k 2


This will:

load all .txt files from docs/

build the retriever

print only the top-2 results for the given query.


4. Example script

There is also a small non-interactive example:

python examples/quick_demo.py


It:

loads the corpus from docs/

builds SimpleEmbeddingRetriever

runs a few hard-coded example queries

prints their top-2 results

This is useful as a quick smoke test without typing in the terminal.


How it works

The core logic lives in src/retriever.py.

At a high level:


1. Corpus loading

docs = load_txt_corpus_from_dir(Path("docs"))


Each .txt file is wrapped into a simple Document object with:

doc_id – usually the file name

text – raw content


2. Embedding-based retriever

retriever = SimpleEmbeddingRetriever()
retriever.fit(docs)


Internally it:

turns each document into a basic vector representation (toy “embedding”)

stores them in memory as a matrix

can later compute similarity between a query and all documents


3. Retrieval

results = retriever.retrieve("rocket engine", top_k=3)


Returns a list of (Document, score) pairs, sorted by relevance score
(higher = more similar).

This is intentionally small and easy to read, so the retrieval logic and structure
are the focus, not heavy infrastructure.


Running tests

Unit tests live in tests/.

To run them:

pytest


Tests cover:

loading .txt files with load_txt_corpus_from_dir

simple sanity check that a clearly relevant document is ranked first by
SimpleEmbeddingRetriever

They are small but help keep the retriever behavior stable as you extend it.


Ideas for extensions

If you want to grow this demo into something bigger, here are some directions:

swap the toy embeddings for real sentence embeddings (e.g. sentence-transformers)

add different similarity metrics and compare them

log retrieval quality metrics (precision@k, recall@k) on a small labeled set

plug this retriever into a true RAG chain with an LLM

add a web UI (Streamlit / FastAPI + simple HTML form) for querying


Why this project

This repo is designed as a minimal, readable RAG-style retrieval demo that shows:

you can structure a small ML / IR project

you understand basic retrieval and similarity search

you can write simple tests and scripts around your core logic

It’s not meant to be production-grade infrastructure. It’s a good starting point
for discussions about RAG quality, retrieval evaluation, and scaling up to
larger corpora and real-world telemetry.


Tags

#python #machinelearning #nlp #retrieval #rag
#ir #embeddings #opensearch #mlops #portfolio
