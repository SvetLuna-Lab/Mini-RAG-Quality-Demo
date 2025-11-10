# Mini-rag-quality-demo

Mini Retrieval-Augmented Generation (RAG) prototype with a simple answer quality evaluation layer.

This project is an educational demo of how a small RAG-style system can be built **without external APIs**:

- load a small set of local text documents;
- build a TF-IDF index and retrieve top-K relevant chunks for a question;
- generate an answer using a simple LLM stub (no external model required);
- compute **naive quality metrics**:
  - keyword coverage (does the answer cover the main words from the question?),
  - source overlap (how much the answer is grounded in the retrieved documents).

The focus is on **pipeline structure and evaluation**, not on the power of the language model.

---

## Project structure

```text
mini-rag-quality-demo/
  data/
    docs/
      doc1.txt        # example documents (you can replace with your own)
      doc2.txt
  src/
    __init__.py
    indexer.py        # TF-IDF index over local documents
    retriever.py      # (optional, can be merged with indexer – kept for extensibility)
    rag_pipeline.py   # RAG pipeline: retrieve + simple LLM stub
    eval.py           # simple answer quality metrics
    cli.py            # command-line entry point
  README.md
  requirements.txt
  .gitignore

You can put any .txt files into data/docs/ – the pipeline will index them.


Installation

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt


requirements.txt (minimal):

numpy
pandas
scikit-learn


Usage

Put your .txt documents into data/docs/.

Run the RAG pipeline from the command line:

python -m src.cli --question "Explain the main idea of this document."


Optional arguments:

--docs – path to the documents directory (default: data/docs).

Example:

python -m src.cli \
  --docs data/docs \
  --question "What is this project about and how does the pipeline work?"


The script will:

retrieve the most relevant document chunks,

build an answer with a simple LLM stub,

print answer quality metrics:

keyword_coverage

source_overlap,

show which documents were retrieved and their scores.


Components

DocumentIndexer (src/indexer.py)
Builds a TF-IDF index over all .txt files and supports simple similarity search.

RagPipeline (src/rag_pipeline.py)
Orchestrates retrieval + stubbed LLM answer generation.

SimpleLLMStub
A placeholder for a real language model. It simply formats the retrieved context; later it can be replaced with a real API (e.g. OpenAI, local LLM, etc.).

eval.py
Implements two naive metrics:

keyword_coverage_score(question, answer),

source_overlap_score(answer, retrieved_texts).


Possible extensions

Replace the TF-IDF index with BM25 or dense embeddings.

Plug in a real LLM API instead of the stub.

Add richer evaluation:

per-category metrics,

reference answers,

hallucination checks.


Keywords / Tags

RAG

retrieval-augmented-generation

LLM

evaluation

question-answering

information-retrieval

python

machine-learning

text-mining

prototype
