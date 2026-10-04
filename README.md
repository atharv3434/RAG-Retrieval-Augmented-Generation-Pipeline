# RAG (Retrieval-Augmented Generation) Pipeline

A RAG system over a small fictional product knowledge base ("CloudSync," a
fictional cloud storage product), built around a from-scratch **BM25**
retriever and a pluggable generation backend — including the real
**Anthropic API** for actual LLM generation.

## A note on what was (and wasn't) testable here

This project was built in a sandboxed environment with no network access
to `api.anthropic.com`, so the real LLM generation step could not be
called or tested directly from here. Rather than fake that with a mocked
response, the project is split cleanly along the line that actually matters:

- **Retrieval (BM25 + chunking)** is the part that's fully deterministic
  and locally testable — and it was tested rigorously (see results below).
- **Generation** has two backends: `AnthropicGenerator`, which uses the
  real `anthropic` Python SDK and will work with your own API key wherever
  you run this (just untestable from inside this sandbox), and
  `ExtractiveGenerator`, a clearly-labeled non-LLM stand-in that just
  surfaces retrieved text directly, so the full pipeline is runnable and
  demonstrable without any API key at all.

This split reflects how RAG systems are usually developed and evaluated in
practice: retrieval quality and generation quality are different problems
with different evaluation methods, and retrieval can (and should) be
tested independently of which LLM eventually consumes it.

## Project structure

```
rag-pipeline/
├── config.yaml                     # chunking, BM25, retrieval, and generator settings
├── requirements.txt
├── data/
│   ├── documents/                  # the knowledge base (4 fictional product docs)
│   └── eval_queries.json           # 15 hand-labeled test queries for retrieval eval
├── src/
│   ├── utils.py                    # config + JSON loading
│   ├── chunking.py                 # splits documents into overlapping word-window chunks
│   ├── bm25.py                     # BM25 ranking algorithm, implemented from scratch
│   ├── retriever.py                # chunking + BM25 index, retrieve(query, top_k)
│   ├── generator.py                # AnthropicGenerator (real API) + ExtractiveGenerator (offline)
│   ├── rag_pipeline.py             # ties retrieval + generation together
│   ├── build_index.py              # CLI: builds and saves the BM25 index
│   ├── query.py                    # CLI: ask the pipeline a question
│   └── evaluate_retrieval.py       # CLI: precision/recall/MRR against labeled queries
├── models/
│   └── bm25_index.joblib            # saved index (after build_index.py)
├── output/
│   └── retrieval_eval_report.md     # evaluation results
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Build the index

```bash
python src/build_index.py
```

```
Chunking and indexing documents in data/documents ...
Indexed 14 chunks from 4 documents.
Index saved to models/bm25_index.joblib
```

## 2. Ask a question

With the default offline/extractive backend (no API key needed):

```bash
python src/query.py --question "How do I reset my password?"
```

```
Retrieved context:
  [account_and_security.txt] score=5.9976  "CloudSync Account and Security Guide Resetting your password..."
  ...

Answer:
[extractive stand-in — not a real generated answer, just the retrieved text]
...
```

With the real Claude API (requires `ANTHROPIC_API_KEY` in your
environment, and must be run somewhere with network access to the API):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python src/query.py --question "How do I reset my password?" --backend anthropic
```

This sends the retrieved chunks as context to Claude with a system prompt
instructing it to answer only from that context and cite its sources —
the actual "G" in RAG.

## 3. Evaluate retrieval quality

```bash
python src/evaluate_retrieval.py
```

Against 15 hand-labeled test queries (each with a known-correct source
document):

```
k=1:  precision@k=0.933  recall@k=0.900  MRR=0.967
k=3:  precision@k=0.667  recall@k=1.000  MRR=0.967
k=5:  precision@k=0.453  recall@k=1.000  MRR=0.967
```

At k=3 (the default retrieval depth), the correct document is found for
**every single test query** (recall@3 = 1.000), and it's the very top
result 93% of the time (precision@1 = 0.933, MRR = 0.967).

### A real bug caught during testing

The first version of the evaluation script computed recall@k by counting
every retrieved *chunk* that matched a relevant source — but since a
single document can contribute more than one chunk to the top-k results,
this let recall exceed 1.0 (a mathematically invalid "109%"-style result):
testing produced **recall@3 = 1.9**. The fix was to count *distinct*
relevant documents found, not chunk occurrences, which is what recall is
actually supposed to measure. Worth noting since it's an easy mistake to
make in any retrieval eval that mixes chunk-level results with
document-level relevance labels.

## How BM25 retrieval works

See the detailed docstring in `src/bm25.py`. In short: BM25 scores each
chunk against a query by combining term frequency (saturating, so a term
appearing many times doesn't dominate), inverse document frequency (rare
terms count more), and document-length normalization. It's the same
algorithm family used by default in Elasticsearch, and a strong baseline
even relative to modern embedding-based ("dense") retrieval.

## Using your own documents

1. Drop `.txt` files into `data/documents/` (replacing or adding to the
   sample CloudSync docs).
2. Re-run `python src/build_index.py`.
3. If you want to evaluate retrieval quality, write your own
   `data/eval_queries.json` with real questions and which document(s)
   should answer each one.

## Extending this project

- **Dense/hybrid retrieval**: add an embedding-based retriever (e.g. using
  a local sentence-embedding model) alongside BM25, and combine scores —
  a common real-world RAG upgrade, especially for queries that are
  semantically related but share few exact keywords with the source text.
- **Reranking**: add a second-stage reranker that re-scores BM25's top-N
  candidates more precisely before truncating to top-k.
- **Smarter chunking**: swap the fixed word-window chunker for sentence-
  or section-aware chunking, which often improves retrieval precision.
- **Generation evaluation**: once you can call the Anthropic API, build an
  evaluation of answer quality itself (e.g. an LLM-graded rubric checking
  whether the generated answer is faithful to the retrieved context and
  actually answers the question).
