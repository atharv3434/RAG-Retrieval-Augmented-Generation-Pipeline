"""Ties chunking and BM25 together into a retriever: build an index over a
document directory, then retrieve the top-k most relevant chunks for a
query.
"""

import os

from chunking import load_and_chunk_documents
from bm25 import BM25, tokenize


class Retriever:
    def __init__(self, chunks, bm25):
        self.chunks = chunks   # list of {"chunk_id", "source", "text"}
        self.bm25 = bm25

    @classmethod
    def build(cls, documents_dir, chunk_size_words, chunk_overlap_words, k1=1.5, b=0.75):
        chunks = load_and_chunk_documents(documents_dir, chunk_size_words, chunk_overlap_words)
        if not chunks:
            raise ValueError(f"No chunks produced from documents in '{documents_dir}'.")

        tokenized = [tokenize(c["text"]) for c in chunks]
        bm25 = BM25(k1=k1, b=b)
        bm25.fit(tokenized)

        return cls(chunks, bm25)

    def retrieve(self, query, top_k=3):
        query_tokens = tokenize(query)
        results = self.bm25.top_k(query_tokens, top_k)

        retrieved = []
        for idx, score in results:
            chunk = self.chunks[idx]
            retrieved.append({**chunk, "score": round(float(score), 4)})
        return retrieved
