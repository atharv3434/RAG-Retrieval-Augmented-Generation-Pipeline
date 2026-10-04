"""Okapi BM25 ranking, implemented from scratch — the sparse (keyword-
based) retrieval algorithm behind this RAG pipeline's retriever.

Reference: Robertson & Zaragoza, "The Probabilistic Relevance Framework:
BM25 and Beyond" (Foundations and Trends in Information Retrieval, 2009).

BM25 scores how well a document matches a query by combining, for each
query term: how often that term appears in the document (term frequency,
saturating rather than growing unboundedly — a word appearing 10 times
isn't 10x as relevant as it appearing once), how rare that term is across
the whole corpus (inverse document frequency — rare, distinctive terms
count for more than common ones), and a length-normalization adjustment
(a long document naturally contains more term matches just by being
longer, so raw matches are normalized against the corpus's average
document length).

This is the same family of algorithm used by default in Elasticsearch and
many production retrieval systems — a strong, well-understood baseline for
retrieval even in an era of dense (embedding-based) retrieval, and one
that's simple enough to verify by hand, which is useful for trusting a
RAG pipeline's retrieval step.
"""

import math
import re
from collections import Counter


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_term_counts = []   # Counter per document
        self.doc_lengths = []
        self.avg_doc_length = 0.0
        self.doc_freq = Counter()   # how many documents contain each term
        self.n_docs = 0
        self.idf = {}

    def fit(self, tokenized_docs):
        """tokenized_docs: list of token lists, one per document."""
        self.n_docs = len(tokenized_docs)
        self.doc_term_counts = [Counter(doc) for doc in tokenized_docs]
        self.doc_lengths = [len(doc) for doc in tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / self.n_docs if self.n_docs else 0.0

        self.doc_freq = Counter()
        for counts in self.doc_term_counts:
            for term in counts.keys():
                self.doc_freq[term] += 1

        # Smoothed IDF (the "+1" inside the log keeps IDF non-negative even
        # for terms that appear in more than half the corpus).
        self.idf = {
            term: math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            for term, df in self.doc_freq.items()
        }

    def score(self, query_tokens, doc_index):
        counts = self.doc_term_counts[doc_index]
        doc_len = self.doc_lengths[doc_index]
        score = 0.0

        for term in query_tokens:
            if term not in counts:
                continue
            tf = counts[term]
            idf = self.idf.get(term, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            score += idf * (tf * (self.k1 + 1)) / denom

        return score

    def score_all(self, query_tokens):
        return [self.score(query_tokens, i) for i in range(self.n_docs)]

    def top_k(self, query_tokens, k):
        scores = self.score_all(query_tokens)
        ranked = sorted(range(self.n_docs), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in ranked[:k]]
