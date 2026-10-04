"""Orchestrates the full RAG flow: retrieve relevant chunks, then generate
an answer grounded in them.
"""


class RAGPipeline:
    def __init__(self, retriever, generator, top_k=3):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def answer(self, query, top_k=None):
        k = top_k or self.top_k
        retrieved = self.retriever.retrieve(query, top_k=k)
        answer_text = self.generator.generate(query, retrieved)
        return {
            "query": query,
            "retrieved_chunks": retrieved,
            "answer": answer_text,
        }
