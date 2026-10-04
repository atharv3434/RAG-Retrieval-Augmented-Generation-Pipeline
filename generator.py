"""
Generation backends for the RAG pipeline.

Two backends are provided:

- **AnthropicGenerator**: calls the real Claude API via the official
  `anthropic` Python SDK. This is the actual LLM generation step of a real
  RAG system, and will work wherever you run it with network access to
  the Anthropic API and a valid ANTHROPIC_API_KEY. It could not be tested
  from within the sandboxed environment this project was built in (which
  has no network access to api.anthropic.com), so test it in your own
  environment before relying on it.
  
- **ExtractiveGenerator**: a non-LLM, offline stand-in that does no
  generation at all — it just surfaces the retrieved text directly. This
  exists so the full pipeline (retrieval -> "answer") is runnable and
  testable without an API key or network access, and so you can sanity
  check retrieval quality independent of generation quality. It is not a
  substitute for a real LLM and will produce noticeably worse, un-synthesized
  answers — that's expected.
"""

import os


SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context. If the context "
    "does not contain enough information to answer, say so explicitly "
    "rather than guessing. Cite which source document(s) you used."
)


def build_prompt(query, retrieved_chunks):
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
    context = "\n\n".join(context_blocks)

    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context above, and mention which source(s) you used."
    )


class AnthropicGenerator:
    def __init__(self, model="claude-sonnet-5", max_tokens=500, api_key=None):
        import anthropic  # imported here so the extractive backend never needs this dependency available

        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, query, retrieved_chunks):
        prompt = build_prompt(query, retrieved_chunks)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class ExtractiveGenerator:
    """No LLM call at all — just formats the retrieved chunks as the
    "answer," clearly labeled as such. Useful for testing retrieval and
    the overall pipeline wiring without needing an API key.
    """

    def generate(self, query, retrieved_chunks):
        if not retrieved_chunks:
            return "[extractive stand-in] No relevant context was retrieved for this question."

        lines = [
            "[extractive stand-in — not a real generated answer, just the retrieved text]",
            "",
        ]
        for chunk in retrieved_chunks:
            lines.append(f"From {chunk['source']} (BM25 score {chunk['score']}):")
            lines.append(chunk["text"])
            lines.append("")
        return "\n".join(lines).strip()


def get_generator(config):
    backend = config.get("generator_backend", "extractive")
    if backend == "anthropic":
        return AnthropicGenerator(
            model=config.get("anthropic_model", "claude-sonnet-5"),
            max_tokens=config.get("anthropic_max_tokens", 500),
        )
    if backend == "extractive":
        return ExtractiveGenerator()
    raise ValueError(f"Unknown generator_backend: {backend}")
