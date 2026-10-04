"""
Split documents into overlapping word-window chunks.

Chunking matters for RAG quality: a whole document is often too long and
unfocused to be a good retrieval unit (it may be relevant to a query only
in one small section), while chunks that are too small lose context. Fixed
overlapping word windows are a simple, robust baseline — more advanced
chunking (sentence- or paragraph-aware, semantic chunking) can replace
this without changing anything else in the pipeline.
"""

import os


def chunk_text(text, chunk_size_words, overlap_words):
    """Split `text` into overlapping chunks of `chunk_size_words` words,
    advancing by (chunk_size_words - overlap_words) words each step.
    """
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size_words - overlap_words)
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size_words]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size_words >= len(words):
            break

    return chunks


def load_and_chunk_documents(documents_dir, chunk_size_words, overlap_words):
    """Read every .txt file in `documents_dir` and chunk it.

    Returns a list of dicts: {"chunk_id", "source", "text"}.
    """
    records = []
    filenames = sorted(f for f in os.listdir(documents_dir) if f.endswith(".txt"))

    for filename in filenames:
        path = os.path.join(documents_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text, chunk_size_words, overlap_words)
        for i, chunk in enumerate(chunks):
            records.append({
                "chunk_id": f"{filename}::chunk{i}",
                "source": filename,
                "text": chunk,
            })

    return records
