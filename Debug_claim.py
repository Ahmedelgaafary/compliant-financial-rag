"""
Diagnostic: inspect the retrieval evidence and regex match that produced
the (wrong-looking) claim value='14' for Apple revenue 2025.

Run from the repo root:
    py debug_claim.py
"""

import re

from src.agent.node import (
    _METRIC_PATTERNS,
    _VALUE_PATTERN,
    _load_document_chunks,
    _period_scoped_texts,
)
from src.agent.query_analyzer import analyze_query
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import VectorStore

query = "What was Apple revenue in 2025?"

chunks = _load_document_chunks()
retriever = HybridRetriever(
    bm25_retriever=BM25Retriever(chunks),
    vector_store=VectorStore(chunks),
)

analysis = analyze_query(query)
print("QUERY ANALYSIS:", analysis)

results = list(retriever.retrieve("Apple: " + query, top_k=10))
print(f"\nRetrieved {len(results)} chunks\n")

pattern = re.compile(
    rf"{_METRIC_PATTERNS['revenue']}"
    rf"\s*(?:was|were|is|are|of|to|=|:)?\s*"
    rf"{_VALUE_PATTERN}",
    re.IGNORECASE,
)

for i, result in enumerate(results):
    print(f"--- chunk {i} (id={result.chunk_id}, score={result.score:.4f}) ---")
    print("FULL TEXT (first 500 chars):")
    print(result.text[:500])
    print()

    scoped = _period_scoped_texts(result.text, "2025") or [result.text]

    for j, text in enumerate(scoped):
        match = pattern.search(text)
        if match:
            start = max(0, match.start() - 80)
            end = min(len(text), match.end() + 20)
            print(f"  scoped[{j}] MATCH: {match.group(0)!r}")
            print(f"  ...context...: {text[start:end]!r}")
        else:
            print(f"  scoped[{j}]: no match")
    print()