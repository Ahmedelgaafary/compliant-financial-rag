from typing import List

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import RetrievalResult


class EvidenceAgent:
    """Wrapper around your HybridRetriever."""

    def __init__(self, retriever: HybridRetriever = None):
        self.retriever = retriever or HybridRetriever()

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        return list(self.retriever.retrieve(query, top_k=top_k))