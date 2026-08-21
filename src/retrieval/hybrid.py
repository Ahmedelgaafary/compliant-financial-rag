from collections import defaultdict

from src.retrieval.models import RetrievalResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Combine BM25 and vector retrieval using RRF."""

    def __init__(
        self,
        bm25_retriever,
        vector_store,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")

        self.bm25_retriever = bm25_retriever
        self.vector_store = vector_store
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[RetrievalResult, ...]:
        """Retrieve and fuse results from BM25 and vector search."""

        if not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        bm25_results = self.bm25_retriever.retrieve(
            query,
            top_k=top_k,
        )

        vector_results = self.vector_store.retrieve(
            query,
            top_k=top_k,
        )

        scores: dict[str, float] = defaultdict(float)
        results_by_id: dict[str, RetrievalResult] = {}

        for rank, result in enumerate(bm25_results, start=1):
            scores[result.chunk_id] += 1 / (self.rrf_k + rank)
            results_by_id[result.chunk_id] = result

        for rank, result in enumerate(vector_results, start=1):
            scores[result.chunk_id] += 1 / (self.rrf_k + rank)

            if result.chunk_id not in results_by_id:
                results_by_id[result.chunk_id] = result

        ranked_chunk_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )[:top_k]

        results = tuple(
            RetrievalResult(
                chunk_id=chunk_id,
                document_id=results_by_id[chunk_id].document_id,
                text=results_by_id[chunk_id].text,
                score=scores[chunk_id],
                page_number=results_by_id[chunk_id].page_number,
                section=results_by_id[chunk_id].section,
                document_sha256=results_by_id[
                    chunk_id
                ].document_sha256,
                retrieval_method="hybrid_rrf",
            )
            for chunk_id in ranked_chunk_ids
        )

        logger.info(
            "Hybrid retrieval completed: query=%r results=%d",
            query,
            len(results),
        )

        return results