from rank_bm25 import BM25Okapi

from src.ingestion.chunker import DocumentChunk
from src.retrieval.models import RetrievalResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BM25Retriever:
    """Lexical retrieval using the BM25 ranking algorithm."""

    def __init__(
        self,
        chunks: tuple[DocumentChunk, ...],
    ) -> None:
        if not chunks:
            raise ValueError("chunks cannot be empty")

        self.chunks = chunks

        tokenized_documents = [
            self._tokenize(chunk.text)
            for chunk in chunks
        ]

        self._bm25 = BM25Okapi(tokenized_documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[RetrievalResult, ...]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: User's search query.
            top_k: Maximum number of results.

        Returns:
            Ranked retrieval results.
        """

        if not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_tokens = self._tokenize(query)

        scores = self._bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = tuple(
            RetrievalResult(
                chunk_id=self.chunks[index].chunk_id,
                document_id=self.chunks[index].document_id,
                text=self.chunks[index].text,
                score=float(scores[index]),
                page_number=self.chunks[index].page_number,
                section=self.chunks[index].section,
                document_sha256=self.chunks[index].document_sha256,
                retrieval_method="bm25",
            )
            for index in ranked_indices
        )

        logger.info(
            "BM25 retrieval completed: query=%r results=%d",
            query,
            len(results),
        )

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25."""

        return text.lower().split()