import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingestion.chunker import DocumentChunk
from src.retrieval.models import RetrievalResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Semantic vector retrieval using sentence-transformer embeddings."""

    def __init__(
        self,
        chunks: tuple[DocumentChunk, ...],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        if not chunks:
            raise ValueError("chunks cannot be empty")

        self.chunks = chunks
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.embeddings = self.model.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[RetrievalResult, ...]:
        """Retrieve semantically relevant chunks."""

        if not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        scores = np.dot(
            self.embeddings,
            query_embedding,
        )

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = tuple(
            RetrievalResult(
                chunk_id=self.chunks[index].chunk_id,
                document_id=self.chunks[index].document_id,
                text=self.chunks[index].text,
                score=float(scores[index]),
                page_number=self.chunks[index].page_number,
                section=self.chunks[index].section,
                document_sha256=self.chunks[index].document_sha256,
                retrieval_method="vector",
            )
            for index in ranked_indices
        )

        logger.info(
            "Vector retrieval completed: query=%r results=%d",
            query,
            len(results),
        )

        return results