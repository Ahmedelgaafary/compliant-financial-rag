from dataclasses import dataclass

from src.retrieval.evaluation import (
    recall_at_k,
    reciprocal_rank,
)
from src.retrieval.models import RetrievalResult


@dataclass(frozen=True)
class BenchmarkResult:
    """Evaluation result for one retriever."""

    retriever_name: str
    recall_at_1: float
    recall_at_3: float
    mean_reciprocal_rank: float


def evaluate_retriever(
    retriever,
    cases,
    retriever_name: str,
    top_k: int = 3,
) -> BenchmarkResult:
    """Evaluate one retriever against labeled queries."""

    recall_1_scores: list[float] = []
    recall_3_scores: list[float] = []
    reciprocal_ranks: list[float] = []

    for case in cases:
        results: tuple[RetrievalResult, ...] = (
            retriever.retrieve(
                case.query,
                top_k=top_k,
            )
        )

        recall_1_scores.append(
            recall_at_k(
                results,
                case.expected_chunk_id,
                k=1,
            )
        )

        recall_3_scores.append(
            recall_at_k(
                results,
                case.expected_chunk_id,
                k=3,
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                results,
                case.expected_chunk_id,
            )
        )

    return BenchmarkResult(
        retriever_name=retriever_name,
        recall_at_1=sum(recall_1_scores)
        / len(recall_1_scores),
        recall_at_3=sum(recall_3_scores)
        / len(recall_3_scores),
        mean_reciprocal_rank=sum(reciprocal_ranks)
        / len(reciprocal_ranks),
    )