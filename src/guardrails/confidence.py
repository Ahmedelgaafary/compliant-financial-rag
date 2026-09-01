"""Deterministic confidence scoring."""

from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.models import RetrievalResult
from src.verification.models import (
    VerificationResult,
    VerificationStatus,
)


@dataclass
class ConfidenceScore:
    """Aggregated confidence score and breakdown."""

    overall: float
    retrieval_confidence: float
    verification_confidence: float
    evidence_sufficiency: float
    contradictions_penalty: float


class ConfidenceScorer:
    """
    Compute confidence from retrieval and deterministic verification.

    VerificationStatus is treated as an enum rather than comparing its
    lowercase value to uppercase strings.
    """

    def __init__(
        self,
        retrieval_weight: float = 0.4,
        verification_weight: float = 0.6,
    ) -> None:
        total = (
            retrieval_weight
            + verification_weight
        )

        if total <= 0:
            raise ValueError(
                "Confidence weights must sum to a positive value."
            )

        self.retrieval_weight = (
            retrieval_weight / total
        )
        self.verification_weight = (
            verification_weight / total
        )

    def compute(
        self,
        retrieval_results: list[RetrievalResult],
        verification_results: list[VerificationResult],
    ) -> ConfidenceScore:
        """Compute a deterministic overall confidence score."""

        avg_retrieval_score = self._retrieval_confidence(
            retrieval_results
        )

        if verification_results:
            verified_count = sum(
                1
                for result in verification_results
                if result.status
                == VerificationStatus.VERIFIED
            )

            verification_conf = (
                verified_count
                / len(verification_results)
            )
        else:
            verification_conf = 0.0

        contradictions = sum(
            1
            for result in verification_results
            if result.reason.casefold()
            == "evidence_contradicts"
        )

        contradictions_penalty = (
            0.2 * min(
                contradictions,
                2,
            )
        )

        overall = (
            self.retrieval_weight
            * avg_retrieval_score
            + self.verification_weight
            * verification_conf
            - contradictions_penalty
        )

        overall = max(
            0.0,
            min(
                1.0,
                overall,
            ),
        )

        evidence_sufficiency = min(
            1.0,
            len(retrieval_results) / 5.0,
        )

        return ConfidenceScore(
            overall=overall,
            retrieval_confidence=avg_retrieval_score,
            verification_confidence=verification_conf,
            evidence_sufficiency=evidence_sufficiency,
            contradictions_penalty=(
                contradictions_penalty
            ),
        )

    @staticmethod
    def _retrieval_confidence(
        retrieval_results: list[RetrievalResult],
    ) -> float:
        """Convert retrieval scores to a [0, 1] signal."""

        if not retrieval_results:
            return 0.0

        methods = {
            result.retrieval_method
            for result in retrieval_results
        }

        if methods == {"hybrid_rrf"}:
            # HybridRetriever default is RRF k=60.
            max_rrf = 2.0 / 61.0

            normalized = [
                min(
                    1.0,
                    max(
                        0.0,
                        result.score / max_rrf,
                    ),
                )
                for result in retrieval_results
            ]

            return ConfidenceScorer._weighted_top_k(
                normalized
            )

        if methods == {"vector"}:
            normalized = [
                min(
                    1.0,
                    max(
                        0.0,
                        (float(result.score) + 1.0)
                        / 2.0,
                    ),
                )
                for result in retrieval_results
            ]

            return ConfidenceScorer._weighted_top_k(
                normalized
            )

        if methods == {"bm25"}:
            # BM25 is not calibrated as probability.
            return 1.0

        normalized = [
            min(
                1.0,
                max(
                    0.0,
                    float(result.score),
                ),
            )
            for result in retrieval_results
        ]

        return ConfidenceScorer._weighted_top_k(
            normalized
        )

    @staticmethod
    def _weighted_top_k(
        scores: list[float],
    ) -> float:
        """Emphasize the strongest retrieved evidence."""

        if not scores:
            return 0.0

        ordered = sorted(
            scores,
            reverse=True,
        )

        if len(ordered) == 1:
            return ordered[0]

        if len(ordered) == 2:
            return (
                0.60 * ordered[0]
                + 0.40 * ordered[1]
            )

        remaining = ordered[2:]

        remaining_average = (
            sum(remaining) / len(remaining)
            if remaining
            else 0.0
        )

        confidence = (
            0.50 * ordered[0]
            + 0.30 * ordered[1]
            + 0.20 * remaining_average
        )

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )