
"""
Purpose: Provide input, retrieval, and output validators.

These validators enforce guardrail checks at different stages of the
agent workflow.

Important:
    RetrievalResult.score does not always represent a probability or
    confidence score. In particular, hybrid retrieval uses Reciprocal
    Rank Fusion (RRF), whose score scale is much smaller than [0, 1].

    Therefore, this module converts hybrid RRF scores into a normalized
    confidence signal before comparing them with the configured
    min_retrieval_confidence threshold.
"""

from typing import List

from src.retrieval.models import RetrievalResult
from src.verification.models import VerificationResult

from .policies import GuardrailPolicies


class InputValidator:
    """
    Checks the user query before any processing.
    """

    def __init__(self, policies: GuardrailPolicies):
        self.policies = policies

    def validate(self, query: str) -> bool:
        """
        Return True if the query is valid, otherwise False.
        """

        # Prevent empty queries.
        if not query or not query.strip():
            return False

        # Check for forbidden entities.
        for entity in self.policies.forbidden_entities:
            if entity.lower() in query.lower():
                return False

        # Prevent excessively large queries.
        if len(query) > 10000:
            return False

        return True


class RetrievalValidator:
    """
    Checks retrieved evidence before it is passed to the LLM.

    The validator handles different retrieval score semantics:

    - hybrid_rrf:
        Score is a Reciprocal Rank Fusion score, not a probability.
        It is normalized against the theoretical maximum RRF score.

    - vector:
        Score is cosine similarity and is expected to be approximately
        in the range [-1, 1]. It is converted to a [0, 1] confidence.

    - bm25:
        BM25 scores are query/corpus dependent and therefore are not
        interpreted as probabilities. BM25-only validation relies on
        evidence presence and provenance rather than applying the
        hybrid confidence threshold directly.
    """

    def __init__(self, policies: GuardrailPolicies):
        self.policies = policies

    def validate(
        self,
        retrieval_results: List[RetrievalResult],
    ) -> tuple[bool, List[str]]:
        """
        Return (is_valid, issues) for the retrieved evidence.
        """

        issues: List[str] = []

        # ---------------------------------------------------------
        # 1. Evidence must exist.
        # ---------------------------------------------------------
        if not retrieval_results:
            issues.append("NO_EVIDENCE_RETRIEVED")
            return False, issues

        # ---------------------------------------------------------
        # 2. Every result must have complete provenance.
        # ---------------------------------------------------------
        for result in retrieval_results:
            if (
                not result.document_id
                or not result.chunk_id
                or not result.document_sha256
            ):
                issues.append(
                    f"MISSING_PROVENANCE for chunk {result.chunk_id}"
                )
                return False, issues

            if not result.text or not result.text.strip():
                issues.append(
                    f"EMPTY_EVIDENCE_TEXT for chunk {result.chunk_id}"
                )
                return False, issues

        # ---------------------------------------------------------
        # 3. Check evidence quantity.
        # ---------------------------------------------------------
        if len(retrieval_results) < self.policies.min_evidence_chunks:
            issues.append("INSUFFICIENT_EVIDENCE")

        # ---------------------------------------------------------
        # 4. Evaluate retrieval confidence according to the
        #    retrieval method.
        # ---------------------------------------------------------
        confidence = self._compute_retrieval_confidence(
            retrieval_results
        )

        if confidence < self.policies.min_retrieval_confidence:
            issues.append(
                "LOW_RETRIEVAL_CONFIDENCE"
            )

        return len(issues) == 0, issues

    def _compute_retrieval_confidence(
        self,
        retrieval_results: List[RetrievalResult],
    ) -> float:
        """
        Convert retrieval scores into a normalized confidence value.

        For hybrid RRF retrieval, the maximum possible score for a result
        ranked first by both BM25 and vector retrieval is:

            1 / (rrf_k + 1) + 1 / (rrf_k + 1)

        With the default RRF k=60:

            2 / 61 ~= 0.03279

        Therefore, directly comparing an RRF score such as 0.02 with a
        confidence threshold of 0.5 is incorrect.

        The current HybridRetriever uses the default RRF k=60 and does
        not expose the value through RetrievalResult. We therefore use
        the standard default RRF constant here.

        The confidence calculation emphasizes the strongest retrieved
        evidence rather than treating all top-k results equally.
        """

        if not retrieval_results:
            return 0.0

        methods = {
            result.retrieval_method
            for result in retrieval_results
        }

        # ---------------------------------------------------------
        # Hybrid RRF
        # ---------------------------------------------------------
        if "hybrid_rrf" in methods:
            rrf_k = 60

            max_rrf_score = 2.0 / (rrf_k + 1)

            if max_rrf_score <= 0:
                return 0.0

            normalized_scores = [
                min(
                    1.0,
                    max(
                        0.0,
                        result.score / max_rrf_score,
                    ),
                )
                for result in retrieval_results
                if result.retrieval_method == "hybrid_rrf"
            ]

            if not normalized_scores:
                return 0.0

            return self._weighted_top_k_confidence(
                normalized_scores
            )

        # ---------------------------------------------------------
        # Vector retrieval
        # ---------------------------------------------------------
        if methods == {"vector"}:
            normalized_scores = [
                self._normalize_cosine_similarity(result.score)
                for result in retrieval_results
            ]

            return self._weighted_top_k_confidence(
                normalized_scores
            )

        # ---------------------------------------------------------
        # BM25-only retrieval
        # ---------------------------------------------------------
        #
        # BM25 scores are corpus/query dependent and cannot safely be
        # interpreted as [0, 1] confidence without calibration.
        #
        # In a BM25-only configuration, evidence presence and
        # provenance validation are therefore sufficient here.
        #
        if methods == {"bm25"}:
            return 1.0

        # ---------------------------------------------------------
        # Unknown/mixed retrieval methods.
        # ---------------------------------------------------------
        #
        # Be conservative. If scores are explicitly normalized to
        # [0, 1], use them; otherwise do not manufacture confidence.
        #
        normalized_scores = [
            min(
                1.0,
                max(
                    0.0,
                    float(result.score),
                ),
            )
            for result in retrieval_results
        ]

        return self._weighted_top_k_confidence(
            normalized_scores
        )

    @staticmethod
    def _normalize_cosine_similarity(score: float) -> float:
        """
        Convert cosine similarity from [-1, 1] to [0, 1].
        """

        normalized = (float(score) + 1.0) / 2.0

        return min(
            1.0,
            max(
                0.0,
                normalized,
            ),
        )

    @staticmethod
    def _weighted_top_k_confidence(
        scores: List[float],
    ) -> float:
        """
        Calculate a deterministic confidence score emphasizing the
        strongest retrieved evidence.

        Weights:

            rank 1 -> 0.50
            rank 2 -> 0.30
            remaining -> 0.20 distributed equally

        This avoids allowing many weak tail results to dominate the
        quality of the strongest evidence.
        """

        if not scores:
            return 0.0

        ordered_scores = sorted(
            scores,
            reverse=True,
        )

        if len(ordered_scores) == 1:
            return ordered_scores[0]

        if len(ordered_scores) == 2:
            return (
                0.60 * ordered_scores[0]
                + 0.40 * ordered_scores[1]
            )

        top_score = ordered_scores[0]
        second_score = ordered_scores[1]

        remaining_scores = ordered_scores[2:]

        if remaining_scores:
            remaining_average = (
                sum(remaining_scores)
                / len(remaining_scores)
            )
        else:
            remaining_average = 0.0

        confidence = (
            0.50 * top_score
            + 0.30 * second_score
            + 0.20 * remaining_average
        )

        return min(
            1.0,
            max(
                0.0,
                confidence,
            ),
        )


class OutputValidator:
    """
    Final check on the generated answer before returning it to the user.
    """

    def __init__(self, policies: GuardrailPolicies):
        self.policies = policies

    def validate(
        self,
        generated_answer: str,
        verification_results: List[VerificationResult],
        confidence_score: float,
    ) -> tuple[bool, List[str]]:
        """
        Ensure the generated answer does not contain unsupported
        statements, hallucinated citations, or unverified claims.
        """

        issues: List[str] = []

        # ---------------------------------------------------------
        # 1. Numeric verification bookkeeping.
        # ---------------------------------------------------------
        #
        # The detailed numeric claim-to-answer comparison is handled
        # elsewhere by deterministic claim extraction/verification.
        #
        # We retain the verified numeric values here so this validator
        # can be extended without changing its public interface.
        #
        verified_numeric_claims = set()

        for result in verification_results:
            claim_type = getattr(
                result,
                "claim_type",
                "",
            )

            status = getattr(
                result,
                "status",
                None,
            )

            if (
                status == "VERIFIED"
                and "numeric" in str(claim_type).lower()
            ):
                normalized_value = getattr(
                    result,
                    "normalized_value",
                    None,
                )

                if normalized_value is not None:
                    verified_numeric_claims.add(
                        normalized_value
                    )

        # ---------------------------------------------------------
        # 2. Generated answer must have sufficient confidence.
        # ---------------------------------------------------------
        if confidence_score < self.policies.min_overall_confidence:
            issues.append(
                "LOW_CONFIDENCE_ANSWER"
            )

        # ---------------------------------------------------------
        # 3. Citation validation.
        # ---------------------------------------------------------
        #
        # Citation/provenance validation is performed by the
        # FinalSafetyValidator, which has access to the actual
        # retrieval results.
        #
        # This validator intentionally does not duplicate that logic.
        #

        # ---------------------------------------------------------
        # 4. Numeric mismatch blocking.
        # ---------------------------------------------------------
        if self.policies.block_on_numeric_mismatch:
            for result in verification_results:
                reason = str(
                    getattr(
                        result,
                        "reason",
                        "",
                    )
                ).upper()

                if reason == "NUMERIC_MISMATCH":
                    issues.append(
                        "NUMERIC_MISMATCH_BLOCKED"
                    )
                    break

        return len(issues) == 0, issues

