"""Guardrail pipeline orchestration."""

import logging
from dataclasses import dataclass
from typing import List, Optional

from src.retrieval.models import RetrievalResult
from src.verification.models import VerificationResult, VerificationStatus

from .confidence import ConfidenceScore, ConfidenceScorer
from .generation_guard import GenerationGuard, GenerationGuardResult
from .policies import GuardrailPolicies
from .risk_engine import RiskAssessment, RiskEngine
from .validation import InputValidator, OutputValidator, RetrievalValidator

logger = logging.getLogger(__name__)


@dataclass
class GuardrailPipelineResult:
    """Final result after running all guardrails."""

    input_valid: bool
    retrieval_valid: bool
    retrieval_issues: List[str]
    confidence_score: Optional[ConfidenceScore]
    risk_assessment: Optional[RiskAssessment]
    should_route_to_audit: bool
    generation_guard_result: Optional[GenerationGuardResult]
    output_valid: bool
    output_issues: List[str]
    final_safe_output: str


class GuardrailRunner:
    """Orchestrate the guardrail pipeline."""

    def __init__(self, policies: GuardrailPolicies):
        self.policies = policies
        self.input_validator = InputValidator(policies)
        self.retrieval_validator = RetrievalValidator(policies)
        self.output_validator = OutputValidator(policies)
        self.confidence_scorer = ConfidenceScorer()
        self.risk_engine = RiskEngine(policies)
        self.generation_guard = GenerationGuard(policies)

    def run_full_pipeline(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        verification_results: List[VerificationResult],
        raw_llm_output: Optional[str] = None,
    ) -> GuardrailPipelineResult:
        """Execute all guardrails in order."""
        logger.info("=" * 60)
        logger.info("Starting guardrail pipeline")
        logger.info(f"Query: {query[:100]}...")
        logger.info(f"Retrieval results: {len(retrieval_results)}")
        logger.info(f"Verification results: {len(verification_results)}")

        # 1. Input Validation
        input_valid = self.input_validator.validate(query)
        logger.info(f"Input validation: {'PASSED' if input_valid else 'FAILED'}")

        if not input_valid:
            logger.error("Input validation failed")
            return GuardrailPipelineResult(
                input_valid=False,
                retrieval_valid=False,
                retrieval_issues=[],
                confidence_score=None,
                risk_assessment=None,
                should_route_to_audit=False,
                generation_guard_result=None,
                output_valid=False,
                output_issues=["INVALID_INPUT_QUERY"],
                final_safe_output="Invalid query. Please rephrase.",
            )

        # 2. Retrieval Validation
        retrieval_valid, retrieval_issues = self.retrieval_validator.validate(retrieval_results)
        logger.info(f"Retrieval validation: {'PASSED' if retrieval_valid else 'FAILED'}")
        if retrieval_issues:
            logger.warning(f"Retrieval issues: {retrieval_issues}")

        # 3. Confidence Scoring
        confidence = self.confidence_scorer.compute(retrieval_results, verification_results)
        logger.info(f"Confidence score: {confidence.overall:.3f}")

        # 4. Risk Assessment
        risk = self.risk_engine.assess(retrieval_results, verification_results, confidence)
        logger.info(f"Risk level: {risk.risk_level}, score: {risk.risk_score:.3f}")
        logger.info(f"Recommended action: {risk.recommended_action}")

        # 5. Generation Guard (only if we have LLM output and not all verified)
        gen_guard_result = None
        all_verified = bool(verification_results) and all(
            result.status == VerificationStatus.VERIFIED for result in verification_results
        )

        if raw_llm_output and not all_verified:
            logger.info("Running generation guard")
            gen_guard_result = self.generation_guard.guard(raw_llm_output, verification_results)
            logger.info(f"Generation guard: {'SAFE' if gen_guard_result.is_safe else 'UNSAFE'}")

        # 6. Output Validation
        output_valid = True
        output_issues = []

        if raw_llm_output:
            final_output = raw_llm_output
            if gen_guard_result:
                final_output = gen_guard_result.sanitized_text
                if not gen_guard_result.is_safe:
                    output_valid = False
                    output_issues.extend(gen_guard_result.issues)
                    logger.warning(f"Generation guard issues: {gen_guard_result.issues}")

            validated, validation_issues = self.output_validator.validate(
                final_output,
                verification_results,
                max(confidence.overall, 1.0 if all_verified else 0.0),
            )
            output_valid = output_valid and validated
            output_issues.extend(validation_issues)
            logger.info(f"Output validation: {'PASSED' if output_valid else 'FAILED'}")

        elif all_verified:
            final_output = "Verified claims pending answer generation."
            output_valid = True
            logger.info("All claims verified, answer generation pending")
        else:
            final_output = "No answer generated due to risk constraints."
            output_valid = False
            output_issues.append("NO_LLM_OUTPUT")
            logger.warning("No LLM output and not all claims verified")

        # 7. Determine if we should route to audit
        should_route_to_audit = (
            risk.recommended_action in ["HUMAN_REVIEW", "BLOCK"]
            or not output_valid
            or not retrieval_valid
        )

        logger.info(f"Should route to audit: {should_route_to_audit}")
        logger.info("=" * 60)

        return GuardrailPipelineResult(
            input_valid=input_valid,
            retrieval_valid=retrieval_valid,
            retrieval_issues=retrieval_issues,
            confidence_score=confidence,
            risk_assessment=risk,
            should_route_to_audit=should_route_to_audit,
            generation_guard_result=gen_guard_result,
            output_valid=output_valid,
            output_issues=output_issues,
            final_safe_output=final_output,
        )