# src/guardrails/runner.py
from dataclasses import dataclass
from typing import List, Optional

from src.retrieval.models import RetrievalResult
from src.verification.models import VerificationResult

from .confidence import ConfidenceScore, ConfidenceScorer
from .generation_guard import GenerationGuard, GenerationGuardResult
from .policies import GuardrailPolicies
from .risk_engine import RiskAssessment, RiskEngine
from .validation import (
    InputValidator,
    OutputValidator,
    RetrievalValidator,
)


@dataclass
class GuardrailPipelineResult:
    """Final result after running all guardrails."""
    input_valid: bool
    retrieval_valid: bool
    retrieval_issues: List[str]
    confidence_score: ConfidenceScore
    risk_assessment: RiskAssessment
    should_route_to_audit: bool
    generation_guard_result: Optional[GenerationGuardResult]
    output_valid: bool
    output_issues: List[str]
    final_safe_output: str


class GuardrailRunner:
    """Orchestrates all guardrails in sequence."""

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
        """Executes all guardrails in order."""
        # 1. Input Validation
        input_valid = self.input_validator.validate(query)
        if not input_valid:
            return GuardrailPipelineResult(
                input_valid=False,
                retrieval_valid=False,
                retrieval_issues=[],
                confidence_score=None,
                risk_assessment=None,
                generation_guard_result=None,
                output_valid=False,
                output_issues=["INVALID_INPUT_QUERY"],
                final_safe_output="Invalid query. Please rephrase.",
                should_route_to_audit=False,
            )

        # 2. Retrieval Validation
        retrieval_valid, retrieval_issues = self.retrieval_validator.validate(
            retrieval_results
        )
        if not retrieval_valid and self.policies.min_retrieval_confidence > 0.3:
            return GuardrailPipelineResult(
                input_valid=True,
                retrieval_valid=False,
                retrieval_issues=retrieval_issues,
                confidence_score=None,
                risk_assessment=None,
                generation_guard_result=None,
                output_valid=False,
                output_issues=["RETRIEVAL_VALIDATION_FAILED"] + retrieval_issues,
                final_safe_output="Insufficient or unreliable evidence found.",
                should_route_to_audit=True,
            )

        # 3. Confidence Scoring
        confidence = self.confidence_scorer.compute(
            retrieval_results, verification_results
        )

        # 4. Risk Assessment
        risk = self.risk_engine.assess(
            retrieval_results, verification_results, confidence
        )

        # 5. Generation Guard (if LLM output is provided)
        gen_guard_result = None
        if raw_llm_output:
            gen_guard_result = self.generation_guard.guard(
                raw_llm_output, verification_results
            )

        # 6. Output Validation
        output_valid = True
        output_issues = []
        final_output = ""

        if raw_llm_output:
            if gen_guard_result:
                final_output = gen_guard_result.sanitized_text
                if not gen_guard_result.is_safe:
                    output_valid = False
                    output_issues.extend(gen_guard_result.issues)
            else:
                final_output = raw_llm_output

            output_valid, val_issues = self.output_validator.validate(
                final_output,
                verification_results,
                confidence.overall,
            )
            output_issues.extend(val_issues)
        else:
            final_output = "No answer generated due to risk constraints."
            output_valid = False
            output_issues.append("NO_LLM_OUTPUT")

        # Determine if we should route to audit
        should_route_to_audit = (
            risk.recommended_action in ["HUMAN_REVIEW", "BLOCK"]
            or not output_valid
            or not retrieval_valid
        )

        return GuardrailPipelineResult(
            input_valid=input_valid,
            retrieval_valid=retrieval_valid,
            retrieval_issues=retrieval_issues,
            confidence_score=confidence,
            risk_assessment=risk,
            generation_guard_result=gen_guard_result,
            output_valid=output_valid,
            output_issues=output_issues,
            final_safe_output=final_output,
            should_route_to_audit=should_route_to_audit,
        )