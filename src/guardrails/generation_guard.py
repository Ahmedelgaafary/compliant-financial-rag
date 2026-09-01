# src/guardrails/generation_guard.py
import re
from dataclasses import dataclass
from typing import List, Set, Tuple

from src.guardrails.policies import GuardrailPolicies
from src.verification.models import VerificationResult, VerificationStatus


@dataclass
class GenerationGuardResult:
    """Result of the generation guard check."""
    is_safe: bool
    sanitized_text: str
    issues: List[str]
    flagged_claims: List[str]


class GenerationGuard:
    """
    Intercepts LLM output during generation to prevent hallucinated facts.
    Checks for:
    - Unsupported numeric claims (not found in verification results)
    - Unsupported citations (e.g., page numbers not in evidence)
    - Unsupported entity mentions
    """

    def __init__(self, policies: GuardrailPolicies):
        self.policies = policies
        # Patterns to detect potential financial numbers in text.
        #
        # The integer portion is matched as EITHER a comma-grouped number
        # (e.g. "1,234,567") OR a plain digit run of any length (e.g.
        # "2025", "42", "1000"). The previous pattern capped the
        # non-comma-grouped case at 3 digits (\d{1,3}), which meant any
        # 4+ digit number without thousands separators (a year, a page
        # count, "1000") was split into multiple bogus matches - e.g.
        # "2025" matched as "202" and then "5" separately.
        self.numeric_pattern = re.compile(
            r'(?P<dollar>\$)?\s*'
            r'(?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)'
            r'\s*(?P<unit>billion|million|B|M|bn|mn|%)?',
            re.IGNORECASE
        )
        # Updated to catch "Page 99", "(Page 99)", and "[citation:99]"
        self.citation_pattern = re.compile(
            r'(?:Page\s*(\d+)|\(Page\s*(\d+)\)|\[citation:(\d+)\])',
            re.IGNORECASE
        )

    def _extract_verified_numeric_claims(
        self,
        verification_results: List[VerificationResult]
    ) -> Set[Tuple[float, str]]:
        """Extract normalized numeric values and units from verified claims."""
        verified = set()
        for v in verification_results:
            if v.status == VerificationStatus.VERIFIED and v.claim_type == "NUMERIC":
                if hasattr(v, 'normalized_value') and hasattr(v, 'unit'):
                    verified.add((v.normalized_value, v.unit))
        return verified

    def _extract_verified_pages(
        self,
        verification_results: List[VerificationResult]
    ) -> Set[int]:
        """Extract page numbers from verified evidence chunks."""
        pages = set()
        for v in verification_results:
            if v.status == VerificationStatus.VERIFIED and hasattr(v, 'evidence_chunk_id'):
                # In real implementation, fetch chunk metadata
                if hasattr(v, 'page_number') and v.page_number:
                    pages.add(v.page_number)
        return pages

    def guard(
        self,
        raw_output: str,
        verification_results: List[VerificationResult]
    ) -> GenerationGuardResult:
        """
        Analyzes the raw LLM output against verified evidence.
        Returns sanitized text and a safety flag.
        """
        issues = []
        flagged_claims = []
        sanitized_text = raw_output

        # 1. Check numeric claims
        verified_numbers = self._extract_verified_numeric_claims(verification_results)
        numeric_matches = self.numeric_pattern.finditer(raw_output)

        for match in numeric_matches:
            dollar = match.group('dollar')
            unit = match.group('unit') or ""
            number_str = match.group('value').replace(',', '')

            if not number_str:
                continue

            # Skip bare numbers with no financial context. A number is
            # only treated as a financial claim if it's introduced by a
            # currency symbol or followed by a magnitude/percent unit -
            # otherwise it's incidental (a year, a page reference, an
            # ordinary count) and shouldn't be flagged as unverified.
            if not dollar and not unit:
                continue

            try:
                value = float(number_str)
            except ValueError:
                continue

            # Check if this (value, unit) exists in verified claims
            is_verified = any(
                abs(value - verified_val) / max(verified_val, 1) < 0.001
                and unit.lower() == verified_unit.lower()
                for verified_val, verified_unit in verified_numbers
            )

            if not is_verified and verified_numbers:
                issues.append(f"UNVERIFIED_NUMERIC_CLAIM: {match.group(0)}")
                flagged_claims.append(match.group(0))

        # 2. Check citations (hallucinated page numbers)
        verified_pages = self._extract_verified_pages(verification_results)
        citation_matches = self.citation_pattern.finditer(raw_output)

        for match in citation_matches:
            # Get the first non‑None group (handles any of the three alternatives)
            page_num = next((g for g in match.groups() if g is not None), None)
            if page_num and int(page_num) not in verified_pages:
                issues.append(f"HALLUCINATED_CITATION: Page {page_num}")
                flagged_claims.append(f"Page {page_num}")

        # 3. Apply policy to determine if safe
        is_safe = len(issues) == 0

        # If policy allows unsupported claims, we still flag but don't block
        if self.policies.allow_unsupported_claims:
            is_safe = True

        # If too many issues, force a fallback
        if len(issues) > 5:
            is_safe = False
            sanitized_text = (
                "I detected multiple unsupported statements. "
                "Please consult the original financial documents directly."
            )

        return GenerationGuardResult(
            is_safe=is_safe,
            sanitized_text=sanitized_text,
            issues=issues,
            flagged_claims=flagged_claims,
        )