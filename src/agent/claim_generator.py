import re
from typing import List, Tuple

from src.llm.client import LLMClient
from src.retrieval.models import RetrievalResult
from src.verification.models import Claim, ClaimType


class ClaimGenerator:
    """Generates candidate claims using an LLM, then parses them into Claim objects."""

    def __init__(self, llm: LLMClient = None):
        self.llm = llm or LLMClient()

    def generate(self, query: str, evidence: List[RetrievalResult]) -> Tuple[str, List[Claim]]:
        evidence_text = "\n".join([f"- {r.text}" for r in evidence])
        prompt = (
            f"Based on the following evidence, generate one concise financial claim "
            f"(e.g., 'revenue = $42.8B').\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Query: {query}\n"
            f"Claim:"
        )
        raw_output = self.llm.generate(prompt)

        match = re.search(
            r"(\w+)\s*=\s*\$?(\d+\.?\d*)\s*([BMK]?)",
            raw_output,
            re.IGNORECASE,
        )
        claims: List[Claim] = []
        if match:
            subject = match.group(1)
            value = f"${match.group(2)}"
            unit_map = {"B": "billion", "M": "million", "K": "thousand"}
            unit = unit_map.get(match.group(3).upper(), "") or None
            claim = Claim(
                claim_id=f"claim_{len(claims)+1}",
                claim_type=ClaimType.NUMERIC,
                subject=subject,
                value=value,
                unit=unit,
                period=None,
                source_chunk_id=evidence[0].chunk_id if evidence else None,
            )
            claims.append(claim)

        return raw_output, claims