from src.verification.claim_extractor import (
    ClaimExtractor,
    ExtractionResult,
)
from src.verification.models import ClaimType


def test_extract_numeric_financial_claim() -> None:
    extractor = ClaimExtractor()

    result = extractor.extract(
        text="Revenue was $42.8 billion in 2025.",
        claim_id="claim-001",
        source_chunk_id="chunk-047",
    )

    assert isinstance(result, ExtractionResult)
    assert result.claim is not None

    assert result.claim.claim_id == "claim-001"
    assert result.claim.claim_type == ClaimType.NUMERIC
    assert result.claim.subject == "Revenue"
    assert result.claim.value == "$ 42.8"
    assert result.claim.unit == "$ billion"
    assert result.claim.period == "2025"
    assert result.claim.source_chunk_id == "chunk-047"


def test_extract_euro_claim() -> None:
    extractor = ClaimExtractor()

    result = extractor.extract(
        text="Net income was €3.2 billion in 2025.",
        claim_id="claim-002",
    )

    assert result.claim is not None
    assert result.claim.claim_type == ClaimType.NUMERIC
    assert result.claim.subject == "Net income"
    assert result.claim.value == "€ 3.2"
    assert result.claim.unit == "€ billion"
    assert result.claim.period == "2025"


def test_extract_currency_after_number() -> None:
    extractor = ClaimExtractor()

    result = extractor.extract(
        text="Revenue was 42.8 billion USD in 2025.",
        claim_id="claim-003",
    )

    assert result.claim is not None
    assert result.claim.value == "USD 42.8"
    assert result.claim.unit == "USD billion"
    assert result.claim.period == "2025"


def test_normalize_unit_aliases() -> None:
    extractor = ClaimExtractor()

    result = extractor.extract(
        text="Revenue was $42.8bn in 2025.",
        claim_id="claim-004",
    )

    assert result.claim is not None
    assert result.claim.unit == "$ billion"


def test_extract_million() -> None:
    extractor = ClaimExtractor()

    result = extractor.extract(
        text="Operating expenses were $15.4 million in 2025.",
        claim_id="claim-005",
    )

    assert result.claim is not None
    assert result.claim.subject == "Operating expenses"
    assert result.claim.value == "$ 15.4"
    assert result.claim.unit == "$ million"


def test_extract_without_period() -> None:
    extractor = ClaimExtractor()

    result = extractor.extract(
        text="Revenue was $42.8 billion.",
        claim_id="claim-006",
    )

    assert result.claim is not None
    assert result.claim.period is None


def test_extract_without_currency() -> None:
    extractor = ClaimExtractor()

    result = extractor.extract(
        text="Revenue was 42.8 billion in 2025.",
        claim_id="claim-007",
    )

    assert result.claim is not None
    assert result.claim.value == "42.8"
    assert result.claim.unit == "billion"
    assert result.claim.period == "2025"


def test_empty_text_returns_no_claim() -> None:
    extractor = ClaimExtractor()

    result = extractor.extract(
        text="",
        claim_id="claim-008",
    )

    assert result.claim is None
    assert result.matched_text is None


def test_unsupported_text_returns_no_claim() -> None:
    extractor = ClaimExtractor()

    result = extractor.extract(
        text="The company announced a new strategy.",
        claim_id="claim-009",
    )

    assert result.claim is None
    assert result.matched_text is None


def test_source_chunk_is_preserved() -> None:
    extractor = ClaimExtractor()

    result = extractor.extract(
        text="Revenue was $100 million in 2025.",
        claim_id="claim-010",
        source_chunk_id="chunk-123",
    )

    assert result.claim is not None
    assert result.claim.source_chunk_id == "chunk-123"