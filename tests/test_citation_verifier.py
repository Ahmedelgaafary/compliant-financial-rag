from src.verification.citation_verifier import (
    CitationVerificationResult,
    CitationVerifier,
)
from src.verification.provenance import EvidenceProvenance


def make_provenance() -> EvidenceProvenance:
    return EvidenceProvenance(
        document_id="doc-001",
        document_hash="abc123",
        chunk_id="chunk-001",
        page_number=5,
        section="Revenue",
    )


def test_valid_provenance() -> None:
    verifier = CitationVerifier()

    result = verifier.verify(make_provenance())

    assert isinstance(result, CitationVerificationResult)
    assert result.valid is True
    assert result.reason == "Citation provenance is valid."


def test_missing_provenance_is_invalid() -> None:
    verifier = CitationVerifier()

    result = verifier.verify(None)

    assert result.valid is False
    assert result.reason == "Citation provenance is missing."


def test_invalid_provenance_is_rejected() -> None:
    verifier = CitationVerifier()

    provenance = EvidenceProvenance(
        document_id="",
        document_hash="abc123",
        chunk_id="chunk-001",
        page_number=5,
    )

    result = verifier.verify(provenance)

    assert result.valid is False
    assert result.reason == "Citation provenance is invalid."


def test_invalid_page_number_is_rejected() -> None:
    verifier = CitationVerifier()

    provenance = EvidenceProvenance(
        document_id="doc-001",
        document_hash="abc123",
        chunk_id="chunk-001",
        page_number=0,
    )

    result = verifier.verify(provenance)

    assert result.valid is False
    assert result.reason == "Citation provenance is invalid."


def test_document_id_mismatch_is_rejected() -> None:
    verifier = CitationVerifier()

    result = verifier.verify(
        make_provenance(),
        expected_document_id="doc-999",
    )

    assert result.valid is False
    assert (
        result.reason
        == "Citation document ID does not match expected document."
    )


def test_chunk_id_mismatch_is_rejected() -> None:
    verifier = CitationVerifier()

    result = verifier.verify(
        make_provenance(),
        expected_chunk_id="chunk-999",
    )

    assert result.valid is False
    assert (
        result.reason
        == "Citation chunk ID does not match expected chunk."
    )


def test_matching_document_id_is_accepted() -> None:
    verifier = CitationVerifier()

    result = verifier.verify(
        make_provenance(),
        expected_document_id="doc-001",
    )

    assert result.valid is True


def test_matching_chunk_id_is_accepted() -> None:
    verifier = CitationVerifier()

    result = verifier.verify(
        make_provenance(),
        expected_chunk_id="chunk-001",
    )

    assert result.valid is True


def test_matching_document_and_chunk_are_accepted() -> None:
    verifier = CitationVerifier()

    result = verifier.verify(
        make_provenance(),
        expected_document_id="doc-001",
        expected_chunk_id="chunk-001",
    )

    assert result.valid is True


def test_optional_section_does_not_affect_validation() -> None:
    verifier = CitationVerifier()

    provenance = EvidenceProvenance(
        document_id="doc-001",
        document_hash="abc123",
        chunk_id="chunk-001",
        page_number=5,
        section=None,
    )

    result = verifier.verify(provenance)

    assert result.valid is True