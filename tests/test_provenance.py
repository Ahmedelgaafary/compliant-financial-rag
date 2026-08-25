from src.verification.provenance import EvidenceProvenance


def test_valid_provenance() -> None:
    provenance = EvidenceProvenance(
        document_id="annual-report-2025",
        document_hash="a" * 64,
        chunk_id="annual-report-2025-047",
        page_number=47,
        section="Consolidated Statements",
    )

    assert provenance.is_valid()


def test_empty_document_id_is_invalid() -> None:
    provenance = EvidenceProvenance(
        document_id="",
        document_hash="a" * 64,
        chunk_id="chunk-001",
        page_number=1,
    )

    assert not provenance.is_valid()


def test_empty_document_hash_is_invalid() -> None:
    provenance = EvidenceProvenance(
        document_id="report",
        document_hash="",
        chunk_id="chunk-001",
        page_number=1,
    )

    assert not provenance.is_valid()


def test_empty_chunk_id_is_invalid() -> None:
    provenance = EvidenceProvenance(
        document_id="report",
        document_hash="a" * 64,
        chunk_id="",
        page_number=1,
    )

    assert not provenance.is_valid()


def test_invalid_page_number() -> None:
    provenance = EvidenceProvenance(
        document_id="report",
        document_hash="a" * 64,
        chunk_id="chunk-001",
        page_number=0,
    )

    assert not provenance.is_valid()


def test_provenance_is_immutable() -> None:
    provenance = EvidenceProvenance(
        document_id="report",
        document_hash="a" * 64,
        chunk_id="chunk-001",
        page_number=1,
    )

    try:
        provenance.page_number = 2
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "EvidenceProvenance should be immutable."
        )