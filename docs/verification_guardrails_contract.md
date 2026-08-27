# Verification → Guardrails Contract

**Status**: Stable  
**Last Updated**: 2026-08-27  
**Owner**: Salem (Guardrails/Audit) and Ahmed (Verification)

## Purpose

This document defines the stable interface between the deterministic verification layer and the guardrails/risk engine layer. The verification layer answers: *"Is this claim supported by the evidence?"*  
The guardrails layer answers: *"Is it safe to automatically provide this answer?"*

## Inputs to Guardrails

### `VerificationResult`

The guardrails layer receives a list of `VerificationResult` objects. Each object MUST contain:

| Field | Type | Description |
|-------|------|-------------|
| `claim_id` | `str` | Unique identifier of the claim being verified. |
| `status` | `VerificationStatus` | One of `VERIFIED`, `REJECTED`, `INCONCLUSIVE`. |
| `reason` | `str` | Machine‑readable reason (e.g., `numeric_match`, `numeric_mismatch`, `evidence_missing`, `evidence_contradicts`). |
| `confidence` | `float` | Confidence in the verification result (0.0–1.0). |
| `evidence_chunk_id` | `str` (optional) | Chunk identifier that served as evidence. If missing, the verification is not auditable. |

### `RetrievalResult`

The guardrails layer also receives a list of `RetrievalResult` objects to access evidence and provenance. Required fields:

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | `str` | Unique chunk identifier. |
| `document_id` | `str` | Source document identifier. |
| `document_sha256` | `str` | Hash of the document (integrity). |
| `page_number` | `int` | Page in the source document. |
| `text` | `str` | The evidence text. |

## Contract Rules

1. **Guardrails MUST NOT access verification internals** (e.g., `ClaimVerifier`, `NumericVerifier`).  
   They consume only `VerificationResult` and `RetrievalResult`.

2. **Provenance is preserved** – `RetrievalResult` always carries the document hash, page, and chunk ID.  
   If any of these are missing, the case is considered non‑auditable and should block automatic answering.

3. **Failure/inconclusive states are explicit** –  
   - `INCONCLUSIVE` (e.g., `evidence_missing`) → requires human review.  
   - `REJECTED` with `numeric_mismatch` → high risk, likely block or human review.  
   - `REJECTED` with `evidence_contradicts` → unresolved contradiction, block auto‑answer.

4. **Contradiction signaling** – any `VerificationResult` with `reason == "evidence_contradicts"` MUST result in blocking automatic output and routing to human review.

5. **Confidence threshold** – if the overall confidence (computed by `ConfidenceScorer`) is below the policy minimum, automatic answering is prohibited.

## Example Flow

```text
Claim + Evidence → ClaimVerifier → VerificationResult
    → GuardrailRunner (consumes VerificationResult + RetrievalResult)
    → RiskEngine → RiskAssessment → route to auto‑answer or audit queue.