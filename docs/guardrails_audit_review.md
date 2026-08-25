

# Salem's Guardrails & Audit Review

**Date**: 2026-08-25  
**Status**: Review only – no implementation changes made  
**Scope**: Inspect existing guardrails and audit modules, identify implemented vs missing functionality, integration gaps, and recommended implementation order.

---

## 1. Guardrails Status

### `src/guardrails/confidence.py`

- **Current responsibility**: Compute an aggregate confidence score from retrieval scores and verification outcomes.
- **Implemented**:
  - `ConfidenceScore` dataclass with `overall`, `retrieval_confidence`, `verification_confidence`, `evidence_sufficiency`, `contradictions_penalty`.
  - `ConfidenceScorer` with configurable weights (retrieval vs verification).
  - Basic formula: weighted average of retrieval scores and verification ratio, minus penalty for contradictions.
- **Missing**:
  - Tests (no `test_confidence.py`).
  - Use of `VerificationStatus` enum for status comparison – currently uses string `"VERIFIED"`; may cause type mismatch if enum is used elsewhere.
- **Tests available**: None.
- **Recommended changes**:
  - Add unit tests covering high/low confidence scenarios.
  - Normalize status comparison to use `VerificationStatus.VERIFIED` (or handle both).
  - Consider exposing per‑claim confidence breakdown.

---

### `src/guardrails/policies.py`

- **Current responsibility**: Central configuration for guardrail behaviour (thresholds, risk increments, allowed query types, policy flags).
- **Implemented**:
  - All fields: `max_risk_score`, confidence thresholds, risk thresholds, risk increment values, evidence minimums, allowed query types, etc.
  - Methods `is_query_allowed(query_type)` and `get_risk_level(risk_score)`.
- **Missing**:
  - Validation of field values (e.g., negative risk increments, contradictory thresholds).
  - Tests for policy methods.
- **Tests available**: None.
- **Recommended changes**:
  - Add `__post_init__` validation.
  - Add unit tests for `get_risk_level` boundary values and `is_query_allowed`.

---

### `src/guardrails/risk_engine.py`

- **Current responsibility**: Convert verification results + confidence into a deterministic `RiskAssessment`.
- **Implemented**:
  - Full deterministic scoring: rejected/inconclusive counts, contradictions, low confidence, missing provenance, insufficient evidence, numeric mismatch.
  - `RiskAssessment` dataclass with `risk_score`, `risk_level`, `triggers`, `recommended_action`.
  - Logic for `BLOCK`, `HUMAN_REVIEW`, `AUTO_ANSWER_WITH_DISCLAIMER`, `AUTO_ANSWER`.
- **Missing**:
  - None major; already well‑integrated with `policies` and `confidence`.
- **Tests available**: `tests/test_risk_engine.py` – covers all major categories (verified, inconclusive, rejected, numeric mismatch, contradictions, missing provenance, low confidence, no evidence, determinism).
- **Recommended changes**: None.

---

### `src/guardrails/validation.py`

- **Current responsibility**: Provide input, retrieval, and output validators.
- **Implemented**:
  - `InputValidator` – checks empty query, forbidden entities, length.
  - `RetrievalValidator` – checks for evidence presence, provenance completeness, sufficient evidence count, retrieval score threshold.
  - `OutputValidator` – placeholder with commented‑out logic; only checks low confidence and optional numeric mismatch block.
- **Missing**:
  - Actual output validation logic (currently only comments and placeholder checks).
  - Tests for any of the validators.
- **Tests available**: None.
- **Recommended changes**:
  - Implement real output validation (unsupported numeric claims, hallucinated citations, entity checks).
  - Add unit tests for each validator.

---

### `src/guardrails/generation_guard.py`

- **Current responsibility**: Intercepts LLM output to prevent unsupported claims or hallucinations.
- **Implemented**:
  - `GenerationGuardResult` dataclass (`is_safe`, `sanitized_text`, `issues`, `flagged_claims`).
  - `GenerationGuard` class with regex patterns for numeric values and page citations.
  - Checks numeric claims against verified values (assuming certain attributes exist).
- **Missing**:
  - Assumes `VerificationResult` has `claim_type`, `normalized_value`, `page_number`, etc. – these are **not** present in the current model.
  - No tests.
- **Tests available**: None.
- **Recommended changes**:
  - Adapt `GenerationGuard` to work with actual `VerificationResult` model (or extend model).
  - Add tests for safe/unsafe outputs.

---

### `src/guardrails/runner.py`

- **Current responsibility**: Orchestrate all guardrails in sequence, produce `GuardrailPipelineResult`.
- **Implemented**:
  - Full pipeline: input validation, retrieval validation, confidence scoring, risk assessment, generation guard, output validation.
  - `GuardrailPipelineResult` dataclass with all relevant fields.
  - `should_route_to_audit` decision based on risk, output validity, retrieval validity.
- **Missing**:
  - Dedicated unit tests (only indirect via `test_agent_pipeline.py`).
- **Tests available**: Partially via `tests/test_agent_pipeline.py` (mocked runner in some tests).
- **Recommended changes**:
  - Add direct unit tests for `run_full_pipeline` with mocked sub‑components.

---

### `src/guardrails/__init__.py`

- **Current responsibility**: Expose public interfaces.
- **Implemented**: Exports `ConfidenceScorer`, `GuardrailPolicies`, `RiskEngine`, and validators.
- **Missing**: None.
- **Tests available**: N/A.
- **Recommended changes**: None.

---

## 2. Audit Status

### `src/audit/models.py`

- **Current responsibility**: Define audit schemas, statuses, and records.
- **Implemented**:
  - `AuditStatus` enum (PENDING, IN_REVIEW, RESOLVED, ESCALATED, REJECTED).
  - `ReviewDecision` enum (APPROVED, REJECTED, NEEDS_MORE_INFO, ESCALATE).
  - `AuditRecord` dataclass with fields for claim, evidence, verification, risk, and review.
- **Missing**:
  - Some duplicate fields (`document_id`, `document_sha256`, `page_number` appear twice in the definition) – minor.
  - No validation of required fields.
- **Tests available**: None.
- **Recommended changes**:
  - Clean up duplicate fields.
  - Add validation or use `dataclasses` with proper defaults.

---

### `src/audit/queue.py`

- **Current responsibility**: Manage the audit queue (in‑memory).
- **Implemented**:
  - `enqueue(record)` – assigns ID, sets status PENDING, stores.
  - `get_pending()` – returns all PENDING records.
  - `get_by_id(audit_id)` – retrieves a specific record.
  - `start_review(audit_id, reviewer)` – moves record to IN_REVIEW.
  - `resolve(audit_id, decision, notes)` – marks as RESOLVED, stores decision.
  - `get_all()` and `size()`.
- **Missing**:
  - Tests.
  - Persistence (only in‑memory; may be okay for development but not production).
  - No rejection of invalid state transitions (e.g., resolving a non‑pending record) – though `start_review` checks status.
- **Tests available**: None.
- **Recommended changes**:
  - Add unit tests for enqueue, get_pending, resolve, invalid transitions.
  - Consider implementing a persistent queue (Redis, DB) for production.
  - Enforce state transition rules (e.g., cannot resolve before start_review).

---

### `src/audit/router.py`

- **Current responsibility**: Route the query to AUTO_ANSWER, HUMAN_REVIEW, or BLOCK based on risk and verification.
- **Implemented**:
  - `RoutingAction` enum.
  - `RoutingDecision` dataclass with action, reason, should_create_audit_record, audit_priority.
  - `AuditRouter.route()` – logic based on risk level, verification statuses, triggers.
- **Missing**:
  - Tests.
  - Potential issue: uses string comparison for verification status (`v.status == "INCONCLUSIVE"`) – should use enum.
- **Tests available**: None.
- **Recommended changes**:
  - Add unit tests.
  - Align with `VerificationStatus` enum.

---

### `src/audit/decisions.py`

- **Current responsibility**: Analyze audit records and suggest a review recommendation.
- **Implemented**:
  - `ReviewRecommendation` enum.
  - `DecisionResult` dataclass.
  - `DecisionEngine.analyze()` – rule‑based logic for approve, escalate, needs more info, reject.
- **Missing**:
  - Tests.
  - No configuration file (`decision_rules.json`) used – the example exists but not loaded.
- **Tests available**: None.
- **Recommended changes**:
  - Add tests.
  - Optionally load decision rules from JSON to make configurable.

---

### `src/audit/audit_log.py`

- **Current responsibility**: Persist audit records to a JSON file using marshmallow serialization.
- **Implemented**:
  - `AuditRecordSchema` with fields for all record attributes.
  - `AuditLogger` – methods: `log(record)`, `get_all()`, `get_by_audit_id()`.
- **Missing**:
  - Tests.
  - JSON file location is hardcoded to `audit_logs/audit_records.json`.
  - No rotation or error handling.
- **Tests available**: None.
- **Recommended changes**:
  - Add unit tests.
  - Make log directory configurable.
  - Consider using a database instead of JSON for production.

---

### `src/audit/review_service.py`

- **Current responsibility**: Orchestrate the entire audit workflow.
- **Implemented**:
  - `ReviewOutcome` dataclass.
  - `ReviewService.initiate_review()` – routes, creates audit record if needed, enqueues, gets recommendation, logs.
  - `get_pending_reviews()` and `submit_review_decision()`.
- **Missing**:
  - Tests.
  - The `AuditRecord` construction in `initiate_review` may be missing required fields (e.g., `risk_assessment`, `created_at`) – this was a bug we encountered earlier; it may have been fixed, but need to verify.
  - In `submit_review_decision`, it calls `logger.log(record)` again – might create duplicates (depending on logger implementation).
- **Tests available**: Indirect via `test_agent_pipeline.py` (mocked).
- **Recommended changes**:
  - Add dedicated unit tests for `ReviewService`.
  - Fix potential duplicate logging.
  - Ensure correct `AuditRecord` construction.

---

## 3. Integration Gaps

### Expected Flow

### Current Implementation Gaps

| Step | Status | Gap |
|------|--------|-----|
| **Verification → Guardrails** | ✅ | `GuardrailRunner` receives `VerificationResult` list. Works. |
| **Guardrails → Risk Decision** | ✅ | `RiskEngine` produces `RiskAssessment`. Works. |
| **Risk Decision → Audit Queue** | ⚠️ | `ReviewService` calls `AuditRouter` and `AuditQueue`, but `AuditRecord` construction may miss required fields (like `risk_assessment`), and the router uses string comparison for verification status. |
| **Audit Queue → Human Decision** | ⚠️ | `AuditQueue` supports `start_review` and `resolve`, but `ReviewService.submit_review_decision` may double‑log and there are no tests for these flows. |
| **Human Decision → Audit Log** | ⚠️ | `AuditLogger.log(record)` writes to JSON, but the schema may not match current `AuditRecord` (e.g., missing `risk_assessment` field). Also, re‑logging a record may append duplicate entries. |

### Specific Issues

1. **Status enum mismatch** across modules: `VerificationResult.status` is a `VerificationStatus` enum, but `confidence.py`, `router.py`, and some parts of `generation_guard.py` compare with string literals (`"VERIFIED"`, `"INCONCLUSIVE"`). This will cause errors.

2. **`AuditRecord` construction incomplete**: In `review_service.py`, the `AuditRecord` created in `initiate_review` passes `confidence_score`, `risk_score`, etc., but the model now also requires `risk_assessment` and `created_at` – likely missing, causing runtime errors.

3. **No tests for audit modules**: Only `test_risk_engine.py` and `test_agent_pipeline.py` exist; no direct tests for queue, router, decisions, audit_log, review_service.

4. **`GenerationGuard` not aligned with model**: It expects `VerificationResult` to have `normalized_value`, `unit`, `page_number`, etc., which are not present.

5. **Potential duplicate logging**: In `submit_review_decision`, it calls `logger.log(record)` again; if the logger appends without dedup, records may appear twice.

---

## 4. Recommended Implementation Order

1. **Fix status enum usage** across `confidence.py`, `generation_guard.py`, `router.py` – replace string comparisons with `VerificationStatus` enum.
2. **Fix `AuditRecord` construction** in `review_service.py` – add missing `risk_assessment` and `created_at` fields.
3. **Align `GenerationGuard` with actual `VerificationResult` model** – either extend the model or adjust guard logic.
4. **Add unit tests for audit modules**:
   - `test_audit_queue.py`
   - `test_audit_router.py`
   - `test_audit_decisions.py`
   - `test_audit_log.py`
   - `test_review_service.py`
5. **Add unit tests for validators** (`test_validation.py`) and confidence (`test_confidence.py`).
6. **Improve output validator** – implement real checks for unsupported claims and hallucinated citations.
7. **Consider persistence for audit queue** – replace in‑memory with Redis/DB for production.
8. **Integrate `decision_rules.json`** into `DecisionEngine` to make recommendations configurable.
9. **Final integration test** – end‑to‑end test with real components (or thorough mocks) verifying the full flow.

---

## Verification Summary

- **Existing tests executed**: `pytest tests/test_risk_engine.py tests/test_agent_pipeline.py` – all passed (19 tests).
- **Ruff check**: All checks passed.
- **No duplicate files created** – only this report.

**Status**: Review complete. No implementation changes made. Awaiting discussion with Ahmed before proceeding with fixes.
