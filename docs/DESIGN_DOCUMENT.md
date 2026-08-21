# Compliant Financial Document RAG & Audit Agent

## 1. Project Overview

### Project Name

**Compliant Financial Document RAG & Audit Agent**

### Objective

Build a production-oriented Financial AI system capable of answering questions over complex financial documents such as:

* SEC 10-K filings
* Annual reports
* Quarterly earnings reports
* Financial statements
* Bank statements
* Regulatory documents

The system must prioritize:

1. **Accuracy**
2. **Traceability**
3. **Deterministic verification**
4. **Hallucination prevention**
5. **Auditability**
6. **Human oversight for high-risk cases**

The system is explicitly designed to avoid the common architecture:

```text
User
  ↓
LLM
  ↓
Answer
```

Instead, our architecture is:

```text
User
  ↓
Query Analysis
  ↓
Hybrid Retrieval
  ↓
Evidence
  ↓
Claim Extraction
  ↓
Deterministic Verification
  ↓
Guardrails
  ↓
Risk Assessment
  ↓
 ┌───────────────┬────────────────┐
 ▼               ▼                ▼
Verified      Inconclusive     High Risk
 ▼               ▼                ▼
Answer        Explanation      Human Audit
```

---

# 2. Key Differentiator

The primary differentiator is:

> **The LLM is not the final authority for financial facts.**

The LLM may:

* interpret a question
* summarize retrieved evidence
* explain verified results
* generate a human-readable response

But deterministic components must control:

* numerical claims
* dates and reporting periods
* document identity
* evidence provenance
* claim/evidence consistency
* contradictions
* high-risk decisions

This creates a separation between:

```text
Generation
```

and:

```text
Verification
```

---

# 3. Core Requirements

## 3.1 Document Processing

The system must support:

* PDF ingestion
* text extraction
* page preservation
* document hashing
* section detection
* chunking
* metadata preservation

Every chunk should remain traceable to its source document.

---

## 3.2 Hybrid Retrieval

The retrieval layer must combine:

### BM25

For:

* exact terminology
* financial terms
* company names
* accounting terminology
* numbers
* identifiers

### Vector Search

For:

* semantic similarity
* paraphrased questions
* concept matching
* contextual financial questions

### Hybrid Retrieval

Combine BM25 and vector rankings using:

**Reciprocal Rank Fusion (RRF)**.

Architecture:

```text
                    Query
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
        BM25                 Vector Store
          │                       │
          ▼                       ▼
    Lexical Ranking         Semantic Ranking
          │                       │
          └───────────┬───────────┘
                      ▼
                   RRF Fusion
                      │
                      ▼
                 Top-K Evidence
```

---

# 4. Current Retrieval Implementation

The current retrieval system contains:

```text
src/retrieval/
├── __init__.py
├── models.py
├── bm25.py
├── vector_store.py
├── hybrid.py
├── evaluation.py
└── benchmark.py
```

### Vector Model

Current embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are normalized and similarity is calculated using vector dot product, equivalent to cosine similarity for normalized embeddings.

---

# 5. Retrieval Provenance

Every retrieval result must preserve:

```text
chunk_id
document_id
text
score
page_number
section
document_sha256
retrieval_method
```

Example:

```text
RetrievalResult
├── chunk_id
├── document_id
├── text
├── score
├── page_number
├── section
├── document_sha256
└── retrieval_method
```

This allows an answer to be traced back to:

```text
Answer
  ↓
Claim
  ↓
Evidence Chunk
  ↓
Page
  ↓
Section
  ↓
Document
  ↓
SHA-256 Hash
```

---

# 6. Retrieval Evaluation

The project includes a labeled retrieval benchmark.

Current metrics:

* Recall@1
* Recall@3
* Mean Reciprocal Rank (MRR)

Current initial benchmark:

| Retriever   | Recall@1 | Recall@3 |   MRR |
| ----------- | -------: | -------: | ----: |
| BM25        |    0.750 |    1.000 | 0.875 |
| VectorStore |    1.000 |    1.000 | 1.000 |
| Hybrid RRF  |    1.000 |    1.000 | 1.000 |

### Important limitation

The current benchmark contains only four synthetic financial cases.

Therefore these results are an **initial engineering baseline**, not a production-quality statistical evaluation.

The benchmark should later be expanded using real financial documents and more diverse questions.

---

# 7. Deterministic Claim Verification

This is the central compliance component.

The verifier must operate independently from the LLM.

A generated financial claim should be transformed into a structured representation.

Example:

```text
"The company's revenue was $42.8 billion in 2025."
```

becomes:

```text
Claim
├── claim_id
├── claim_type = NUMERIC
├── subject = revenue
├── value = 42.8
├── unit = USD billion
├── period = 2025
└── source_chunk_id
```

---

# 8. Claim Types

Initial claim types:

```text
NUMERIC
DATE
ENTITY
TEXT
```

Future claim types may include:

```text
PERCENTAGE
CURRENCY
RATIO
COUNT
FINANCIAL_METRIC
COMPARISON
```

---

# 9. Verification Status

The verifier must return one of:

```text
VERIFIED
REJECTED
INCONCLUSIVE
```

### VERIFIED

Evidence supports the claim.

### REJECTED

Evidence contradicts the claim.

### INCONCLUSIVE

Available evidence is insufficient to make a deterministic decision.

The system must **not guess** when evidence is insufficient.

---

# 10. Verification Reasons

Machine-readable verification reasons include:

```text
NUMERIC_MATCH
NUMERIC_MISMATCH

PERIOD_MATCH
PERIOD_MISMATCH

ENTITY_MATCH
ENTITY_MISMATCH

EVIDENCE_MISSING
EVIDENCE_CONTRADICTS

UNSUPPORTED_CLAIM
```

These codes make audit decisions machine-readable and explainable.

---

# 11. Numeric Verification

Numeric claims are especially important in financial AI.

Example:

### Generated claim

```text
Revenue = $45.2 billion
```

### Evidence

```text
Revenue = $42.8 billion
```

The verifier must produce:

```text
status:
REJECTED

reason:
NUMERIC_MISMATCH
```

The LLM must not override this result.

---

# 12. Financial Numeric Normalization

The verifier should normalize equivalent representations.

Examples:

```text
$42.8 billion
42.8B
42,800 million
$42,800,000,000
```

These should be recognized as equivalent where currency and context agree.

The system must also distinguish:

```text
42.8 billion
```

from:

```text
42.8 million
```

and:

```text
42.8%
```

---

# 13. Period Verification

Financial values must be associated with reporting periods.

Examples:

```text
FY2025
2025
Q4 2025
Year ended December 31, 2025
```

The verifier should detect period mismatches.

Example:

```text
Claim:
Revenue in 2025 = $42.8B

Evidence:
Revenue in 2024 = $39.1B
```

Result:

```text
REJECTED
PERIOD_MISMATCH
```

---

# 14. Entity Verification

The system must verify that the evidence belongs to the requested entity.

Example:

```text
Question:
What was Apple's revenue?

Evidence:
Microsoft revenue = ...
```

The system must not treat this as valid evidence.

Entity verification should consider:

* company name
* issuer
* document identity
* document metadata
* filing identity

---

# 15. Provenance Verification

Every verified claim should be traceable to evidence.

Minimum provenance:

```text
document_id
document_sha256
page_number
section
chunk_id
```

The final response should therefore be capable of saying:

```text
Verified claim
    ↓
Evidence chunk
    ↓
Page 42
    ↓
Financial Statements
    ↓
Document SHA-256
```

---

# 16. Contradiction Detection

The system must detect when multiple retrieved pieces of evidence disagree.

Example:

```text
Evidence A:
Revenue = $42.8B

Evidence B:
Revenue = $45.1B
```

The system should not arbitrarily select one.

Instead:

```text
STATUS:
INCONCLUSIVE

REASON:
EVIDENCE_CONTRADICTS
```

The case can then be routed to human review.

---

# 17. Guardrails

Guardrails operate before and after LLM generation.

## Input Guardrails

Detect:

* unsupported requests
* malformed queries
* requests outside document scope
* attempts to bypass verification

## Retrieval Guardrails

Detect:

* insufficient evidence
* low retrieval confidence
* missing provenance
* irrelevant evidence

## Generation Guardrails

Prevent the LLM from:

* inventing unsupported numbers
* inventing citations
* changing verified values
* presenting unverified claims as facts

## Output Guardrails

Verify the generated answer before returning it to the user.

---

# 18. Human-in-the-Loop Audit Routing

High-risk or uncertain cases should be routed to human auditors.

Example:

```text
Claim
  ↓
Verifier
  ↓
INCONCLUSIVE
  ↓
Risk Engine
  ↓
HIGH RISK
  ↓
Human Audit Queue
```

Possible audit triggers:

```text
NUMERIC_MISMATCH
EVIDENCE_CONTRADICTS
EVIDENCE_MISSING
ENTITY_MISMATCH
PERIOD_MISMATCH
LOW_RETRIEVAL_CONFIDENCE
MULTIPLE_CONFLICTING_SOURCES
```

---

# 19. Risk Scoring

The system should assign a risk score to verification cases.

Example factors:

```text
retrieval confidence
+
claim type
+
numeric mismatch
+
period mismatch
+
entity mismatch
+
contradictory evidence
+
missing evidence
```

Possible risk levels:

```text
LOW
MEDIUM
HIGH
CRITICAL
```

High-risk cases should be routed to human review.

---

# 20. Audit Record

Each audited event should preserve:

```text
audit_id
timestamp
user_query
claim
verification_status
verification_reason
risk_level
evidence
document_id
document_sha256
page_number
reviewer
review_decision
review_notes
```

This creates a complete audit trail.

---

# 21. Agentic Workflow

The final system should use a controlled agent workflow.

Conceptually:

```text
                    User
                     │
                     ▼
              Query Analyzer
                     │
                     ▼
              Retrieval Agent
                     │
                     ▼
              Evidence Selector
                     │
                     ▼
             Claim Extraction
                     │
                     ▼
          Deterministic Verifier
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       VERIFIED   INCONCLUSIVE  REJECTED
          │          │          │
          ▼          ▼          ▼
       Response    Audit       Audit
       Generator   Router      Router
          │
          ▼
      Output Guard
          │
          ▼
        User
```

The LLM is therefore one component inside a controlled workflow rather than the entire system.

---

# 22. LLM Responsibilities

The LLM may be used for:

### Query Understanding

Convert natural-language questions into structured retrieval requirements.

### Claim Extraction

Extract candidate claims from generated responses.

### Explanation

Explain verified evidence in natural language.

### Summarization

Summarize verified financial information.

The LLM must **not** be trusted for deterministic validation.

---

# 23. Non-LLM Responsibilities

Deterministic Python components should handle:

* hashing
* numeric normalization
* numerical comparison
* date/period comparison
* entity matching
* provenance validation
* retrieval scoring
* RRF
* risk scoring
* audit routing
* final verification status

This separation is fundamental to the compliance architecture.

---

# 24. Proposed Project Structure

```text
compliant-financial-rag/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── docs/
│   ├── DESIGN_DOCUMENT.md
│   ├── ARCHITECTURE.md
│   └── AUDIT_PROTOCOL.md
│
├── reports/
│   └── retrieval_benchmark.md
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── exceptions.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py
│   │   ├── document_hash.py
│   │   ├── section_detector.py
│   │   ├── chunker.py
│   │   └── pipeline.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── bm25.py
│   │   ├── vector_store.py
│   │   ├── hybrid.py
│   │   ├── evaluation.py
│   │   └── benchmark.py
│   │
│   ├── verification/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── reasons.py
│   │   ├── numeric.py
│   │   ├── dates.py
│   │   ├── entities.py
│   │   ├── provenance.py
│   │   ├── contradiction.py
│   │   └── verifier.py
│   │
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── input.py
│   │   ├── retrieval.py
│   │   ├── generation.py
│   │   └── output.py
│   │
│   ├── audit/
│   │   ├── __init__.py
│   │   ├── risk.py
│   │   ├── router.py
│   │   ├── models.py
│   │   └── queue.py
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   └── policies.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── prompts.py
│   │   └── response_parser.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── schemas.py
│   │   └── routes.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── logging.py
│
├── tests/
│   ├── fixtures/
│   ├── evaluation/
│   ├── test_ingestion.py
│   ├── test_chunking.py
│   ├── test_section_detector.py
│   ├── test_bm25.py
│   ├── test_vector_store.py
│   ├── test_hybrid.py
│   ├── test_retrieval_integration.py
│   ├── test_retrieval_evaluation.py
│   ├── test_benchmark.py
│   └── test_verification_models.py
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 25. Testing Strategy

Testing will happen at multiple levels.

## Unit Tests

Every deterministic component must have unit tests.

Examples:

```text
numeric verification
date verification
entity matching
provenance
risk scoring
RRF
```

## Integration Tests

Test interactions such as:

```text
PDF
 ↓
Chunks
 ↓
BM25 + Vector
 ↓
Hybrid
 ↓
Verifier
```

## Evaluation Tests

Measure:

```text
Recall@1
Recall@3
MRR
verification accuracy
false verification rate
```

## Guardrail Tests

Test:

```text
unsupported claims
wrong numbers
wrong dates
wrong entities
missing evidence
contradictory evidence
```

---

# 26. CI/CD

GitHub Actions should execute:

```text
install dependencies
      ↓
ruff
      ↓
pytest
      ↓
benchmark checks
```

The `main` branch should remain protected by passing CI.

---

# 27. Git Strategy

Both contributors work from the same repository.

Recommended branches:

```text
main
│
├── feature/ahmed-...
├── feature/salem-...
└── fix/...
```

Avoid directly mixing unrelated work.

Each commit should represent a coherent change.

Examples:

```text
feat(retrieval): add hybrid RRF retrieval
feat(verification): add numeric claim verifier
test(verification): add numeric mismatch cases
feat(audit): add risk-based routing
```

---

# 28. Development Rules

### Rule 1 — Tests before integration

Every major component must have tests.

### Rule 2 — Deterministic logic stays deterministic

Do not replace Python verification logic with an LLM prompt.

### Rule 3 — Preserve provenance

Never discard document/page/chunk metadata.

### Rule 4 — No unsupported answer

If evidence is insufficient:

```text
INCONCLUSIVE
```

not a fabricated answer.

### Rule 5 — Human review for high-risk cases

High-risk cases must enter the audit workflow.

### Rule 6 — CI must remain green

Before merging:

```text
ruff ✓
pytest ✓
```

---

# 29. Development Roadmap

## Phase 1 — Foundation

Completed:

* repository structure
* configuration
* exceptions
* logging
* PDF parsing
* document hashing
* section detection
* chunking

## Phase 2 — Retrieval

Completed:

* BM25
* VectorStore
* Hybrid RRF
* retrieval models
* retrieval evaluation
* benchmark

Current status:

```text
41 tests passing
```

## Phase 3 — Deterministic Verification

Next:

* claim model
* numeric verifier
* date verifier
* entity verifier
* provenance verifier
* contradiction detection
* combined verifier

## Phase 4 — Guardrails

Implement:

* input guardrails
* retrieval guardrails
* generation guardrails
* output verification

## Phase 5 — Audit

Implement:

* risk scoring
* audit routing
* audit queue
* audit records
* reviewer decisions

## Phase 6 — Agent

Implement:

* state
* workflow graph
* retrieval node
* verification node
* generation node
* audit node
* retry/fallback policies

## Phase 7 — API/UI

Implement:

* FastAPI
* request/response schemas
* document upload
* question answering
* evidence display
* verification status
* audit interface

## Phase 8 — Productionization

Implement:

* Docker
* CI/CD
* observability
* latency tracking
* token-cost tracking
* evaluation dashboard
* documentation

---

# 30. Team Structure

The project has two contributors:

```text
Ahmed
Salem
```

The architecture is divided according to component boundaries rather than arbitrary file counts.

This minimizes merge conflicts and allows both contributors to work independently.

---

# 31. Ahmed — Responsibilities

Ahmed owns the **retrieval, orchestration, and application layer**.

## A. Retrieval

Ahmed owns:

```text
src/retrieval/
```

Including:

* BM25
* VectorStore
* Hybrid RRF
* retrieval models
* retrieval benchmarks
* retrieval evaluation

Current work is already under Ahmed's ownership.

## B. Agent

Ahmed owns:

```text
src/agent/
```

Responsibilities:

* LangGraph/state-machine architecture
* agent state
* workflow nodes
* retrieval orchestration
* generation orchestration
* retry/fallback policies

## C. LLM Layer

Ahmed owns:

```text
src/llm/
```

Responsibilities:

* LLM client
* prompts
* structured output
* response parsing
* generation integration

## D. API

Ahmed owns:

```text
src/api/
```

Responsibilities:

* FastAPI
* endpoints
* schemas
* document upload
* question answering
* integration with agent

## E. Integration

Ahmed is responsible for integrating:

```text
Ingestion
   ↓
Retrieval
   ↓
Verification
   ↓
Guardrails
   ↓
Audit
   ↓
Agent
   ↓
API
```

---

# 32. Salem — Responsibilities

Salem owns the **verification, guardrail, risk, and audit layer**.

## A. Deterministic Verification

Salem owns:

```text
src/verification/
```

Responsibilities:

* claim models
* numeric verification
* date verification
* entity verification
* provenance verification
* contradiction detection
* verification engine

This is the most important ownership area for Salem.

## B. Guardrails

Salem owns:

```text
src/guardrails/
```

Responsibilities:

* input guardrails
* retrieval guardrails
* generation guardrails
* output guardrails
* unsupported claim detection
* hallucination prevention

## C. Audit

Salem owns:

```text
src/audit/
```

Responsibilities:

* risk scoring
* risk classification
* audit routing
* audit queue
* audit records
* human review workflow

## D. Verification Evaluation

Salem owns evaluation of:

```text
claim verification accuracy
numeric verification
period verification
entity verification
contradiction detection
false verification rate
```

---

# 33. Shared Responsibilities

Both contributors jointly own:

### Testing

Every PR must include appropriate tests.

### Code Quality

```text
ruff
pytest
type-safe interfaces
documentation
```

### Architecture Reviews

Major interface changes must be discussed before implementation.

### GitHub CI

Both contributors must keep CI passing.

### Final Integration

Ahmed and Salem jointly review the final end-to-end workflow.

---

# 34. Ownership Matrix

| Component               |    Ahmed    |    Salem    |
| ----------------------- | :---------: | :---------: |
| Configuration           |   Primary   |    Review   |
| Exceptions              |   Primary   |    Review   |
| Logging                 |   Primary   |    Review   |
| PDF ingestion           |   Primary   |    Review   |
| Hashing                 |   Primary   |    Review   |
| Section detection       |   Primary   |    Review   |
| Chunking                |   Primary   |    Review   |
| BM25                    | **Primary** |    Review   |
| VectorStore             | **Primary** |    Review   |
| Hybrid RRF              | **Primary** |    Review   |
| Retrieval evaluation    | **Primary** |    Review   |
| Claim models            |    Review   | **Primary** |
| Numeric verification    |    Review   | **Primary** |
| Date verification       |    Review   | **Primary** |
| Entity verification     |    Review   | **Primary** |
| Provenance verification |    Review   | **Primary** |
| Contradiction detection |    Review   | **Primary** |
| Verification engine     |    Review   | **Primary** |
| Guardrails              |    Review   | **Primary** |
| Risk scoring            |    Review   | **Primary** |
| Audit routing           |    Review   | **Primary** |
| Audit queue             |    Review   | **Primary** |
| Agent workflow          | **Primary** |    Review   |
| LLM integration         | **Primary** |    Review   |
| FastAPI                 | **Primary** |    Review   |
| UI                      | **Primary** |    Review   |
| CI/CD                   |    Shared   |    Shared   |
| Final integration       |    Shared   |    Shared   |
| Documentation           |    Shared   |    Shared   |

---

# 35. Immediate Task Distribution

## Ahmed — Next Tasks

Continue from the current completed retrieval system:

```text
41 tests ✓
```

Ahmed's next work:

1. Finish retrieval benchmark report.
2. Maintain retrieval interfaces.
3. Begin agent architecture design.
4. Prepare interfaces between retrieval and verification.
5. Prepare structured evidence objects for the verifier.

## Salem — Next Tasks

Salem starts with deterministic verification:

1. Complete `verification/models.py`.
2. Complete `verification/reasons.py`.
3. Implement numeric normalization.
4. Implement numeric claim verification.
5. Add numeric verification tests.
6. Implement period/date verification.
7. Add period verification tests.
8. Implement entity verification.
9. Add entity verification tests.
10. Implement provenance verification.
11. Add contradiction detection.

---

# 36. Interface Contract Between Ahmed and Salem

The most important collaboration boundary is:

```text
Ahmed
Retrieval
   │
   │ RetrievalResult
   ▼
Salem
Verification
```

Ahmed must provide evidence containing:

```text
chunk_id
document_id
text
page_number
section
document_sha256
score
retrieval_method
```

Salem's verifier consumes this evidence.

The verifier returns:

```text
claim_id
status
reason
confidence
evidence_chunk_id
```

This interface should remain stable.

---

# 37. Integration Contract

The final integration should follow:

```text
Ahmed Retrieval
      │
      ▼
RetrievalResult
      │
      ▼
Salem Verification
      │
      ▼
VerificationResult
      │
      ▼
Ahmed Agent
      │
      ├──────── VERIFIED ────────► Answer
      │
      ├──── INCONCLUSIVE ───────► Audit
      │
      └──── REJECTED ───────────► Audit / Safe Response
```

This allows Ahmed and Salem to develop their components independently.

---

# 38. Definition of Done

A feature is considered complete only when:

```text
Implementation ✓
Unit tests     ✓
Ruff           ✓
Integration    ✓
Documentation  ✓
```

For compliance-critical verification:

```text
Implementation
      +
Positive tests
      +
Negative tests
      +
Edge cases
      +
Auditability
```

are required.

---

# 39. Final Target Architecture

The final system should look like:

```text
                         ┌─────────────┐
                         │    User     │
                         └──────┬──────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Query Analyzer  │
                       └────────┬────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │  Hybrid Retrieval   │
                     │ BM25 + Vector + RRF │
                     └──────────┬──────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Evidence Store  │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Claim Extractor │
                       └────────┬────────┘
                                │
                                ▼
                 ┌────────────────────────────┐
                 │ Deterministic Verification │
                 │                            │
                 │ Numeric                    │
                 │ Date                       │
                 │ Entity                     │
                 │ Provenance                 │
                 │ Contradiction              │
                 └─────────────┬──────────────┘
                               │
                  ┌────────────┼────────────┐
                  ▼            ▼            ▼
              VERIFIED    INCONCLUSIVE   REJECTED
                  │            │            │
                  │            └─────┬──────┘
                  │                  ▼
                  │          ┌───────────────┐
                  │          │ Risk Engine   │
                  │          └───────┬───────┘
                  │                  │
                  │                  ▼
                  │          ┌───────────────┐
                  │          │ Human Audit   │
                  │          └───────────────┘
                  │
                  ▼
             ┌────────────┐
             │ LLM Explain│
             └─────┬──────┘
                   │
                   ▼
             ┌────────────┐
             │Output Guard│
             └─────┬──────┘
                   │
                   ▼
                User
```

---

# 40. Success Criteria

The project will be considered successful when it demonstrates:

### Retrieval

* Hybrid BM25 + vector retrieval
* measurable Recall@K and MRR
* preserved provenance

### Verification

* deterministic numerical verification
* deterministic period verification
* entity verification
* contradiction detection
* citation/provenance validation

### Safety

* unsupported claims rejected
* hallucinated financial numbers detected
* uncertain cases routed to humans
* high-risk cases never silently accepted

### Agent

* controlled multi-step workflow
* deterministic verification inside the workflow
* fallback handling
* auditable state transitions

### Production Engineering

* FastAPI
* automated tests
* Ruff
* GitHub Actions
* Docker
* logging
* latency tracking
* token-cost tracking
* documented architecture

---

# 41. Team Goal

The final portfolio demonstration should not simply show:

> "Ask questions about a PDF."

It should demonstrate:

> **"Ask a financial question, retrieve evidence using hybrid search, generate a candidate claim, deterministically verify the claim against the original evidence, identify contradictions or unsupported claims, and route high-risk cases to human audit."**

That is the core FinTech AI engineering story of this project.

---

# 42. Current Project Status

As of the beginning of the Ahmed + Salem collaboration:

```text
Python 3.12.9
```

Retrieval system:

```text
BM25                    ✓
VectorStore             ✓
Hybrid RRF              ✓
Retrieval evaluation    ✓
Benchmark               ✓
```

Current test suite:

```text
41 passed
```

Initial benchmark:

```text
BM25:
Recall@1 = 0.750
Recall@3 = 1.000
MRR      = 0.875

VectorStore:
Recall@1 = 1.000
Recall@3 = 1.000
MRR      = 1.000

Hybrid RRF:
Recall@1 = 1.000
Recall@3 = 1.000
MRR      = 1.000
```

Current development stage:

```text
INGESTION          ✓
CHUNKING           ✓
RETRIEVAL          ✓
RETRIEVAL EVAL     ✓
VERIFICATION       → CURRENT
GUARDRAILS         → NEXT
AUDIT              → NEXT
AGENT              → NEXT
API/UI             → NEXT
PRODUCTION         → NEXT
```

## Team

```text
Ahmed
├── Retrieval
├── Agent
├── LLM
├── API
└── Integration

Salem
├── Deterministic Verification
├── Guardrails
├── Risk Engine
└── Audit Workflow
```

**Next implementation milestone:** Salem begins the deterministic numeric verifier while Ahmed maintains the retrieval/evidence interface. The two components will be integrated through the `RetrievalResult → Claim → VerificationResult` contract defined above.
