# 📄 **Complete README.md File**


# 📊 Compliant Financial RAG & Audit Agent

<div align="center">

**A production-ready, compliant financial document retrieval and analysis system with deterministic verification, guardrails, and human-in-the-loop auditing.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Tests](https://img.shields.io/badge/Tests-218%20Passing-brightgreen.svg)](https://github.com/yourusername/compliant-financial-rag/actions)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Query Examples](#query-examples)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**Compliant Financial RAG & Audit Agent** is an enterprise-grade system for analyzing financial documents (10-K filings, annual reports) with:

- **Deterministic claim verification** - No hallucinations, every claim is verified against source evidence
- **Multi-company & multi-metric support** - Compare Apple, Microsoft, and more in a single query
- **Intelligent routing** - Low-risk queries auto-answer, high-risk queries route to human review
- **Full audit trail** - Every query is logged with provenance for compliance
- **RESTful API** - Easy integration with existing systems

---

## ✨ Key Features

### 🔍 **Retrieval System**
- **Hybrid+ Retrieval**: BM25 + Vector + Pattern + Context search
- **Weighted Score Fusion**: Configurable BM25/Vector weights (default: 0.3/0.7)
- **Persistent Storage**: FAISS + Pickle and PostgreSQL + pgvector
- **Company-Aware Scoping**: Results filtered by company

### ✅ **Verification Pipeline**
- **Numeric Verification**: Exact value matching with unit handling
- **Period Verification**: Fiscal year validation
- **Entity Verification**: Company name matching
- **Contradiction Detection**: Identifies conflicting evidence
- **Provenance Tracking**: Every claim links to source document, page, and chunk

### 🛡️ **Guardrails & Safety**
- **Multi-Layer Guardrails**: Input, Retrieval, Confidence, Risk, Output
- **Deterministic Risk Assessment**: Rule-based risk scoring (0.0 - 1.0)
- **Confidence Scoring**: Normalized retrieval + verification confidence
- **Auto-Routing**: High-risk queries → Human review

### 📊 **Audit System**
- **Human Review Queue**: Pending reviews management
- **Audit Logging**: JSON-based audit trail
- **Decision Tracking**: Approve/Reject/Needs More Evidence
- **Full Provenance**: Document → Page → Chunk tracking

### 🤖 **LLM Integration**
- **Multi-Provider**: Gemini, OpenAI, Anthropic, Ollama
- **Limited Usage**: LLM only for answer formatting (not extraction)
- **Evidence-Grounded**: Prompts enforce evidence-only answers

### 📡 **API Layer**
- **FastAPI**: High-performance RESTful API
- **Interactive Docs**: Swagger UI at `/docs`
- **Health Checks**: Monitoring endpoints

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                    │
│                    FastAPI - /query, /audits, /documents                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENT WORKFLOW (LangGraph)                        │
│                                                                             │
│  Query Analysis → Retrieval → Claim Extraction → Verification → Guardrails │
│                                                                             │
│  Risk Assessment → Routing → (Audit | Answer Generation → Output Guard)    │
└─────────────────────────────────────────────────────────────────────────────┘
          │                     │                     │
          ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────────┐
│   RETRIEVAL     │   │  VERIFICATION   │   │       GUARDRAILS             │
│  Hybrid+        │   │  Deterministic  │   │  Confidence Scoring          │
│  BM25 + Vector  │   │  Numeric/Period │   │  Risk Assessment             │
│  RRF Fusion     │   │  Entity/Date    │   │  Policy Enforcement          │
└─────────────────┘   └─────────────────┘   └─────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.12+
python --version

# Docker (optional, for PostgreSQL)
docker --version
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/compliant-financial-rag.git
cd compliant-financial-rag

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment configuration
cp .env.example .env

# 5. Edit .env with your API keys
# Required: GEMINI_API_KEY or OPENAI_API_KEY
```

### Environment Setup

```bash
# .env - Minimum configuration
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp
DATABASE_URL=sqlite:///./data/vectors.db  # or PostgreSQL
USE_PG_VECTOR=false
```

### Index Documents

```bash
# Place your PDFs in data/raw/
# Then build the index:

# For FAISS (default, no external dependencies)
python reindex.py

# For PostgreSQL + pgvector
docker-compose up -d  # Start PostgreSQL
python reindex_pg.py
```

### Run the API Server

```bash
# Start the server
python run.py

# Server will run at: http://127.0.0.1:8000
```

### Test a Query

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"user_query": "Compare Apple and Microsoft revenue in 2025?"}'
```

---

## 📡 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Submit a financial question |
| `GET` | `/audits` | List pending human reviews |
| `GET` | `/audits/{id}` | Get specific audit record |
| `POST` | `/audits/{id}/decision` | Submit review decision |
| `GET` | `/documents` | List available documents |
| `GET` | `/documents/{name}` | Get document metadata |
| `GET` | `/companies` | List detected companies |
| `GET` | `/health` | Health check |

### Query Request

```json
{
  "user_query": "Compare Apple and Microsoft revenue in 2025?"
}
```

### Query Response

```json
{
  "final_answer": "Apple's revenue was $416,161 million...",
  "status": "generated",
  "should_route_to_audit": false,
  "audit_id": null,
  "evidence": [...],
  "verification_results": [...],
  "risk_assessment": {
    "risk_score": 0.0,
    "risk_level": "LOW",
    "triggers": [],
    "recommended_action": "AUTO_ANSWER"
  },
  "claims_count": 2
}
```

### Audit Decision Request

```json
{
  "decision": "APPROVE",
  "notes": "Revenue figures verified correctly",
  "reviewer": "john.doe@company.com"
}
```

---

## 📝 Query Examples

### ✅ Supported Query Types

| Type | Example |
|------|---------|
| **Single Company, Single Metric** | "What was Apple revenue in 2025?" |
| **Single Company, Multi-Metric** | "What was Apple revenue and net income in 2025?" |
| **Multi-Company, Single Metric** | "Compare Apple and Microsoft revenue in 2025?" |
| **Multi-Company, Multi-Metric** | "Compare Apple and Microsoft revenue and net income in 2025?" |
| **Narrative (What drove)** | "What drove Apple revenue growth in 2025?" |
| **Narrative (Why)** | "Why did Apple revenue increase in 2025?" |
| **Comparative Narrative** | "Why did Apple revenue increase more than Microsoft in 2025?" |
| **Different Period** | "What was Apple revenue in 2024?" |
| **Margin Query** | "What was Apple gross margin in 2025?" |
| **Growth Query** | "What was Apple revenue growth from 2024 to 2025?" |

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | - | `gemini`, `openai`, `anthropic`, `ollama` |
| `GEMINI_API_KEY` | - | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.0-flash-exp` | Gemini model |
| `LLM_MAX_TOKENS` | `4096` | Max tokens for LLM |
| `LLM_TEMPERATURE` | `0.0` | LLM temperature |
| `MIN_RETRIEVAL_CONFIDENCE` | `0.01` | RRF score threshold |
| `MIN_OVERALL_CONFIDENCE` | `0.70` | Overall confidence threshold |
| `MIN_EVIDENCE_CHUNKS` | `1` | Minimum evidence chunks |
| `BM25_WEIGHT` | `0.3` | BM25 weight in hybrid retrieval |
| `VECTOR_WEIGHT` | `0.7` | Vector weight in hybrid retrieval |
| `DATABASE_URL` | `sqlite:///./data/vectors.db` | Database connection |
| `USE_PG_VECTOR` | `false` | Use PostgreSQL (true/false) |
| `BLOCK_ON_NUMERIC_MISMATCH` | `false` | Block on numeric mismatch |
| `ALLOW_UNSUPPORTED_CLAIMS` | `false` | Allow unsupported claims |

### Vector Store Options

```bash
# FAISS (default, no external dependencies)
DATABASE_URL=sqlite:///./data/vectors.db
USE_PG_VECTOR=false

# PostgreSQL + pgvector (production)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/financial_rag
USE_PG_VECTOR=true
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest -v
```

### Run Specific Tests

```bash
pytest tests/test_agent_pipeline.py -v
pytest tests/test_verification/ -v
```

### Test Coverage

```bash
pytest --cov=src --cov-report=html
```

### Expected Results

```
218 passed in ~80s
```

---

## 🚀 Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/financial_rag
      - USE_PG_VECTOR=true
    depends_on:
      - postgres

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_DB=financial_rag
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Production Run

```bash
# Set production environment
export ENVIRONMENT=production
export DEBUG=false

# Run with uvicorn
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 📁 Project Structure

```
compliant-financial-rag/
├── src/
│   ├── agent/          # LangGraph workflow
│   ├── api/            # FastAPI endpoints
│   ├── audit/          # Human review system
│   ├── database/       # PostgreSQL models
│   ├── guardrails/     # Safety & risk assessment
│   ├── ingestion/      # PDF processing
│   ├── llm/            # LLM client
│   ├── retrieval/      # BM25 + Vector + Hybrid
│   ├── verification/   # Deterministic verification
│   └── utils/          # Logging & utilities
├── tests/              # 218 passing tests
├── data/
│   ├── raw/            # PDF documents
│   ├── processed/      # Processed chunks
│   └── indexes/        # FAISS indexes
├── .env.example
├── docker-compose.yml
├── pyproject.toml
├── reindex.py
├── reindex_pg.py
├── requirements.txt
├── run.py
└── README.md
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- **Tests**: All new features must include tests
- **Documentation**: Update README and docstrings
- **Linting**: Run `ruff check --fix` before committing
- **Type Hints**: Use Python type annotations

---

## 📊 System Metrics

| Metric | Value |
|--------|-------|
| **Total Tests** | 218 |
| **Pass Rate** | 100% |
| **Languages** | Python 3.12 |
| **Dependencies** | 18 packages |
| **API Endpoints** | 8 |
| **LLM Providers** | 4 (Gemini, OpenAI, Anthropic, Ollama) |
| **Retrieval Methods** | 3 (BM25, Vector, Hybrid RRF) |
| **Verifiers** | 6 (Numeric, Period, Entity, Date, Citation, Contradiction) |
| **Guardrails** | 7 |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangGraph](https://langchain-ai.github.io/langgraph/) - Workflow orchestration
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [Sentence-Transformers](https://www.sbert.net/) - Embeddings
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF parsing
- [pgvector](https://github.com/pgvector/pgvector) - PostgreSQL vector extension

---

## 📞 Support

- **Documentation**: [docs.example.com](https://docs.example.com)
- **Issues**: [GitHub Issues](https://github.com/yourusername/compliant-financial-rag/issues)
- **Email**: support@example.com

---

<div align="center">

**Built with ❤️ for compliance, accuracy, and auditability**

[⬆ Back to Top](#compliant-financial-rag--audit-agent)

</div>
```

---

## 📋 **README Sections Summary**

| Section | Content |
|---------|---------|
| **Overview** | System description and value proposition |
| **Key Features** | Detailed feature list |
| **Architecture** | System diagram |
| **Quick Start** | Installation and first run |
| **API Reference** | All endpoints and examples |
| **Query Examples** | Supported query types |
| **Configuration** | Environment variables |
| **Testing** | Test commands and expectations |
| **Deployment** | Docker and production setup |
| **Project Structure** | File tree |
| **Contributing** | Development guidelines |
| **System Metrics** | Test counts, dependencies |

---

