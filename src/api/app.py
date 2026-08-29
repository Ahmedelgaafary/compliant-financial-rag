# src/api/app.py

from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(
    title="Compliant Financial RAG & Audit Agent",
    description=(
        "API for compliant financial document retrieval, "
        "deterministic claim verification, risk assessment, "
        "and human-in-the-loop audit routing."
    ),
    version="0.1.0",
)

app.include_router(router)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    """Return basic API information."""

    return {
        "name": "Compliant Financial RAG & Audit Agent",
        "version": "0.1.0",
        "status": "ok",
    }