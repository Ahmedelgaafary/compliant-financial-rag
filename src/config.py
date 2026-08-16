from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEXES_DIR = DATA_DIR / "indexes"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------
# Application settings
# ---------------------------------------------------------


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # Application
    app_name: str = "Compliant Financial RAG & Audit Agent"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True

    # LLM
    llm_model: str = "gpt-4o-mini"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    top_k_retrieval: int = 20
    top_k_reranking: int = 5

    # Verification
    numerical_tolerance: float = 0.01
    minimum_verification_score: float = 0.90

    # Risk thresholds
    low_risk_threshold: float = 0.90
    medium_risk_threshold: float = 0.70

    # Database
    database_url: str = "sqlite:///./data/audit.db"

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()