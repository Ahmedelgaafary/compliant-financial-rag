#!/usr/bin/env python
"""
Production entry point for the Compliant Financial RAG Agent.

Run from project root:
    python run.py

Or with uvicorn directly:
    uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
"""

import logging
import os
import sys
from pathlib import Path

import uvicorn

# Now import application modules
from src.utils.logging import configure_logging

# Add src to path so imports work
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables FIRST
from dotenv import load_dotenv

load_dotenv()




def main():
    """Start the production server."""
    # Configure logging
    configure_logging()
    logger = logging.getLogger(__name__)

    # Get configuration from environment
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    debug = os.environ.get("DEBUG", "true").lower() == "true"
    environment = os.environ.get("ENVIRONMENT", "development")

    # Validate critical environment variables
    llm_provider = os.environ.get("LLM_PROVIDER")
    if not llm_provider:
        logger.warning("LLM_PROVIDER not set in .env")
        logger.warning("Will attempt auto-detection")
    
    # Check API keys based on provider
    if llm_provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY not set in .env")
        logger.error("Please add GEMINI_API_KEY to your .env file")
        sys.exit(1)
    
    if llm_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set in .env")
        logger.error("Please add OPENAI_API_KEY to your .env file")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("🚀 Compliant Financial RAG & Audit Agent")
    logger.info("=" * 60)
    logger.info(f"  Environment:    {environment}")
    logger.info(f"  Host:           {host}:{port}")
    logger.info(f"  Debug:          {debug}")
    logger.info(f"  LLM Provider:   {llm_provider or 'auto-detect'}")
    logger.info(f"  Log Level:      {logging.getLevelName(logging.getLogger().level)}")
    logger.info("=" * 60)

    # Import app here so it loads after environment is configured

    # Run server
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=debug,
        log_level="debug" if debug else "info",
        access_log=debug,
    )


if __name__ == "__main__":
    main()