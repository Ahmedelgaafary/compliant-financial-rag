class FinancialRAGError(Exception):
    """Base exception for the financial RAG application."""


class ConfigurationError(FinancialRAGError):
    """Raised when application configuration is invalid."""


class DocumentProcessingError(FinancialRAGError):
    """Raised when a financial document cannot be processed."""


class RetrievalError(FinancialRAGError):
    """Raised when evidence retrieval fails."""


class VerificationError(FinancialRAGError):
    """Raised when claim verification fails unexpectedly."""


class AuditError(FinancialRAGError):
    """Raised when an audit operation fails."""