import hashlib
from pathlib import Path

from src.exceptions import DocumentProcessingError
from src.utils.logging import get_logger

logger = get_logger(__name__)


def calculate_sha256(file_path: Path) -> str:
    """
    Calculate the SHA-256 hash of a file.

    Args:
        file_path: Path to the file.

    Returns:
        Hexadecimal SHA-256 digest.

    Raises:
        DocumentProcessingError: If the file cannot be read.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise DocumentProcessingError(
            f"Cannot hash missing file: {file_path}"
        )

    if not file_path.is_file():
        raise DocumentProcessingError(
            f"Cannot hash non-file path: {file_path}"
        )

    sha256 = hashlib.sha256()

    try:
        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                sha256.update(chunk)

    except OSError as exc:
        raise DocumentProcessingError(
            f"Unable to read file for hashing: {file_path}"
        ) from exc

    digest = sha256.hexdigest()

    logger.info(
        "SHA-256 calculated: file=%s hash=%s",
        file_path.name,
        digest,
    )

    return digest