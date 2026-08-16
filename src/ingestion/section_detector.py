import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    """A detected document section."""

    title: str
    text: str


class FinancialSectionDetector:
    """Detect common financial-document section headings."""

    DEFAULT_SECTIONS = {
        "business overview",
        "risk factors",
        "management discussion and analysis",
        "financial statements",
        "income statement",
        "balance sheet",
        "cash flow statement",
        "notes to the financial statements",
    }

    def __init__(
        self,
        known_sections: set[str] | None = None,
    ) -> None:
        self.known_sections = {
            section.lower()
            for section in (
                known_sections or self.DEFAULT_SECTIONS
            )
        }

    def detect(self, text: str) -> tuple[Section, ...]:
        """
        Detect known financial sections in document text.

        Args:
            text: Raw document text.

        Returns:
            Tuple of detected sections.
        """

        lines = text.splitlines()

        sections: list[Section] = []
        current_title: str | None = None
        current_lines: list[str] = []

        for line in lines:
            normalized_line = self._normalize_heading(line)

            if self._is_heading(normalized_line):
                if current_title is not None:
                    sections.append(
                        Section(
                            title=current_title,
                            text="\n".join(current_lines).strip(),
                        )
                    )

                current_title = self._format_title(
                    normalized_line
                )
                current_lines = []

                continue

            if current_title is not None:
                current_lines.append(line)

        if current_title is not None:
            sections.append(
                Section(
                    title=current_title,
                    text="\n".join(current_lines).strip(),
                )
            )

        return tuple(sections)

    def _is_heading(self, line: str) -> bool:
        """Return whether a line matches a known section heading."""

        normalized = line.lower().strip()

        return normalized in self.known_sections

    @staticmethod
    def _normalize_heading(line: str) -> str:
        """Normalize heading text for matching."""

        return re.sub(
            r"\s+",
            " ",
            line.strip(),
        )

    @staticmethod
    def _format_title(title: str) -> str:
        """Convert a normalized heading into title case."""

        return title.title()