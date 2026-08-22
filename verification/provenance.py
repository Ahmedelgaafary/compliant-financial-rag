"""
Validate that evidence has complete provenance metadata.
"""

from typing import Dict, Any, List

class ProvenanceVerifier:
    """
    Check that each evidence item contains required provenance fields.
    """

    REQUIRED_FIELDS = [
        "chunk_id",
        "document_id",
        "page_number",
        "section",
        "document_sha256"
    ]

    def verify(self, evidence_item: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Returns (is_valid, missing_fields)
        """
        missing = [field for field in self.REQUIRED_FIELDS if field not in evidence_item]
        return len(missing) == 0, missing

    def verify_list(self, evidence_list: List[Dict[str, Any]]) -> Tuple[bool, List[Dict]]:
        """
        Verify all items; return (all_valid, list of invalid items with missing fields)
        """
        invalid_items = []
        for idx, item in enumerate(evidence_list):
            is_valid, missing = self.verify(item)
            if not is_valid:
                invalid_items.append({"index": idx, "missing": missing})
        return len(invalid_items) == 0, invalid_items
