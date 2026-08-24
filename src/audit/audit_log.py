""""
This component handles writing audit records to a persistent store (e.g., JSON file, SQLite, or a cloud database). 
For now, we use a simple file‑based logger.
"""
# src/audit/audit_log.py
import json
import os
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from src.audit.models import AuditRecord


class AuditLogger:
    """
    Persists audit records to a JSON file (or database) for compliance.
    """

    def __init__(self, log_dir: str = "audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._file_path = self.log_dir / "audit_records.json"

    def _load_all(self) -> List[dict]:
        """Load existing records from the JSON file."""
        if not self._file_path.exists():
            return []
        with open(self._file_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    def _save_all(self, records: List[dict]):
        """Write all records back to the JSON file."""
        with open(self._file_path, "w") as f:
            json.dump(records, f, indent=2, default=str)

    def log(self, record: AuditRecord) -> None:
        """Append one audit record to the log."""
        records = self._load_all()
        # Convert record to dict (handle dataclasses and enums)
        record_dict = {
            k: v.value if hasattr(v, "value") else v
            for k, v in record.__dict__.items()
        }
        # Convert datetime to ISO string for JSON
        if "timestamp" in record_dict and isinstance(record_dict["timestamp"], datetime):
            record_dict["timestamp"] = record_dict["timestamp"].isoformat()
        if "review_timestamp" in record_dict and isinstance(record_dict["review_timestamp"], datetime):
            record_dict["review_timestamp"] = record_dict["review_timestamp"].isoformat()
        records.append(record_dict)
        self._save_all(records)

    def get_all(self) -> List[AuditRecord]:
        """Retrieve all logged records (as AuditRecord objects)."""
        records = self._load_all()
        # Convert back to AuditRecord (simplified – handle nested fields carefully)
        # In real code, you might use a serialization library like marshmallow.
        # For brevity, we return dicts or implement a proper deserialization.
        # We'll keep as dict for simplicity here.
        return records 
         # Or you can map back to AuditRecord objects

    def get_by_audit_id(self, audit_id: str) -> Optional[dict]:
        """Retrieve a specific record by ID from the log."""
        records = self._load_all()
        for rec in records:
            if rec.get("audit_id") == audit_id:
                return rec
        return None