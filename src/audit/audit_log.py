"""
Persists audit records to a JSON file using marshmallow serialization.
"""

import json
from pathlib import Path
from typing import List, Optional

from marshmallow import Schema, fields, post_load

from src.audit.models import AuditRecord, AuditStatus, ReviewDecision


class AuditRecordSchema(Schema):
    """Marshmallow schema for AuditRecord."""
    audit_id = fields.Str(required=True)
    timestamp = fields.DateTime(required=True)
    user_query = fields.Str(required=True)
    claim = fields.Str(required=True)
    verification_status = fields.Str(required=True)
    verification_reason = fields.Str(required=True)
    risk_level = fields.Str(required=True)
    evidence = fields.List(fields.Dict(), required=True)
    document_id = fields.Str(required=True)
    document_sha256 = fields.Str(required=True)
    page_number = fields.Int(required=True)

    reviewer = fields.Str(allow_none=True)
    review_decision = fields.Enum(ReviewDecision, allow_none=True)
    review_notes = fields.Str(allow_none=True)
    review_timestamp = fields.DateTime(allow_none=True)
    status = fields.Enum(AuditStatus, required=True)

    confidence_score = fields.Float(allow_none=True)
    risk_score = fields.Float(allow_none=True)
    triggers = fields.List(fields.Str())
    verification_results = fields.List(fields.Dict())

    @post_load
    def make_record(self, data, **kwargs) -> AuditRecord:
        """Convert loaded dict back to AuditRecord instance."""
        data.setdefault('triggers', [])
        data.setdefault('verification_results', [])
        return AuditRecord(**data)


class AuditLogger:
    """Persists audit records to a JSON file using marshmallow."""

    def __init__(self, log_dir: str = "audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._file_path = self.log_dir / "audit_records.json"
        self.schema = AuditRecordSchema()

    def _load_all(self) -> List[dict]:
        if not self._file_path.exists():
            return []
        with open(self._file_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    def _save_all(self, records: List[dict]) -> None:
        with open(self._file_path, "w") as f:
            json.dump(records, f, indent=2, default=str)

    def log(self, record: AuditRecord) -> None:
        records = self._load_all()
        dumped = self.schema.dump(record)
        records.append(dumped)
        self._save_all(records)

    def get_all(self) -> List[AuditRecord]:
        records = self._load_all()
        return self.schema.load(records, many=True)

    def get_by_audit_id(self, audit_id: str) -> Optional[AuditRecord]:
        records = self._load_all()
        for rec in records:
            if rec.get("audit_id") == audit_id:
                return self.schema.load(rec)
        return None