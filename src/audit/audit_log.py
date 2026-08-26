# src/audit/audit_log.py
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
    claim_id = fields.Str(allow_none=True)
    verification_status = fields.Str(required=True)
    verification_reason = fields.Str(required=True)
    risk_level = fields.Str(required=True)
    evidence = fields.List(fields.Dict(), required=True)
    provenance = fields.List(fields.Dict())
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
        data.setdefault('claim_id', '')
        data.setdefault('provenance', [])
        data.setdefault('triggers', [])
        data.setdefault('verification_results', [])
        return AuditRecord(**data)


class AuditLogger:
    """Persists audit records to a JSON file (append-only)."""

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
        """Append a record. Raises if the same audit_id already exists."""
        records = self._load_all()
        existing_ids = {rec.get("audit_id") for rec in records}
        if record.audit_id in existing_ids:
            raise ValueError(f"Audit record with ID {record.audit_id} already exists")
        dumped = self.schema.dump(record)
        records.append(dumped)
        self._save_all(records)

    def get_all(self) -> List[AuditRecord]:
        return self.schema.load(self._load_all(), many=True)

    def get_by_audit_id(self, audit_id: str) -> Optional[AuditRecord]:
        for rec in self._load_all():
            if rec.get("audit_id") == audit_id:
                return self.schema.load(rec)
        return None