"""
Manages the audit queue (in-memory, easily swappable for Redis/DB later).
"""
# src/audit/queue.py
import uuid
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional

from src.audit.models import AuditRecord, AuditStatus, ReviewDecision


class AuditQueue:
    """
    Simple in-memory audit queue with thread-safe operations.
    """

    def __init__(self):
        self._records: Dict[str, AuditRecord] = {}
        self._lock = Lock()

    def enqueue(self, record: AuditRecord) -> str:
        """Add an audit record to the queue."""
        with self._lock:
            # Generate audit_id if not provided
            if not record.audit_id:
                record.audit_id = f"AUDIT-{uuid.uuid4().hex[:8].upper()}"
            record.status = AuditStatus.PENDING
            record.timestamp = datetime.now()
            self._records[record.audit_id] = record
            return record.audit_id

    def get_pending(self) -> List[AuditRecord]:
        """Retrieve all pending audit records."""
        with self._lock:
            return [r for r in self._records.values() if r.status == AuditStatus.PENDING]

    def get_by_id(self, audit_id: str) -> Optional[AuditRecord]:
        """Retrieve a specific audit record by ID."""
        with self._lock:
            return self._records.get(audit_id)

    def start_review(self, audit_id: str, reviewer: str) -> bool:
        """Mark a record as being reviewed."""
        with self._lock:
            record = self._records.get(audit_id)
            if not record or record.status != AuditStatus.PENDING:
                return False
            record.status = AuditStatus.IN_REVIEW
            record.reviewer = reviewer
            return True

    def resolve(
        self,
        audit_id: str,
        decision: ReviewDecision,
        notes: str,
    ) -> bool:
        """Resolve an audit record with a final decision."""
        with self._lock:
            record = self._records.get(audit_id)
            if not record:
                return False
            record.status = AuditStatus.RESOLVED
            record.review_decision = decision
            record.review_notes = notes
            record.review_timestamp = datetime.now()
            return True

    def get_all(self) -> List[AuditRecord]:
        """Return all audit records (for reporting)."""
        with self._lock:
            return list(self._records.values())

    def size(self) -> int:
        """Return the number of pending records."""
        with self._lock:
            return len([r for r in self._records.values() if r.status == AuditStatus.PENDING])