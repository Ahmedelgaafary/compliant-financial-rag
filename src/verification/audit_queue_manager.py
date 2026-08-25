from src.verification.audit_queue import (
    AuditQueueItem,
    AuditQueueStatus,
)


class AuditQueueManager:
    """Manage the lifecycle of human audit queue items."""

    def __init__(self) -> None:
        self._items: dict[str, AuditQueueItem] = {}

    def add(self, item: AuditQueueItem) -> None:
        """Add a new audit item to the queue."""

        if item.queue_id in self._items:
            raise ValueError(
                f"Audit queue item already exists: {item.queue_id}"
            )

        self._items[item.queue_id] = item

    def get(
        self,
        queue_id: str,
    ) -> AuditQueueItem:
        """Return an audit item by queue ID."""

        try:
            return self._items[queue_id]
        except KeyError as exc:
            raise KeyError(
                f"Audit queue item not found: {queue_id}"
            ) from exc

    def pending(self) -> list[AuditQueueItem]:
        """Return all pending audit items."""

        return [
            item
            for item in self._items.values()
            if item.status == AuditQueueStatus.PENDING
        ]

    def history(self) -> list[AuditQueueItem]:
        """Return all audit items."""

        return list(self._items.values())

    def approve(
        self,
        queue_id: str,
    ) -> AuditQueueItem:
        """Approve a pending audit item."""

        item = self.get(queue_id)

        self._ensure_pending(item)

        updated = item.approve()
        self._items[queue_id] = updated

        return updated

    def reject(
        self,
        queue_id: str,
    ) -> AuditQueueItem:
        """Reject a pending audit item."""

        item = self.get(queue_id)

        self._ensure_pending(item)

        updated = item.reject()
        self._items[queue_id] = updated

        return updated

    @staticmethod
    def _ensure_pending(
        item: AuditQueueItem,
    ) -> None:
        """Ensure an item is still pending."""

        if item.status != AuditQueueStatus.PENDING:
            raise ValueError(
                f"Audit item is already resolved: {item.queue_id}"
            )