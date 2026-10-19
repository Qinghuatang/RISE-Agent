"""Business repository errors, separate from RISE scientific failures."""


class TaskRepositoryError(RuntimeError):
    """Base class for business database operation errors."""


class TaskNotFound(TaskRepositoryError):
    """The requested task does not exist."""


class IdempotencyConflict(TaskRepositoryError):
    """An idempotency key was reused for a different request or event."""


class InvalidTaskTransition(TaskRepositoryError):
    """A requested lifecycle transition is not allowed."""


class LeaseOwnershipError(TaskRepositoryError):
    """The worker no longer owns a live task lease."""
