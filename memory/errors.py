from __future__ import annotations


class MemoryError(Exception):
    """Base class for all memory subsystem errors."""


class InvalidUserIdError(MemoryError):
    """The provided user id is empty or otherwise invalid after sanitization."""


class SessionNotFoundError(MemoryError):
    """No session exists with the given id."""


class LaunchNotFoundError(MemoryError):
    """No launch exists with the given id."""
