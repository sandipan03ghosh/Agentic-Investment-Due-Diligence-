from __future__ import annotations


class IngestionError(Exception):
    """Base class for all ingestion pipeline errors."""


class InvalidFilenameError(IngestionError):
    """The uploaded filename could not be sanitized into a safe, non-empty name."""


class InvalidCollectionNameError(IngestionError):
    """The requested collection/workspace name does not meet the naming policy."""


class UnsupportedFileTypeError(IngestionError):
    """The uploaded file extension or content does not match an allowed document type."""


class FileTooLargeError(IngestionError):
    """The uploaded file exceeds the configured maximum size."""


class DuplicateDocumentError(IngestionError):
    """A document with the same content hash is already ingested (or in progress)."""

    def __init__(self, message: str, existing_document_id: str):
        super().__init__(message)
        self.existing_document_id = existing_document_id


class DocumentNotFoundError(IngestionError):
    """No document exists with the given id."""


class DocumentNotReindexableError(IngestionError):
    """The document has no original file on disk to reindex from (e.g. it was deleted)."""


class ExtractionError(IngestionError):
    """Text/metadata extraction from the uploaded file failed."""
