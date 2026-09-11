from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from .models import Document


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_filename: str
    file_type: str
    file_size_bytes: int
    content_hash: str
    collection: str
    status: str
    chunk_count: int
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_document(cls, document: Document) -> "DocumentResponse":
        return cls.model_validate(document)


class DocumentListResponse(BaseModel):
    items: List[DocumentResponse]
    limit: int
    offset: int


class ErrorResponse(BaseModel):
    detail: str
    existing_document_id: Optional[str] = None


class CollectionListResponse(BaseModel):
    collections: List[str]
