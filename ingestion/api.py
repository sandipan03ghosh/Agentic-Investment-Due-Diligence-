from __future__ import annotations

import asyncio
import hmac
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Header, HTTPException, Query, UploadFile

import vector_db
from config import settings
from logging_config import get_logger

from .errors import (
    DocumentNotFoundError,
    DocumentNotReindexableError,
    DuplicateDocumentError,
    FileTooLargeError,
    InvalidCollectionNameError,
    InvalidFilenameError,
    UnsupportedFileTypeError,
)
from .schemas import CollectionListResponse, DocumentListResponse, DocumentResponse
from .service import IngestionService, build_service

logger = get_logger(__name__)

if not settings.INGESTION_API_KEY:
    logger.warning(
        "ingestion_api_key_unset",
        extra={"detail": "INGESTION_API_KEY is not set -- the ingestion API is running unauthenticated. Set it before any non-local deployment."},
    )


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Shared-secret gate. A no-op (any caller allowed) only when INGESTION_API_KEY is
    unset, e.g. for local/dev use -- the startup warning above makes that state visible
    rather than silently open in a real deployment. Attached at the router level below
    so every route under this prefix (upload/list/get/delete/reindex/collections/health)
    is covered by construction. Verified this is the only router in the codebase that
    registers /api/v1/ingestion/* -- ingestion/api.py is the sole place these routes
    are defined (memory/api.py is a separate router under a different prefix), so there
    is no other registration point that could bypass this dependency."""
    if not settings.INGESTION_API_KEY:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.INGESTION_API_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


router = APIRouter(prefix="/api/v1/ingestion", tags=["ingestion"], dependencies=[Depends(require_api_key)])


def get_ingestion_service() -> IngestionService:
    return build_service()


@router.post("/documents", response_model=DocumentResponse, status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection: Optional[str] = Form(default=None, description="Target collection/workspace; defaults to WEAVIATE_COLLECTION"),
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentResponse:
    try:
        document = await asyncio.to_thread(service.ingest_upload, file, collection)
    except DuplicateDocumentError as exc:
        raise HTTPException(
            status_code=409,
            detail={"detail": str(exc), "existing_document_id": exc.existing_document_id},
        ) from exc
    except (UnsupportedFileTypeError, InvalidFilenameError, InvalidCollectionNameError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception:
        logger.exception("upload_failed")
        raise HTTPException(status_code=500, detail="Upload failed due to an internal error.")

    background_tasks.add_task(service.process_document, document.id)
    return DocumentResponse.from_document(document)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentListResponse:
    documents = await asyncio.to_thread(service.list_documents, status, limit, offset)
    return DocumentListResponse(
        items=[DocumentResponse.from_document(d) for d in documents], limit=limit, offset=offset
    )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str, service: IngestionService = Depends(get_ingestion_service)
) -> DocumentResponse:
    try:
        document = await asyncio.to_thread(service.get_document, document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DocumentResponse.from_document(document)


@router.delete("/documents/{document_id}", response_model=DocumentResponse)
async def delete_document(
    document_id: str, service: IngestionService = Depends(get_ingestion_service)
) -> DocumentResponse:
    try:
        document = await asyncio.to_thread(service.delete_document, document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception:
        logger.exception("delete_failed", extra={"document_id": document_id})
        raise HTTPException(status_code=500, detail="Delete failed due to an internal error.")
    return DocumentResponse.from_document(document)


@router.post("/documents/{document_id}/reindex", response_model=DocumentResponse, status_code=202)
async def reindex_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentResponse:
    try:
        document = await asyncio.to_thread(service.prepare_reindex, document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentNotReindexableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    background_tasks.add_task(service.process_document, document_id)
    return DocumentResponse.from_document(document)


@router.get("/collections", response_model=CollectionListResponse)
async def list_collections(
    service: IngestionService = Depends(get_ingestion_service),
) -> CollectionListResponse:
    try:
        collections = await asyncio.to_thread(service.list_collections)
    except Exception:
        logger.exception("list_collections_failed")
        raise HTTPException(status_code=503, detail="Vector store is unavailable.")
    return CollectionListResponse(collections=collections)


@router.get("/health")
async def health() -> dict:
    db_ok = True
    vector_ok = True
    try:
        from .db import get_session

        with get_session():
            pass
    except Exception:
        db_ok = False
    vector_ok = vector_db.is_ready()
    status_code = "ok" if db_ok and vector_ok else "degraded"
    return {"status": status_code, "database": db_ok, "vector_store": vector_ok}
