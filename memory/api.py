from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from logging_config import get_logger

from .db import get_session
from .errors import InvalidUserIdError
from .repository import MemoryRepository
from .schemas import (
    LaunchDetailResponse,
    LaunchListResponse,
    LaunchResponse,
    MessageResponse,
    SessionSummaryResponse,
    UserFactResponse,
    UserMemoryResponse,
)
from .security import require_user_id

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])
logger = get_logger(__name__)


@router.get("/launches", response_model=LaunchListResponse)
async def list_launches(
    user_id: str = Query(...),
    keyword: Optional[str] = Query(default=None),
    limit: int = Query(default=10, ge=1, le=100),
) -> LaunchListResponse:
    try:
        clean_user_id = require_user_id(user_id)
    except InvalidUserIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        with get_session() as db_session:
            repo = MemoryRepository(db_session)
            rows = (
                repo.search_launches(clean_user_id, keyword, limit)
                if keyword
                else repo.list_recent_launches(clean_user_id, limit)
            )
            return LaunchListResponse(items=[LaunchResponse.from_launch(r) for r in rows])
    except Exception:
        logger.exception("list_launches_failed")
        raise HTTPException(status_code=500, detail="Failed to list launches due to an internal error.")


@router.get("/launches/{launch_id}", response_model=LaunchDetailResponse)
async def get_launch(launch_id: str) -> LaunchDetailResponse:
    try:
        with get_session() as db_session:
            repo = MemoryRepository(db_session)
            launch = repo.get_launch(launch_id)
            if launch is None:
                raise HTTPException(status_code=404, detail=f"No launch with id '{launch_id}'")
            return LaunchDetailResponse.model_validate(launch)
    except HTTPException:
        raise
    except Exception:
        logger.exception("get_launch_failed")
        raise HTTPException(status_code=500, detail="Failed to fetch launch due to an internal error.")


@router.get("/sessions/{session_id}/messages", response_model=List[MessageResponse])
async def list_messages(session_id: str) -> List[MessageResponse]:
    try:
        with get_session() as db_session:
            repo = MemoryRepository(db_session)
            messages = repo.list_messages(session_id)
            return [MessageResponse.model_validate(m) for m in messages]
    except Exception:
        logger.exception("list_messages_failed")
        raise HTTPException(status_code=500, detail="Failed to list messages due to an internal error.")


@router.get("/sessions/{session_id}/summary", response_model=Optional[SessionSummaryResponse])
async def get_session_summary(session_id: str):
    try:
        with get_session() as db_session:
            repo = MemoryRepository(db_session)
            summary = repo.get_session_summary(session_id)
            return SessionSummaryResponse.model_validate(summary) if summary else None
    except Exception:
        logger.exception("get_session_summary_failed")
        raise HTTPException(status_code=500, detail="Failed to fetch session summary due to an internal error.")


@router.get("/users/{user_id}/memory", response_model=UserMemoryResponse)
async def get_user_memory(user_id: str, limit: int = Query(default=20, ge=1, le=100)) -> UserMemoryResponse:
    try:
        clean_user_id = require_user_id(user_id)
    except InvalidUserIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        with get_session() as db_session:
            repo = MemoryRepository(db_session)
            facts = repo.list_user_facts(clean_user_id, limit=limit)
            return UserMemoryResponse(
                user_id=clean_user_id,
                facts=[UserFactResponse.model_validate(f) for f in facts],
            )
    except Exception:
        logger.exception("get_user_memory_failed")
        raise HTTPException(status_code=500, detail="Failed to fetch user memory due to an internal error.")
