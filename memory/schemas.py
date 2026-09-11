from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from .models import Launch


class LaunchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    user_id: str
    topic: str
    mode: str
    blog_kind: Optional[str] = None
    blog_title: Optional[str] = None
    plan_summary: Optional[str] = None
    created_at: datetime

    @classmethod
    def from_launch(cls, launch: Launch) -> "LaunchResponse":
        return cls.model_validate(launch)


class LaunchDetailResponse(LaunchResponse):
    final_markdown: str


class LaunchListResponse(BaseModel):
    items: List[LaunchResponse]


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    role: str
    content: str
    created_at: datetime


class SessionSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: str
    summary: str
    updated_at: datetime


class UserFactResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    fact: str
    category: Optional[str] = None
    created_at: datetime


class UserMemoryResponse(BaseModel):
    user_id: str
    facts: List[UserFactResponse]
