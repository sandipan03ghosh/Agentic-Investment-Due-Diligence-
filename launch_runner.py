from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from backend import llm
from logging_config import get_logger
from memory.models import Launch
from memory.service import MemoryService, build_memory_service

logger = get_logger(__name__)

_memory_service: Optional[MemoryService] = None


def _get_memory_service() -> MemoryService:
    global _memory_service
    if _memory_service is None:
        _memory_service = build_memory_service(llm)
    return _memory_service


def prepare_inputs(topic: str, as_of: date, user_id: Optional[str], session_id: str) -> Dict[str, Any]:
    """Build the LangGraph State inputs dict for a run, enriched with prior memory context."""
    memory_context = ""
    try:
        memory_context = _get_memory_service().get_context_for_run(user_id, session_id, topic)
    except Exception:
        logger.warning("memory_context_fetch_failed")

    return {
        "topic": topic,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
        "memory_context": memory_context,
    }


def _plan_to_dict(plan: Any) -> Dict[str, Any]:
    if plan is None:
        return {}
    if hasattr(plan, "model_dump"):
        return plan.model_dump()
    if isinstance(plan, dict):
        return plan
    return {}


def record_result(user_id: Optional[str], session_id: str, inputs: Dict[str, Any], out: Dict[str, Any]) -> None:
    """Persist a completed run to memory (conversation turn + launch record). Never raises —
    a memory failure must not take down an otherwise-successful report generation."""
    try:
        topic = out.get("topic") or inputs.get("topic", "")
        mode = out.get("mode") or inputs.get("mode") or "unknown"
        plan_dict = _plan_to_dict(out.get("plan"))
        blog_kind = plan_dict.get("blog_kind")
        blog_title = plan_dict.get("blog_title")
        tasks = plan_dict.get("tasks") or []

        plan_summary = None
        if plan_dict:
            plan_summary = (
                f"Audience: {plan_dict.get('audience', 'n/a')} | "
                f"Tone: {plan_dict.get('tone', 'n/a')} | "
                f"{len(tasks)} section(s)"
            )

        _get_memory_service().record_interaction(
            user_id=user_id,
            session_id=session_id,
            topic=topic,
            mode=mode,
            blog_kind=blog_kind,
            blog_title=blog_title,
            plan_summary=plan_summary,
            final_markdown=out.get("final") or "",
        )
    except Exception:
        logger.warning("record_result_failed")


def list_recent_launches(user_id: Optional[str], session_id: str, limit: int = 10) -> List[Launch]:
    try:
        return _get_memory_service().list_recent_launches(user_id, session_id, limit)
    except Exception:
        logger.warning("list_recent_launches_failed")
        return []
