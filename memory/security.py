from __future__ import annotations

import re

from .errors import InvalidUserIdError

MAX_USER_ID_LEN = 200
MAX_TITLE_LEN = 300

# Strips all ASCII control characters — used for single-line identifiers (user id, title).
_STRICT_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

# Strips control characters but preserves \t, \n, \r — used for free-text content that may
# legitimately span multiple lines (message/summary/fact text).
_LENIENT_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_user_id(user_id: str | None) -> str:
    """Strip control characters and cap length. Returns "" for blank/None input —
    callers should treat "" as anonymous/session-only rather than raising."""
    if not user_id:
        return ""
    cleaned = _STRICT_CONTROL_RE.sub("", user_id).strip()
    return cleaned[:MAX_USER_ID_LEN]


def require_user_id(user_id: str | None) -> str:
    """Like sanitize_user_id but raises when the result is empty, for code paths that
    genuinely need a real identity (e.g. writing long-term memory)."""
    cleaned = sanitize_user_id(user_id)
    if not cleaned:
        raise InvalidUserIdError("A non-empty user id is required.")
    return cleaned


def validate_message_content(content: str, max_chars: int) -> str:
    cleaned = _LENIENT_CONTROL_RE.sub("", content or "").strip()
    return cleaned[:max_chars]


def validate_text_field(text: str | None, max_chars: int = MAX_TITLE_LEN) -> str:
    if not text:
        return ""
    return _STRICT_CONTROL_RE.sub("", text).strip()[:max_chars]
