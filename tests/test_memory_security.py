from __future__ import annotations

import pytest

from memory.errors import InvalidUserIdError
from memory.security import (
    require_user_id,
    sanitize_user_id,
    validate_message_content,
    validate_text_field,
)


def test_sanitize_user_id_strips_control_characters():
    assert sanitize_user_id("alice\x00\x01@example.com") == "alice@example.com"


def test_sanitize_user_id_caps_length():
    result = sanitize_user_id("a" * 300)
    assert len(result) == 200


def test_sanitize_user_id_returns_empty_for_none_or_blank():
    assert sanitize_user_id(None) == ""
    assert sanitize_user_id("   ") == ""
    assert sanitize_user_id("") == ""


def test_require_user_id_raises_for_blank():
    with pytest.raises(InvalidUserIdError):
        require_user_id("")
    with pytest.raises(InvalidUserIdError):
        require_user_id(None)


def test_require_user_id_returns_sanitized_value():
    assert require_user_id("  bob@example.com  ") == "bob@example.com"


def test_validate_message_content_caps_length():
    text = "x" * 100
    assert validate_message_content(text, max_chars=10) == "x" * 10


def test_validate_message_content_preserves_newlines():
    text = "line one\nline two"
    assert validate_message_content(text, max_chars=100) == text


def test_validate_message_content_strips_other_control_chars():
    text = "hello\x00world"
    assert validate_message_content(text, max_chars=100) == "helloworld"


def test_validate_text_field_strips_all_control_chars_including_newline():
    text = "title\nwith\nnewlines"
    assert "\n" not in validate_text_field(text, max_chars=100)


def test_validate_text_field_returns_empty_for_none():
    assert validate_text_field(None) == ""
