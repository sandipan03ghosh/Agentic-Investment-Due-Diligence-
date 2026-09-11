from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pytest

from ingestion.errors import (
    FileTooLargeError,
    InvalidCollectionNameError,
    InvalidFilenameError,
    UnsupportedFileTypeError,
)
from ingestion.security import (
    resolve_safe_path,
    sanitize_filename,
    sniff_content_type,
    stream_save_with_limit,
    validate_collection_name,
    validate_extension,
)


def test_sanitize_filename_strips_directory_components():
    assert sanitize_filename("../../etc/passwd.txt") == "passwd.txt"


def test_sanitize_filename_strips_windows_path_components():
    assert sanitize_filename("C:\\Users\\evil\\..\\..\\secrets.txt") == "secrets.txt"


def test_sanitize_filename_replaces_unsafe_characters():
    assert sanitize_filename("my report (final)!.txt") == "my_report__final__.txt"


def test_sanitize_filename_rejects_empty_result():
    with pytest.raises(InvalidFilenameError):
        sanitize_filename("...")


def test_validate_extension_accepts_allowed_types():
    assert validate_extension("brief.pdf") == ".pdf"
    assert validate_extension("brief.DOCX") == ".docx"


def test_validate_extension_rejects_disallowed_types():
    with pytest.raises(UnsupportedFileTypeError):
        validate_extension("payload.exe")


def test_resolve_safe_path_rejects_traversal(tmp_path: Path):
    with pytest.raises(InvalidFilenameError):
        resolve_safe_path(tmp_path, "..", "..", "evil.txt")


def test_resolve_safe_path_allows_nested_valid_path(tmp_path: Path):
    result = resolve_safe_path(tmp_path, "doc-id", "file.txt")
    assert result.is_relative_to(tmp_path.resolve())


def test_sniff_content_type_rejects_mismatched_pdf_header():
    with pytest.raises(UnsupportedFileTypeError):
        sniff_content_type(".pdf", b"not a real pdf")


def test_sniff_content_type_accepts_valid_pdf_header():
    sniff_content_type(".pdf", b"%PDF-1.7 rest of header")


def test_sniff_content_type_rejects_binary_disguised_as_txt():
    with pytest.raises(UnsupportedFileTypeError):
        sniff_content_type(".txt", b"\x00\x01\x02binary")


def test_stream_save_with_limit_writes_file_and_computes_hash(tmp_path: Path):
    data = b"hello world" * 100
    dest = tmp_path / "sub" / "out.txt"
    hasher = hashlib.sha256()
    size = stream_save_with_limit(BytesIO(data), dest, max_bytes=10_000, hasher=hasher)

    assert size == len(data)
    assert dest.read_bytes() == data
    assert hasher.hexdigest() == hashlib.sha256(data).hexdigest()


def test_stream_save_with_limit_enforces_max_size_and_cleans_up(tmp_path: Path):
    data = b"x" * 1000
    dest = tmp_path / "out.txt"
    hasher = hashlib.sha256()

    with pytest.raises(FileTooLargeError):
        stream_save_with_limit(BytesIO(data), dest, max_bytes=100, hasher=hasher)

    assert not dest.exists()


def test_validate_collection_name_accepts_simple_identifiers():
    assert validate_collection_name("product_briefs") == "product_briefs"
    assert validate_collection_name("WorkspaceA1") == "WorkspaceA1"


def test_validate_collection_name_rejects_leading_digit():
    with pytest.raises(InvalidCollectionNameError):
        validate_collection_name("1_workspace")


def test_validate_collection_name_rejects_spaces_and_special_characters():
    with pytest.raises(InvalidCollectionNameError):
        validate_collection_name("bad name!")
    with pytest.raises(InvalidCollectionNameError):
        validate_collection_name("../etc")


def test_validate_collection_name_rejects_empty_string():
    with pytest.raises(InvalidCollectionNameError):
        validate_collection_name("")
