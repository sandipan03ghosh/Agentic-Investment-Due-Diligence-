from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO

from .errors import (
    FileTooLargeError,
    InvalidCollectionNameError,
    InvalidFilenameError,
    UnsupportedFileTypeError,
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}

_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]")
_MAX_FILENAME_LEN = 255

# Must be a safe Weaviate collection/class identifier: starts with a letter, then
# letters/digits/underscores only. Keeps user-supplied "workspace" names from ever
# reaching the vector store as anything other than a validated identifier.
_COLLECTION_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")


def validate_collection_name(name: str) -> str:
    if not _COLLECTION_NAME_RE.match(name or ""):
        raise InvalidCollectionNameError(
            "Collection name must start with a letter and contain only letters, "
            "digits, and underscores (max 128 characters)."
        )
    return name

_MAGIC_BYTES: dict[str, bytes] = {
    ".pdf": b"%PDF-",
    ".docx": b"PK\x03\x04",
}

_READ_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def sanitize_filename(filename: str) -> str:
    """Strip any path component and reduce to a safe character set. Raises on empty/unsafe result."""
    name = Path(filename or "").name
    name = _SAFE_CHARS_RE.sub("_", name)
    name = name.lstrip(".")
    name = name[:_MAX_FILENAME_LEN]
    if not name or name in {"_", "-"}:
        raise InvalidFilenameError("Uploaded filename is empty or invalid after sanitization.")
    return name


def validate_extension(filename: str) -> str:
    """Return the lowercase extension if allowed, else raise UnsupportedFileTypeError."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise UnsupportedFileTypeError(f"File type '{ext or '(none)'}' is not supported. Allowed: {allowed}")
    return ext


def resolve_safe_path(base_dir: Path, *parts: str) -> Path:
    """Join base_dir with parts and assert the result stays within base_dir (defense-in-depth vs traversal)."""
    base_resolved = base_dir.resolve()
    candidate = base_resolved.joinpath(*parts).resolve()
    if not candidate.is_relative_to(base_resolved):
        raise InvalidFilenameError("Resolved upload path escapes the configured upload directory.")
    return candidate


def _looks_like_binary(sample: bytes) -> bool:
    return b"\x00" in sample


def sniff_content_type(extension: str, header: bytes) -> None:
    """Verify the file's magic bytes match its claimed extension; raise on mismatch/spoofing."""
    if extension in _MAGIC_BYTES:
        expected = _MAGIC_BYTES[extension]
        if not header.startswith(expected):
            raise UnsupportedFileTypeError(
                f"File content does not match the expected format for '{extension}' files."
            )
    elif extension in {".txt", ".md", ".markdown"}:
        if _looks_like_binary(header):
            raise UnsupportedFileTypeError(
                f"File claims to be '{extension}' but appears to contain binary data."
            )


def stream_save_with_limit(
    source: BinaryIO,
    destination: Path,
    max_bytes: int,
    hasher,
) -> int:
    """
    Stream-copy `source` into `destination` in chunks, updating `hasher` and enforcing `max_bytes`.
    Deletes the partial file and raises FileTooLargeError if the limit is exceeded.
    Returns the total number of bytes written. Also validates content-type magic bytes on the first chunk.
    """
    total = 0
    first_chunk = True
    extension = destination.suffix.lower()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(destination, "wb") as out:
            while True:
                chunk = source.read(_READ_CHUNK_SIZE)
                if not chunk:
                    break
                if first_chunk:
                    sniff_content_type(extension, chunk[:16])
                    first_chunk = False
                total += len(chunk)
                if total > max_bytes:
                    raise FileTooLargeError(
                        f"Upload exceeds the maximum allowed size of {max_bytes} bytes."
                    )
                hasher.update(chunk)
                out.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return total


def delete_upload_file(base_dir: Path, relative_path: Path) -> None:
    """Delete a file that must live under base_dir. Refuses to delete anything outside it."""
    base_resolved = base_dir.resolve()
    target = (base_resolved / relative_path).resolve()
    if not target.is_relative_to(base_resolved):
        raise InvalidFilenameError("Refusing to delete a path outside the upload directory.")
    target.unlink(missing_ok=True)
