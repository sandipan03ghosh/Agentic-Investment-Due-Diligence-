from __future__ import annotations

import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from config import settings

from .errors import ExtractionError


@dataclass
class LoadedDocument:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentLoader(ABC):
    """Extracts text and metadata from a single file. One subclass per supported file type."""

    @abstractmethod
    def load(self, path: Path) -> LoadedDocument:
        raise NotImplementedError


def _enforce_extracted_size(text: str) -> str:
    """Reject text that exceeds the configured extraction cap. Applied uniformly across
    every loader -- this is the second, format-independent line of defense: the upload
    -size limit only bounds bytes on disk, and the DOCX zip-bomb check below only bounds
    what the archive *claims* to contain, so this is what actually caps memory use
    regardless of how a file got past those earlier checks."""
    if len(text) > settings.MAX_EXTRACTED_CHARS:
        raise ExtractionError(
            f"Extracted text ({len(text)} chars) exceeds the maximum allowed "
            f"({settings.MAX_EXTRACTED_CHARS} chars); the document was rejected "
            "to avoid excessive memory/CPU use."
        )
    return text


def _check_docx_not_a_zip_bomb(path: Path) -> None:
    """First-line, header-only heuristic: sum the ZIP central directory's declared
    file_size per entry (no decompression) BEFORE python-docx ever opens/decompresses
    the archive -- that call is what actually inflates the data, so this must run
    strictly earlier to be a guard at all.

    This is a heuristic, not a complete zip-bomb defense on its own -- a crafted
    archive could misreport or exploit parser-specific handling of the central
    directory. _enforce_extracted_size() below is the real backstop: it caps actual
    memory use post-parse regardless of what this check missed or how python-docx
    behaves internally.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            total_uncompressed = sum(info.file_size for info in zf.infolist())
    except zipfile.BadZipFile as exc:
        raise ExtractionError(f"File is not a valid DOCX/zip archive: {exc}") from exc

    if total_uncompressed > settings.MAX_DOCX_UNCOMPRESSED_BYTES:
        raise ExtractionError(
            f"DOCX would decompress to {total_uncompressed} bytes, exceeding the "
            f"maximum allowed ({settings.MAX_DOCX_UNCOMPRESSED_BYTES} bytes); "
            "rejected as a likely zip bomb."
        )


class TxtLoader(DocumentLoader):
    def load(self, path: Path) -> LoadedDocument:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ExtractionError(f"Failed to read text file: {exc}") from exc
        text = _enforce_extracted_size(text)
        return LoadedDocument(
            text=text,
            metadata={"char_count": len(text), "line_count": text.count("\n") + 1},
        )


class MarkdownLoader(DocumentLoader):
    def load(self, path: Path) -> LoadedDocument:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ExtractionError(f"Failed to read markdown file: {exc}") from exc
        text = _enforce_extracted_size(text)

        title = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break

        metadata: Dict[str, Any] = {"char_count": len(text), "line_count": text.count("\n") + 1}
        if title:
            metadata["title"] = title
        return LoadedDocument(text=text, metadata=metadata)


class PdfLoader(DocumentLoader):
    def load(self, path: Path) -> LoadedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ExtractionError("pypdf is required to extract PDF files") from exc

        try:
            reader = PdfReader(str(path))
            # len(reader.pages) only walks the page tree/xref -- it does not decode any
            # page's content stream -- so this check is cheap and runs before the
            # actual (expensive) per-page text extraction below.
            page_count = len(reader.pages)
            if page_count > settings.MAX_PDF_PAGES:
                raise ExtractionError(
                    f"PDF has {page_count} pages, exceeding the maximum allowed "
                    f"({settings.MAX_PDF_PAGES}); rejected to avoid excessive CPU use."
                )

            pages_text = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(pages_text)
            text = _enforce_extracted_size(text)

            info = reader.metadata or {}
            metadata: Dict[str, Any] = {
                "page_count": page_count,
                "title": getattr(info, "title", None),
                "author": getattr(info, "author", None),
            }
            metadata = {k: v for k, v in metadata.items() if v}
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"Failed to extract PDF content: {exc}") from exc
        return LoadedDocument(text=text, metadata=metadata)


class DocxLoader(DocumentLoader):
    def load(self, path: Path) -> LoadedDocument:
        try:
            import docx
        except ImportError as exc:
            raise ExtractionError("python-docx is required to extract DOCX files") from exc

        # Must run before docx.Document() -- that call is what actually decompresses
        # the archive; this check has to happen strictly earlier to be a guard at all.
        _check_docx_not_a_zip_bomb(path)

        try:
            document = docx.Document(str(path))
            paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
            text = "\n\n".join(paragraphs)
            text = _enforce_extracted_size(text)

            props = document.core_properties
            metadata: Dict[str, Any] = {
                "paragraph_count": len(paragraphs),
                "title": props.title or None,
                "author": props.author or None,
            }
            metadata = {k: v for k, v in metadata.items() if v}
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"Failed to extract DOCX content: {exc}") from exc
        return LoadedDocument(text=text, metadata=metadata)


_LOADERS: Dict[str, DocumentLoader] = {
    ".txt": TxtLoader(),
    ".md": MarkdownLoader(),
    ".markdown": MarkdownLoader(),
    ".pdf": PdfLoader(),
    ".docx": DocxLoader(),
}


def get_loader(extension: str) -> DocumentLoader:
    loader = _LOADERS.get(extension.lower())
    if loader is None:
        raise ExtractionError(f"No loader registered for extension '{extension}'")
    return loader
