from __future__ import annotations

import re

# Structural prompt-injection defenses only -- never keyword/content-based. A
# legitimate document that discusses "risk factors" or contains the word
# "instructions" is completely unaffected; only HTML/XML markup and hidden Unicode
# characters are removed. Markdown syntax (|, #, *, [], (), numbered "Item 1A."
# section headers) is never touched, since none of it is <...>-tag-shaped -- the
# common case (PDF/DOCX/TXT/Markdown ingestion, none of which emit HTML) passes
# through this function completely unaffected by the tag-stripping step.

# <script>/<style> blocks: tag AND content removed -- these are executable/
# presentational, never evidentiary text.
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)

# One or more consecutive HTML/XML tags (optionally separated by whitespace),
# collapsed as a single run to exactly one space -- e.g. "</td><td>" becomes one
# space, not two. This only ever fires where a tag actually was; it never touches
# whitespace in tag-free text, so pre-existing spacing elsewhere is untouched.
_TAG_RUN_RE = re.compile(r"(?:<[^>]+>\s*)+")

# Zero-width and bidirectional-override code points: a known technique for hiding
# injected instructions from human review while an LLM still reads them. Built from
# explicit integer code points (not embedded Unicode escapes) so the source stays
# plain ASCII and auditable -- no invisible characters live in this file itself.
_ZERO_WIDTH_CODEPOINTS = [
    0x200B,  # ZERO WIDTH SPACE
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0x200E,  # LEFT-TO-RIGHT MARK
    0x200F,  # RIGHT-TO-LEFT MARK
    0x202A,  # LEFT-TO-RIGHT EMBEDDING
    0x202B,  # RIGHT-TO-LEFT EMBEDDING
    0x202C,  # POP DIRECTIONAL FORMATTING
    0x202D,  # LEFT-TO-RIGHT OVERRIDE
    0x202E,  # RIGHT-TO-LEFT OVERRIDE
    0x2066,  # LEFT-TO-RIGHT ISOLATE
    0x2067,  # RIGHT-TO-LEFT ISOLATE
    0x2068,  # FIRST STRONG ISOLATE
    0x2069,  # POP DIRECTIONAL ISOLATE
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
]
_ZERO_WIDTH_CHARS = "".join(chr(cp) for cp in _ZERO_WIDTH_CODEPOINTS)
_ZERO_WIDTH_RE = re.compile("[" + re.escape(_ZERO_WIDTH_CHARS) + "]")

# Non-printable control characters, excluding \n, \r, \t (preserves line/paragraph
# structure and normal tab-indentation).
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Runs of 2+ horizontal whitespace (spaces/tabs only -- \n and \r are deliberately
# excluded from this class) collapsed to a single space. Tag/script removal above can
# leave two spaces adjacent to each other (e.g. "<td> 100</td>" -> "  100"); this pass
# normalizes that. It never touches newlines, so Markdown paragraph/section breaks and
# blank-line-separated formatting are always preserved exactly.
_HORIZONTAL_WHITESPACE_RUN_RE = re.compile(r"[ \t]{2,}")


def sanitize_retrieved_text(text: str) -> str:
    """Neutralize structural prompt-injection vectors in untrusted retrieved/uploaded
    text (KB search results, web search results, ingested documents, memory context)
    while preserving all evidentiary content: prose, numbers, currency/percent
    symbols, Markdown syntax (tables, headers, lists, links), citation markers, and
    URLs.

    Removes only:
    - <script>/<style> blocks (tag + content).
    - All other HTML/XML tags (collapsed to a single space per run, never deleted
      outright -- prevents adjacent tag-separated tokens, e.g. table cells, from
      being glued into one corrupted token).
    - Zero-width and bidirectional-override Unicode characters.
    - Non-printable control characters other than \\n/\\r/\\t.

    Horizontal whitespace runs (spaces/tabs) left behind by the above are collapsed
    to a single space; newlines are never collapsed, so Markdown paragraph/section
    breaks always survive.

    Never raises -- returns the original text unchanged on any internal error.
    """
    if not text:
        return text or ""
    try:
        cleaned = _SCRIPT_STYLE_RE.sub(" ", text)
        cleaned = _TAG_RUN_RE.sub(" ", cleaned)
        cleaned = _ZERO_WIDTH_RE.sub("", cleaned)
        cleaned = _CONTROL_CHAR_RE.sub("", cleaned)
        cleaned = _HORIZONTAL_WHITESPACE_RUN_RE.sub(" ", cleaned)
        return cleaned.strip()
    except Exception:
        return text
