from __future__ import annotations

from content_security import sanitize_retrieved_text


def test_strips_script_block_tag_and_content():
    assert sanitize_retrieved_text("before<script>alert(1)</script>after") == "before after"


def test_strips_style_block_tag_and_content():
    assert sanitize_retrieved_text("<style>body{color:red}</style>") == ""


def test_strips_other_html_tags_but_keeps_text_separated():
    assert sanitize_retrieved_text("<b>bold</b> and <i>italic</i>") == "bold and italic"


def test_html_in_table_case_preserves_and_separates_numbers():
    # Deleting tags outright would glue "100" and "200" into "100200"; the
    # sanitizer must replace each tag run with a single space instead.
    assert sanitize_retrieved_text("<td>100</td><td>200</td>") == "100 200"
    assert (
        sanitize_retrieved_text("<table><tr><td>100</td><td>200</td></tr></table>")
        == "100 200"
    )


def test_strips_zero_width_characters():
    # Built from explicit code points (not literal invisible characters in this
    # source file) so the test file itself stays plain ASCII and auditable.
    zero_width_space = chr(0x200B)
    text = "hello" + zero_width_space + "world"
    assert sanitize_retrieved_text(text) == "helloworld"


def test_strips_bidi_override_characters():
    rtl_override = chr(0x202E)
    pop_directional_formatting = chr(0x202C)
    text = rtl_override + "malicious" + pop_directional_formatting
    assert sanitize_retrieved_text(text) == "malicious"


def test_strips_non_printable_control_characters():
    text = "abc" + chr(0x00) + chr(0x07) + "def"
    assert sanitize_retrieved_text(text) == "abcdef"


def test_preserves_plain_prose_byte_for_byte():
    text = "The company's revenue grew 12% year-over-year to $45.2 million in Q3."
    assert sanitize_retrieved_text(text) == text


def test_preserves_markdown_table_byte_for_byte():
    text = "| Metric | 2023 | 2024 |\n|---|---|---|\n| Revenue | 100 | 120 |"
    assert sanitize_retrieved_text(text) == text


def test_preserves_markdown_headers_bold_and_bullets_byte_for_byte():
    text = "# Overview\n\n**Key risk:** customer concentration.\n\n- Risk 1\n- Risk 2"
    assert sanitize_retrieved_text(text) == text


def test_preserves_sec_style_section_header_byte_for_byte():
    text = "Item 1A. Risk Factors"
    assert sanitize_retrieved_text(text) == text


def test_preserves_citation_markers_byte_for_byte():
    text = "Revenue grew significantly [E1]."
    assert sanitize_retrieved_text(text) == text


def test_preserves_urls_byte_for_byte():
    text = "See https://example.com/filing for details."
    assert sanitize_retrieved_text(text) == text


def test_preserves_unmatched_angle_bracket_as_inequality_notation():
    # A lone "<" with no closing ">" is not tag-shaped and must survive, since
    # it's ordinary financial notation ("less than"), not markup.
    text = "Revenue < COGS this quarter."
    assert sanitize_retrieved_text(text) == text


def test_idempotent_on_already_clean_text():
    text = "Clean evidence text with no markup or hidden characters."
    once = sanitize_retrieved_text(text)
    twice = sanitize_retrieved_text(once)
    assert once == twice == text


def test_empty_and_none_input_returns_empty_string():
    assert sanitize_retrieved_text("") == ""
    assert sanitize_retrieved_text(None) == ""  # type: ignore[arg-type]


def test_never_raises_on_malformed_input():
    # Not a str at all -- the function must degrade gracefully rather than crash.
    result = sanitize_retrieved_text(b"\x00\x01binary")  # type: ignore[arg-type]
    assert result == b"\x00\x01binary"
