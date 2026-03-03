"""
Tests for alibaba_coding tool call parser re-exports.

Verifies that the alibaba_coding parser module correctly re-exports
all functions from the chutes canonical parser, and that the
is_native_tool_call_model alias works correctly.
"""

from litellm.llms.alibaba_coding.chat.tool_call_parser import (
    TOOL_CALL_ARGUMENT_BEGIN,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    TOOL_CALLS_SECTION_BEGIN,
    TOOL_CALLS_SECTION_END,
    THINK_START_TAG,
    THINK_END_TAG,
    THINK_TAG_PATTERN,
    TOOL_CALL_PATTERN,
    MAX_SPECIAL_TOKEN_LENGTH,
    TOOL_CALL_FIELDS,
    deduplicate_reasoning_parts,
    extract_think_content,
    extract_think_content_complete,
    extract_tool_call_id_parts,
    has_think_end_tag,
    has_think_start_tag,
    has_tool_call_tokens,
    is_kimi_k2_model,
    is_native_tool_call_model,
    parse_tool_calls_from_content,
    parse_tool_calls_from_message,
    strip_native_tool_tokens,
    strip_think_tags,
)
from litellm.llms.chutes.chat.kimi_k2_tool_call_parser import (
    is_kimi_k2_model as chutes_is_kimi_k2_model,
    parse_tool_calls_from_content as chutes_parse_tool_calls_from_content,
)


def test_is_native_tool_call_model():
    """Test is_native_tool_call_model alias works for Kimi K2 models."""
    assert is_native_tool_call_model("moonshotai/Kimi-K2") is True
    assert is_native_tool_call_model("KIMI-k2-instruct") is True
    assert is_native_tool_call_model("kimi-k2.5") is True
    assert is_native_tool_call_model("kimi-1") is False
    assert is_native_tool_call_model("gpt-4o") is False


def test_is_native_tool_call_model_is_alias_of_is_kimi_k2_model():
    """Test that is_native_tool_call_model is the same function as is_kimi_k2_model from chutes."""
    assert is_native_tool_call_model is is_kimi_k2_model
    assert is_kimi_k2_model is chutes_is_kimi_k2_model


def test_parse_tool_calls_from_content_is_same_as_chutes():
    """Test that parse_tool_calls_from_content is the same function from chutes."""
    assert parse_tool_calls_from_content is chutes_parse_tool_calls_from_content


def test_parse_tool_calls_from_content_with_tokens():
    """Test parsing tool calls from content with native tokens."""
    content = (
        "Before tool call\n"
        f"{TOOL_CALLS_SECTION_BEGIN}\n"
        f"{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}"
        f'{{"city":"Tokyo"}}{TOOL_CALL_END}\n'
        f"{TOOL_CALLS_SECTION_END}\n"
        "After tool call"
    )

    tool_calls, cleaned = parse_tool_calls_from_content(content)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "functions.get_weather:0"
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == '{"city":"Tokyo"}'
    assert "Before tool call" in cleaned
    assert "After tool call" in cleaned
    assert TOOL_CALL_BEGIN not in cleaned


def test_strip_native_tool_tokens_cleans_content():
    """Test stripping native tool tokens from content."""
    content = (
        "Result starts. "
        f"{TOOL_CALLS_SECTION_BEGIN}"
        f'{TOOL_CALL_BEGIN}functions.lookup:1{TOOL_CALL_ARGUMENT_BEGIN}{{"id":123}}{TOOL_CALL_END}'
        f"{TOOL_CALLS_SECTION_END}"
        " Result ends."
    )

    cleaned = strip_native_tool_tokens(content)

    assert cleaned == "Result starts.  Result ends."
    assert TOOL_CALLS_SECTION_BEGIN not in cleaned
    assert TOOL_CALL_BEGIN not in cleaned


def test_extract_think_content_complete():
    """Test extracting think content with complete tags."""
    content = "<think>I need to analyze this</think>Here is the answer"
    thinking, remaining = extract_think_content_complete(content)

    assert thinking == "I need to analyze this"
    assert remaining == "Here is the answer"


def test_extract_think_content_complete_unclosed():
    """Test extracting think content with unclosed tag."""
    content = "<think>Still thinking about this..."
    thinking, remaining = extract_think_content_complete(content)

    assert thinking == "Still thinking about this..."
    assert remaining is None


def test_extract_think_content():
    """Test simple think content extraction."""
    content = "Before <think>reasoning here</think> after"
    thinking, cleaned = extract_think_content(content)

    assert thinking == "reasoning here"
    assert "Before" in cleaned
    assert "after" in cleaned


def test_has_think_end_tag():
    """Test detecting think end tag."""
    assert has_think_end_tag("some </think> content") is True
    assert has_think_end_tag("no end tag here") is False


def test_parse_tool_calls_from_message():
    """Test parsing tool calls from a message dict with multiple fields."""
    message = {
        "content": f"{TOOL_CALL_BEGIN}functions.search:0{TOOL_CALL_ARGUMENT_BEGIN}"
        f'{{"q":"test"}}{TOOL_CALL_END}',
        "reasoning": "Some reasoning text",
    }

    tool_calls, updated = parse_tool_calls_from_message(message)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "search"
    assert updated["reasoning"] == "Some reasoning text"


def test_deduplicate_reasoning_parts():
    """Test deduplication of reasoning parts."""
    parts = ["same reasoning", "different reasoning", "same reasoning"]
    result = deduplicate_reasoning_parts(parts)

    assert result is not None
    assert result.count("same reasoning") == 1
    assert "different reasoning" in result


def test_deduplicate_reasoning_parts_empty():
    """Test deduplication with empty list."""
    assert deduplicate_reasoning_parts([]) is None
    assert deduplicate_reasoning_parts(["", "  "]) is None


def test_extract_tool_call_id_parts():
    """Test extracting function name and index from tool call ID."""
    name, idx = extract_tool_call_id_parts("functions.get_weather:0")
    assert name == "get_weather"
    assert idx == 0

    name, idx = extract_tool_call_id_parts("functions.complex_func:5")
    assert name == "complex_func"
    assert idx == 5


def test_all_exports_present():
    """Test that all expected exports are available from the re-export module."""
    import litellm.llms.alibaba_coding.chat.tool_call_parser as mod

    expected = [
        "MAX_SPECIAL_TOKEN_LENGTH",
        "THINK_END_TAG",
        "THINK_START_TAG",
        "THINK_TAG_PATTERN",
        "TOOL_CALL_ARGUMENT_BEGIN",
        "TOOL_CALL_BEGIN",
        "TOOL_CALL_END",
        "TOOL_CALL_FIELDS",
        "TOOL_CALL_PATTERN",
        "TOOL_CALLS_SECTION_BEGIN",
        "TOOL_CALLS_SECTION_END",
        "deduplicate_reasoning_parts",
        "extract_think_content",
        "extract_think_content_complete",
        "extract_tool_call_id_parts",
        "has_think_end_tag",
        "has_think_start_tag",
        "has_tool_call_tokens",
        "is_kimi_k2_model",
        "is_native_tool_call_model",
        "parse_tool_calls_from_content",
        "parse_tool_calls_from_message",
        "strip_native_tool_tokens",
        "strip_think_tags",
    ]
    for name in expected:
        assert hasattr(mod, name), f"Missing export: {name}"
