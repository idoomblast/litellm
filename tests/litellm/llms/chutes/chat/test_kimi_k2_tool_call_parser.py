"""
Unit tests for Kimi K2 native tool call parser.

Tests the parsing logic for Kimi K2's special token format for tool calls.
"""

import pytest

from litellm.llms.chutes.chat.kimi_k2_tool_call_parser import (
    extract_think_content,
    extract_tool_call_id_parts,
    has_tool_call_tokens,
    is_kimi_k2_model,
    parse_tool_calls_from_content,
    parse_tool_calls_from_message,
    strip_native_tool_tokens,
    strip_think_tags,
    TOOL_CALLS_SECTION_BEGIN,
    TOOL_CALLS_SECTION_END,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    TOOL_CALL_ARGUMENT_BEGIN,
)


class TestHasToolCallTokens:
    """Test has_tool_call_tokens function."""

    def test_content_with_tool_call_tokens(self):
        """Test detection of tool call tokens in content."""
        content = f"Some text {TOOL_CALLS_SECTION_BEGIN}tool calls{TOOL_CALLS_SECTION_END}"
        assert has_tool_call_tokens(content) is True

    def test_content_without_tool_call_tokens(self):
        """Test content without tool call tokens."""
        content = "Just regular content without any tool calls"
        assert has_tool_call_tokens(content) is False

    def test_empty_content(self):
        """Test empty content."""
        assert has_tool_call_tokens("") is False

    def test_partial_token(self):
        """Test partial token (incomplete) doesn't trigger detection."""
        content = "<|tool_calls_section"  # Incomplete token
        assert has_tool_call_tokens(content) is False


class TestExtractToolCallIdParts:
    """Test extract_tool_call_id_parts function."""

    def test_valid_tool_call_id(self):
        """Test extracting parts from a valid tool call ID."""
        func_name, idx = extract_tool_call_id_parts("functions.get_weather:0")
        assert func_name == "get_weather"
        assert idx == 0

    def test_tool_call_id_with_higher_index(self):
        """Test tool call ID with index > 0."""
        func_name, idx = extract_tool_call_id_parts("functions.search_database:42")
        assert func_name == "search_database"
        assert idx == 42

    def test_tool_call_id_with_underscore_in_name(self):
        """Test tool call ID with underscores in function name."""
        func_name, idx = extract_tool_call_id_parts("functions.get_user_info:1")
        assert func_name == "get_user_info"
        assert idx == 1

    def test_invalid_prefix(self):
        """Test tool call ID with invalid prefix."""
        with pytest.raises(ValueError, match="Invalid tool call ID format"):
            extract_tool_call_id_parts("invalid.get_weather:0")

    def test_missing_index(self):
        """Test tool call ID without index."""
        with pytest.raises(ValueError, match="missing index"):
            extract_tool_call_id_parts("functions.get_weather")

    def test_invalid_index(self):
        """Test tool call ID with non-numeric index."""
        with pytest.raises(ValueError, match="invalid index"):
            extract_tool_call_id_parts("functions.get_weather:abc")


class TestParseToolCallsFromContent:
    """Test parse_tool_calls_from_content function."""

    def test_single_tool_call(self):
        """Test parsing a single tool call."""
        content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        tool_calls, remaining = parse_tool_calls_from_content(content)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "functions.get_weather:0"
        assert tool_calls[0].type == "function"
        assert tool_calls[0].function.name == "get_weather"
        assert tool_calls[0].function.arguments == '{"city": "Beijing"}'
        assert remaining == ""

    def test_multiple_tool_calls(self):
        """Test parsing multiple tool calls in one section."""
        content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f'{TOOL_CALL_BEGIN}functions.get_time:1{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"timezone": "UTC"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        tool_calls, remaining = parse_tool_calls_from_content(content)

        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0].id == "functions.get_weather:0"
        assert tool_calls[0].function.name == "get_weather"
        assert tool_calls[1].id == "functions.get_time:1"
        assert tool_calls[1].function.name == "get_time"

    def test_mixed_content_with_tool_calls(self):
        """Test parsing content with text before/after tool call section."""
        content = (
            "Here is some text before.\n"
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}\n"
            "And some text after."
        )

        tool_calls, remaining = parse_tool_calls_from_content(content)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert "Here is some text before." in remaining
        assert "And some text after." in remaining
        assert TOOL_CALLS_SECTION_BEGIN not in remaining

    def test_no_tool_calls(self):
        """Test parsing content without tool calls."""
        content = "Just regular content without any tool calls"

        tool_calls, remaining = parse_tool_calls_from_content(content)

        assert tool_calls is None
        assert remaining == content

    def test_empty_content(self):
        """Test parsing empty content."""
        tool_calls, remaining = parse_tool_calls_from_content("")

        assert tool_calls is None
        assert remaining == ""

    def test_complex_json_arguments(self):
        """Test parsing tool call with nested JSON arguments."""
        args = '{"query": {"type": "search", "filters": {"date": "2024-01-01"}}}'
        content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f"{TOOL_CALL_BEGIN}functions.search:0{TOOL_CALL_ARGUMENT_BEGIN}"
            f"{args}{TOOL_CALL_END}\n"
            f"{TOOL_CALLS_SECTION_END}"
        )

        tool_calls, remaining = parse_tool_calls_from_content(content)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.arguments == args

    def test_malformed_tool_call_id_skipped(self):
        """Test that malformed tool call IDs are gracefully skipped."""
        # The regex requires "functions." prefix, so this won't match
        content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}invalid_format{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        tool_calls, remaining = parse_tool_calls_from_content(content)

        # Malformed tool call should be skipped (regex doesn't match)
        assert tool_calls is None
        # Content remains unchanged since the regex pattern didn't match
        # (the section exists but contains no valid tool calls)
        assert remaining == content

    def test_malformed_tool_call_id_with_valid_prefix_skipped(self):
        """Test that tool calls with valid prefix but invalid format are skipped."""
        # This has the functions. prefix but invalid format (no index)
        content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        tool_calls, remaining = parse_tool_calls_from_content(content)

        # This won't match the regex pattern (requires :digit at end)
        assert tool_calls is None

    def test_whitespace_handling(self):
        """Test parsing with various whitespace patterns."""
        content = (
            f"  {TOOL_CALLS_SECTION_BEGIN}  \n"
            f"  {TOOL_CALL_BEGIN}  functions.get_weather:0  "
            f"{TOOL_CALL_ARGUMENT_BEGIN}  "
            f'{{"city": "Beijing"}}  {TOOL_CALL_END}  \n'
            f"  {TOOL_CALLS_SECTION_END}  "
        )

        tool_calls, remaining = parse_tool_calls_from_content(content)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"


class TestParseToolCallsFromMessage:
    """Test parse_tool_calls_from_message function."""

    def test_tool_calls_in_content(self):
        """Test parsing tool calls from content field."""
        message = {
            "role": "assistant",
            "content": (
                f"{TOOL_CALLS_SECTION_BEGIN}\n"
                f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
                f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
                f"{TOOL_CALLS_SECTION_END}"
            ),
        }

        tool_calls, updated_message = parse_tool_calls_from_message(message)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        # Content should be cleaned (empty -> None)
        assert updated_message["content"] is None

    def test_tool_calls_in_reasoning_content(self):
        """Test parsing tool calls from reasoning_content field."""
        message = {
            "role": "assistant",
            "content": "Regular response",
            "reasoning_content": (
                f"{TOOL_CALLS_SECTION_BEGIN}\n"
                f'{TOOL_CALL_BEGIN}functions.analyze:0{TOOL_CALL_ARGUMENT_BEGIN}'
                f'{{"data": "test"}}{TOOL_CALL_END}\n'
                f"{TOOL_CALLS_SECTION_END}"
            ),
        }

        tool_calls, updated_message = parse_tool_calls_from_message(message)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "analyze"
        assert updated_message["content"] == "Regular response"
        assert updated_message["reasoning_content"] is None

    def test_tool_calls_in_thinking(self):
        """Test parsing tool calls from thinking field."""
        message = {
            "role": "assistant",
            "content": "I'll help you with that.",
            "thinking": (
                f"Let me think about this...\n"
                f"{TOOL_CALLS_SECTION_BEGIN}\n"
                f'{TOOL_CALL_BEGIN}functions.think:0{TOOL_CALL_ARGUMENT_BEGIN}'
                f'{{"topic": "weather"}}{TOOL_CALL_END}\n'
                f"{TOOL_CALLS_SECTION_END}"
            ),
        }

        tool_calls, updated_message = parse_tool_calls_from_message(message)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "think"
        assert "Let me think about this..." in updated_message["thinking"]

    def test_tool_calls_in_multiple_fields(self):
        """Test parsing tool calls from multiple fields simultaneously."""
        message = {
            "role": "assistant",
            "content": (
                f"{TOOL_CALLS_SECTION_BEGIN}\n"
                f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
                f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
                f"{TOOL_CALLS_SECTION_END}"
            ),
            "reasoning_content": (
                f"{TOOL_CALLS_SECTION_BEGIN}\n"
                f'{TOOL_CALL_BEGIN}functions.analyze:1{TOOL_CALL_ARGUMENT_BEGIN}'
                f'{{"data": "test"}}{TOOL_CALL_END}\n'
                f"{TOOL_CALLS_SECTION_END}"
            ),
        }

        tool_calls, updated_message = parse_tool_calls_from_message(message)

        assert tool_calls is not None
        assert len(tool_calls) == 2
        # Both tool calls should be collected
        func_names = {tc.function.name for tc in tool_calls}
        assert "get_weather" in func_names
        assert "analyze" in func_names

    def test_no_tool_calls_in_any_field(self):
        """Test message without tool calls in any field."""
        message = {
            "role": "assistant",
            "content": "Regular response",
            "reasoning_content": "Just thinking normally",
        }

        tool_calls, updated_message = parse_tool_calls_from_message(message)

        assert tool_calls is None
        assert updated_message["content"] == "Regular response"
        assert updated_message["reasoning_content"] == "Just thinking normally"

    def test_empty_content_after_removal(self):
        """Test that empty content after token removal becomes None."""
        message = {
            "role": "assistant",
            "content": (
                f"{TOOL_CALLS_SECTION_BEGIN}\n"
                f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
                f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
                f"{TOOL_CALLS_SECTION_END}"
            ),
        }

        tool_calls, updated_message = parse_tool_calls_from_message(message)

        assert tool_calls is not None
        assert updated_message["content"] is None

    def test_mixed_content_preserved(self):
        """Test that text before/after tool calls is preserved."""
        message = {
            "role": "assistant",
            "content": (
                "Before the tool calls.\n"
                f"{TOOL_CALLS_SECTION_BEGIN}\n"
                f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
                f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
                f"{TOOL_CALLS_SECTION_END}\n"
                "After the tool calls."
            ),
        }

        tool_calls, updated_message = parse_tool_calls_from_message(message)

        assert tool_calls is not None
        assert "Before the tool calls." in updated_message["content"]
        assert "After the tool calls." in updated_message["content"]


class TestIsKimiK2Model:
    """Test is_kimi_k2_model function."""

    def test_kimi_k2_model(self):
        """Test detection of Kimi K2 model."""
        assert is_kimi_k2_model("moonshotai/Kimi-K2") is True
        assert is_kimi_k2_model("chutes/moonshotai/Kimi-K2") is True
        assert is_kimi_k2_model("kimi-k2-instruct") is True
        assert is_kimi_k2_model("KIMI-K2") is True  # Case insensitive

    def test_non_kimi_k2_model(self):
        """Test non-Kimi K2 models."""
        assert is_kimi_k2_model("gpt-4") is False
        assert is_kimi_k2_model("claude-3") is False
        assert is_kimi_k2_model("kimi-1") is False  # Kimi but not K2
        assert is_kimi_k2_model("some-k2-model") is False  # K2 but not Kimi

    def test_model_with_version_suffix(self):
        """Test Kimi K2 model with version suffix."""
        assert is_kimi_k2_model("Kimi-K2-v1") is True
        assert is_kimi_k2_model("kimi-k2-instruct-2024") is True


class TestExtractThinkContent:
    """Test extract_think_content function for <think>...</think> tags."""

    def test_simple_think_tags(self):
        """Test extracting content from simple think tags."""
        content = "<think>Let me analyze this problem.</think> Here is my response."
        thinking, remaining = extract_think_content(content)

        assert thinking == "Let me analyze this problem."
        assert remaining == "Here is my response."

    def test_think_tags_with_spaces(self):
        """Test extracting content from think tags with spaces around content."""
        content = " <think> Let me think about this... </think> Here's my answer."
        thinking, remaining = extract_think_content(content)

        assert thinking == "Let me think about this..."
        assert remaining == "Here's my answer."

    def test_no_think_tags(self):
        """Test content without think tags."""
        content = "Just a regular response without any thinking."
        thinking, remaining = extract_think_content(content)

        assert thinking is None
        assert remaining == content

    def test_empty_content(self):
        """Test empty content."""
        thinking, remaining = extract_think_content("")

        assert thinking is None
        assert remaining == ""

    def test_multiline_think_content(self):
        """Test multiline content inside think tags."""
        content = """<think>
First I'll analyze the request.
Then I'll formulate a response.
Finally I'll check my work.
</think>
Here is my response after thinking."""
        thinking, remaining = extract_think_content(content)

        assert thinking is not None
        assert "First I'll analyze" in thinking
        assert "Finally I'll check" in thinking
        assert "Here is my response" in remaining

    def test_empty_think_tags(self):
        """Test empty think tags."""
        content = "<think></think> Some response."
        thinking, remaining = extract_think_content(content)

        assert thinking is None  # Empty think content should be None
        assert remaining == "Some response."


class TestStripThinkTags:
    """Test strip_think_tags function."""

    def test_strip_simple_think_tags(self):
        """Test stripping simple think tags."""
        content = "<think>Internal reasoning</think> Final answer."
        result = strip_think_tags(content)
        assert result == "Final answer."

    def test_strip_empty_content(self):
        """Test stripping from empty content."""
        assert strip_think_tags("") == ""

    def test_no_think_tags(self):
        """Test content without think tags is unchanged."""
        content = "Just regular content."
        assert strip_think_tags(content) == "Just regular content."

    def test_only_think_tags(self):
        """Test content with only think tags."""
        content = "<think>Only thinking, no response.</think>"
        result = strip_think_tags(content)
        assert result == ""


class TestStripNativeToolTokens:
    """Test strip_native_tool_tokens function."""

    def test_strip_complete_tool_call(self):
        """Test stripping a complete tool call pattern."""
        content = (
            "Some text before. "
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}'
            " Some text after."
        )
        result = strip_native_tool_tokens(content)

        assert "Some text before." in result
        assert "Some text after." in result
        assert TOOL_CALL_BEGIN not in result
        assert TOOL_CALL_END not in result

    def test_strip_section_markers(self):
        """Test stripping section markers."""
        content = f"{TOOL_CALLS_SECTION_BEGIN} some content {TOOL_CALLS_SECTION_END}"
        result = strip_native_tool_tokens(content)

        assert TOOL_CALLS_SECTION_BEGIN not in result
        assert TOOL_CALLS_SECTION_END not in result
        assert "some content" in result

    def test_strip_empty_content(self):
        """Test stripping from empty content."""
        assert strip_native_tool_tokens("") == ""

    def test_no_tokens_unchanged(self):
        """Test content without tokens is unchanged."""
        content = "Just regular content."
        assert strip_native_tool_tokens(content) == "Just regular content."


class TestChutesFormatParsing:
    """Test parsing Chutes-specific format (no section wrappers, spaces around tokens)."""

    def test_chutes_format_single_tool_call(self):
        """Test parsing Chutes format with spaces around tokens."""
        # Real Chutes format from debug files - has spaces around tokens
        content = (
            f"{TOOL_CALL_BEGIN} functions.read_file:0 {TOOL_CALL_ARGUMENT_BEGIN} "
            f'{{"filePath": "/path/to/file"}} {TOOL_CALL_END}'
        )

        tool_calls, remaining = parse_tool_calls_from_content(content)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "functions.read_file:0"
        assert tool_calls[0].function.name == "read_file"
        assert '"/path/to/file"' in tool_calls[0].function.arguments
        assert remaining == ""

    def test_chutes_format_multiple_tool_calls(self):
        """Test parsing multiple Chutes format tool calls."""
        content = (
            f"{TOOL_CALL_BEGIN} functions.list_dir:0 {TOOL_CALL_ARGUMENT_BEGIN} "
            f'{{"path": "/"}} {TOOL_CALL_END}'
            f" {TOOL_CALL_BEGIN} functions.read_file:1 {TOOL_CALL_ARGUMENT_BEGIN} "
            f'{{"filePath": "/test.txt"}} {TOOL_CALL_END}'
        )

        tool_calls, remaining = parse_tool_calls_from_content(content)

        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0].function.name == "list_dir"
        assert tool_calls[1].function.name == "read_file"

    def test_both_formats_are_supported(self):
        """Test that both official and Chutes formats parse correctly."""
        # Official format (with section wrappers, no extra spaces)
        official_content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        # Chutes format (no section wrappers, with spaces)
        chutes_content = (
            f"{TOOL_CALL_BEGIN} functions.get_weather:0 {TOOL_CALL_ARGUMENT_BEGIN} "
            f'{{"city": "Beijing"}} {TOOL_CALL_END}'
        )

        official_calls, _ = parse_tool_calls_from_content(official_content)
        chutes_calls, _ = parse_tool_calls_from_content(chutes_content)

        assert official_calls is not None
        assert chutes_calls is not None
        assert len(official_calls) == 1
        assert len(chutes_calls) == 1
        assert official_calls[0].function.name == chutes_calls[0].function.name
