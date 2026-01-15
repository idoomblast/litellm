"""
Unit tests for Kimi-K2 tool call parsing.

Tests validate the KimiK2ToolParser class which handles parsing of
native Kimi-K2 special token tool call format.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import pytest
import json

from litellm.llms.chutes.chat.kimi_k2_parser import KimiK2ToolParser


class TestKimiK2ToolParser:
    """Test class for Kimi-K2 tool call parsing functionality"""

    def test_extract_tool_calls_basic(self):
        """Test basic tool call extraction from Kimi-K2 format"""
        parser = KimiK2ToolParser()

        content = """
        Some reasoning text here
        <|tool_calls_section_begin|>
        <|tool_call_begin|> functions.Task:0
        <|tool_call_argument_begin|> {"description": "Test", "prompt": "Hello"}
        <|tool_call_end|>
        <|tool_calls_section_end|>
        """

        tool_calls = parser.extract_tool_calls(content)

        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call_0"
        assert tool_calls[0]["type"] == "function"
        assert tool_calls[0]["function"]["name"] == "functions.Task:0"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {
            "description": "Test",
            "prompt": "Hello"
        }

    def test_extract_tool_calls_multiple(self):
        """Test multiple tool calls extraction"""
        parser = KimiK2ToolParser()

        content = """
        <|tool_calls_section_begin|>
        <|tool_call_begin|> get_weather
        <|tool_call_argument_begin|> {"city": "Jakarta"}
        <|tool_call_end|>
        <|tool_call_begin|> get_time
        <|tool_call_argument_begin|> {"timezone": "UTC"}
        <|tool_call_end|>
        <|tool_calls_section_end|>
        """

        tool_calls = parser.extract_tool_calls(content)

        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[1]["function"]["name"] == "get_time"
        assert tool_calls[1]["id"] == "call_1"

    def test_extract_tool_calls_complex_arguments(self):
        """Test tool call extraction with complex (nested) arguments"""
        parser = KimiK2ToolParser()

        content = """
        <|tool_calls_section_begin|>
        <|tool_call_begin|> search_web
        <|tool_call_argument_begin|> {"query": "AI", "filters": {"date": "2025", "language": "en"}}
        <|tool_call_end|>
        <|tool_calls_section_end|>
        """

        tool_calls = parser.extract_tool_calls(content)

        assert len(tool_calls) == 1
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args["query"] == "AI"
        assert args["filters"]["date"] == "2025"
        assert args["filters"]["language"] == "en"

    def test_extract_tool_calls_no_marker(self):
        """Test that empty list is returned when no tool call markers"""
        parser = KimiK2ToolParser()

        content = "This is just normal text with no tool calls."

        tool_calls = parser.extract_tool_calls(content)

        assert tool_calls == []

    def test_extract_tool_calls_incomplete_format(self):
        """Test handling of incomplete tool call format"""
        parser = KimiK2ToolParser()

        content = "<|tool_calls_section_begin|> <|tool_call_begin|> test_func"

        tool_calls = parser.extract_tool_calls(content)

        # Should handle gracefully and return empty or partial results
        assert tool_calls == []

    def test_extract_reasoning_and_content(self):
        """Test extraction of reasoning content and main content"""
        parser = KimiK2ToolParser()

        content = """
        I will analyze the code structure first.
        <|tool_calls_section_begin|>
        <|tool_call_begin|> explore_codebase
        <|tool_call_argument_begin|> {"prompt": "Explore"}
        <|tool_call_end|>
        <|tool_calls_section_end|>
        Then I'll create the documentation.
        """

        reasoning, main_content = parser.extract_reasoning_content(content)

        assert reasoning == "I will analyze the code structure first."
        assert main_content == "Then I'll create the documentation."

    def test_extract_reasoning_only(self):
        """Test extraction when only reasoning exists (no main content)"""
        parser = KimiK2ToolParser()

        content = """
        Let me think about this...
        <|tool_calls_section_begin|>
        <|tool_call_begin|> calculate
        <|tool_call_argument_begin|> {"x": 1}
        <|tool_call_end|>
        <|tool_calls_section_end|>
        """

        reasoning, main_content = parser.extract_reasoning_content(content)

        assert reasoning is not None
        assert "Let me think about this..." in reasoning
        assert main_content is None

    def test_extract_reasoning_no_tool_calls(self):
        """Test with no tool calls - content should be main content"""
        parser = KimiK2ToolParser()

        content = "This is just regular text."

        reasoning, main_content = parser.extract_reasoning_content(content)

        assert reasoning is None
        assert main_content == "This is just regular text."

    def test_clean_tool_calls_from_content(self):
        """Test removal of tool call markers from content"""
        parser = KimiK2ToolParser()

        content = """
        Before tool calls.
        <|tool_calls_section_begin|>
        <|tool_call_begin|> func
        <|tool_call_argument_begin|> {}
        <|tool_call_end|>
        <|tool_calls_section_end|>
        After tool calls.
        """

        cleaned = parser.clean_tool_calls_from_content(content)

        assert cleaned is not None
        assert "Before tool calls." in cleaned
        assert "After tool calls." in cleaned
        assert "<|tool_calls_section_begin|>" not in cleaned

    def test_clean_tool_calls_only_after(self):
        """Test cleanup when content only exists after tool calls"""
        parser = KimiK2ToolParser()

        content = "<|tool_calls_section_begin|> <|tool_call_begin|> func <|tool_call_end|> <|tool_calls_section_end|> Some text after."

        cleaned = parser.clean_tool_calls_from_content(content)

        assert cleaned == "Some text after."

    def test_clean_tool_calls_none_result(self):
        """Test that None is returned when content is empty after cleanup"""
        parser = KimiK2ToolParser()

        content = "<|tool_calls_section_begin|> <|tool_call_end|> <|tool_calls_section_end|>"

        cleaned = parser.clean_tool_calls_from_content(content)

        assert cleaned is None

    def test_is_kimi_k2_model_true(self):
        """Test Kimi-K2 model detection - positive cases"""
        assert KimiK2ToolParser.is_kimi_k2_model("kimi-k2-thinking")
        assert KimiK2ToolParser.is_kimi_k2_model("kimi-k2-instruct")
        assert KimiK2ToolParser.is_kimi_k2_model("Kimi-K2-Thinking")
        assert KimiK2ToolParser.is_kimi_k2_model("KIMI-K2")
        assert KimiK2ToolParser.is_kimi_k2_model("kimi_k2")
        assert KimiK2ToolParser.is_kimi_k2_model("chutes/moonshotai/Kimi-K2-Instruct")

    def test_is_kimi_k2_model_false(self):
        """Test Kimi-K2 model detection - negative cases"""
        assert not KimiK2ToolParser.is_kimi_k2_model("gpt-4")
        assert not KimiK2ToolParser.is_kimi_k2_model("claude-3")
        assert not KimiK2ToolParser.is_kimi_k2_model("mini-max")
        assert not KimiK2ToolParser.is_kimi_k2_model("kimi-v1")  # Different model

    def test_has_native_tool_calls_true_reasoning(self):
        """Test detection of native tool calls in reasoning_content"""
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "I'll use a tool. <|tool_calls_section_begin|> <|tool_call_end|> <|tool_calls_section_end|>"
                }
            }]
        }

        assert KimiK2ToolParser.has_native_tool_calls(response) is True

    def test_has_native_tool_calls_true_content(self):
        """Test detection of native tool calls in content"""
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<|tool_calls_section_begin|> <|tool_call_end|> <|tool_calls_section_end|>",
                    "reasoning_content": None
                }
            }]
        }

        assert KimiK2ToolParser.has_native_tool_calls(response) is True

    def test_has_native_tool_calls_false(self):
        """Test no native tool calls detected"""
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "This is normal text.",
                    "reasoning_content": "Some reasoning."
                }
            }]
        }

        assert KimiK2ToolParser.has_native_tool_calls(response) is False

    def test_has_native_tool_calls_empty_choices(self):
        """Test with empty choices"""
        response = {"choices": []}

        assert KimiK2ToolParser.has_native_tool_calls(response) is False

    def test_has_native_tool_calls_no_choices(self):
        """Test with no choices key"""
        response = {}

        assert KimiK2ToolParser.has_native_tool_calls(response) is False

    def test_convenience_parse_function(self):
        """Test the convenience parse function"""
        from litellm.llms.chutes.chat.kimi_k2_parser import parse_kimi_k2_tool_calls

        content = """
        <|tool_calls_section_begin|>
        <|tool_call_begin|> test_func
        <|tool_call_argument_begin|> {"arg": "value"}
        <|tool_call_end|>
        <|tool_calls_section_end|>
        """

        tool_calls = parse_kimi_k2_tool_calls(content)

        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "test_func"

    def test_convenience_extract_reasoning_function(self):
        """Test the convenience extract_reasoning_and_content function"""
        from litellm.llms.chutes.chat.kimi_k2_parser import extract_kimi_k2_reasoning_and_content

        content = """
        Reasoning here.
        <|tool_calls_section_begin|>
        <|tool_call_begin|> func
        <|tool_call_argument_begin|> {}
        <|tool_call_end|>
        <|tool_calls_section_end|>
        Main content here.
        """

        reasoning, main_content = extract_kimi_k2_reasoning_and_content(content)

        assert reasoning == "Reasoning here."
        assert main_content == "Main content here."

    def test_real_world_example_from_user(self):
        """Test with the real-world example provided by the user"""
        parser = KimiK2ToolParser()

        reasoning_content = """ Saya akan menganalisis kode ini dan membuat file CLAUDE.md yang komprehensif. Mari saya mulai dengan menjelajahi struktur proyek dan memahami arsitekturnya. <|tool_calls_section_begin|> <|tool_call_begin|> functions.Task:0 <|tool_call_argument_begin|> {"description": "Explore codebase structure", "prompt": "Explore the codebase structure and identify:\n1. The main project type (what kind of project this is)\n2. Key directories and their purposes\n3. Configuration files that reveal build/test/lint setup\n4. Any existing documentation files (README.md, CONTRIBUTING.md, etc.)\n5. Any cursor rules or copilot instructions\n\nStart by examining the root directory structure, then read key files like package.json, README.md, and any configuration files to understand the project setup.", "subagent_type": "Explore"} <|tool_call_end|> <|tool_calls_section_end|>"""

        tool_calls = parser.extract_tool_calls(reasoning_content)

        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call_0"
        assert tool_calls[0]["function"]["name"] == "functions.Task:0"

        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args["description"] == "Explore codebase structure"
        assert args["subagent_type"] == "Explore"
        assert "Explore the codebase structure" in args["prompt"]

        # Test reasoning extraction
        reasoning, main = parser.extract_reasoning_content(reasoning_content)
        assert reasoning is not None
        assert "Saya akan menganalisis kode ini" in reasoning
        assert main is None  # No content after tool calls

    def test_normalize_function_name_basic(self):
        """Test basic function name normalization"""
        assert KimiK2ToolParser.normalize_function_name("functions.Read:3") == "Read"
        assert KimiK2ToolParser.normalize_function_name("functions.Bash:1") == "Bash"
        assert KimiK2ToolParser.normalize_function_name("Bash") == "Bash"
        assert KimiK2ToolParser.normalize_function_name("") == ""

    def test_normalize_function_name_variations(self):
        """Test various function name format variations"""
        assert KimiK2ToolParser.normalize_function_name("functions.Read") == "Read"
        assert KimiK2ToolParser.normalize_function_name("Read:10") == "Read"
        assert KimiK2ToolParser.normalize_function_name("Grep:1") == "Grep"
        assert KimiK2ToolParser.normalize_function_name("functions.Grep:123") == "Grep"
        assert KimiK2ToolParser.normalize_function_name("myFunction:0") == "myFunction"

    def test_normalize_tool_calls_list(self):
        """Test normalize_tool_calls with multiple tool calls"""
        tool_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "functions.Read:3",
                    "arguments": '{"file_path": "/path/to/file"}'
                }
            },
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "functions.Bash:1",
                    "arguments": '{"command": "ls"}'
                }
            }
        ]

        normalized = KimiK2ToolParser.normalize_tool_calls(tool_calls)

        assert len(normalized) == 2
        assert normalized[0]["function"]["name"] == "Read"
        assert normalized[1]["function"]["name"] == "Bash"
        assert normalized[0]["id"] == "call_0"
        assert normalized[1]["id"] == "call_1"

    def test_normalize_tool_calls_no_changes_needed(self):
        """Test normalize_tool_calls when names are already normalized"""
        tool_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "Read",
                    "arguments": '{"file_path": "/path"}'
                }
            }
        ]

        normalized = KimiK2ToolParser.normalize_tool_calls(tool_calls)

        assert normalized[0]["function"]["name"] == "Read"

    def test_normalize_tool_calls_empty_list(self):
        """Test normalize_tool_calls with empty list"""
        assert KimiK2ToolParser.normalize_tool_calls([]) == []

    def test_normalize_function_name_colons_in_name(self):
        """Test function names that contain colon as part of name"""
        # The regex should only remove the numeric suffix pattern
        assert KimiK2ToolParser.normalize_function_name("functions.Read:3") == "Read"
        assert KimiK2ToolParser.normalize_function_name("my:tool:10") == "my:tool"

    def test_normalize_function_name_none_input(self):
        """Test normalize_function_name with None input"""
        assert KimiK2ToolParser.normalize_function_name(None) is None


class TestKimiK2StreamingHandler:
    """Test class for Kimi-K2 streaming handler"""

    def test_streaming_detects_model_from_chunk(self):
        """Test that the streaming handler detects Kimi-K2 model from chunk"""
        from litellm.llms.chutes.chat.kimi_k2_streaming import KimiK2StreamingHandler
        from unittest.mock import MagicMock

        handler = KimiK2StreamingHandler(
            streaming_response=[],
            sync_stream=True,
        )

        # First chunk with Kimi-K2 model
        chunk = {
            "model": "kimi-k2-thinking",
            "choices": [{"delta": {"content": "Hello"}}]
        }

        result = handler.chunk_parser(chunk)

        # Handler should now be in Kimi-K2 parsing mode
        assert handler._model == "kimi-k2-thinking"
        assert handler._should_parse_kimi_tool_calls is True

    def test_streaming_non_kimi_model(self):
        """Test that non-Kimi-K2 models don't trigger special parsing"""
        from litellm.llms.chutes.chat.kimi_k2_streaming import KimiK2StreamingHandler

        handler = KimiK2StreamingHandler(
            streaming_response=[],
            sync_stream=True,
        )

        chunk = {
            "model": "gpt-4",
            "choices": [{"delta": {"content": "Hello world"}}]
        }

        result = handler.chunk_parser(chunk)

        assert handler._should_parse_kimi_tool_calls is False

    def test_streaming_text_chunk_no_tool_calls(self):
        """Test streaming chunk with regular text (no tool calls)"""
        from litellm.llms.chutes.chat.kimi_k2_streaming import KimiK2StreamingHandler

        handler = KimiK2StreamingHandler(
            streaming_response=[],
            sync_stream=True,
        )
        handler._should_parse_kimi_tool_calls = True

        chunk = {
            "model": "kimi-k2-thinking",
            "choices": [{"delta": {"content": "This is regular text"}}]
        }

        result = handler.chunk_parser(chunk)

        assert "tool_use" not in result or result["tool_use"] is None

    def test_streaming_chunk_with_start_marker(self):
        """Test streaming chunk with tool call start marker"""
        from litellm.llms.chutes.chat.kimi_k2_streaming import KimiK2StreamingHandler

        handler = KimiK2StreamingHandler(
            streaming_response=[],
            sync_stream=True,
        )
        handler._should_parse_kimi_tool_calls = True

        chunk = {
            "model": "kimi-k2-thinking",
            "choices": [{
                "delta": {
                    "reasoning_content": "I'll use a tool.\n<|tool_calls_section_begin|>\n<|tool_call_begin|> get_weather"
                }
            }]
        }

        result = handler.chunk_parser(chunk)

        # Should buffer the content, not return tool_calls yet
        assert "tool_use" not in result or result["tool_use"] is None

    def test_streaming_complete_tool_calls_section(self):
        """Test streaming with complete tool calls section"""
        from litellm.llms.chutes.chat.kimi_k2_streaming import KimiK2StreamingHandler
        import json

        handler = KimiK2StreamingHandler(
            streaming_response=[],
            sync_stream=True,
        )
        handler._should_parse_kimi_tool_calls = True

        # Complete tool calls section
        chunk = {
            "model": "kimi-k2-thinking",
            "choices": [{
                "delta": {
                    "reasoning_content": "<|tool_calls_section_begin|>\n<|tool_call_begin|> get_weather\n<|tool_call_argument_begin|> {\"city\": \"Jakarta\"}\n<|tool_call_end|>\n<|tool_calls_section_end|>"
                }
            }]
        }

        result = handler.chunk_parser(chunk)

        # Should parse and return tool_calls
        assert result.get("tool_use") is not None
        assert result["tool_use"]["type"] == "function"
        assert result["tool_use"]["function"]["name"] == "get_weather"

    def test_streaming_multiple_chunks_accumulate(self):
        """Test that multiple chunks accumulate buffer correctly"""
        from litellm.llms.chutes.chat.kimi_k2_streaming import KimiK2StreamingHandler

        handler = KimiK2StreamingHandler(
            streaming_response=[],
            sync_stream=True,
        )
        handler._should_parse_kimi_tool_calls = True

        # First chunk with start marker
        chunk1 = {
            "model": "kimi-k2-thinking",
            "choices": [{
                "delta": {
                    "reasoning_content": "<|tool_calls_section_begin|>\n<|tool_call_begin|> test_func"
                }
            }]
        }

        result1 = handler.chunk_parser(chunk1)
        assert "<|tool_call_begin|> test_func" in handler._buffer

        # Second chunk with arguments
        chunk2 = {
            "model": "kimi-k2-thinking",
            "choices": [{
                "delta": {
                    "reasoning_content": "\n<|tool_call_argument_begin|> {\"x\": 1}\n<|tool_call_end|>\n<|tool_calls_section_end|>\nThen some text."
                }
            }]
        }

        result2 = handler.chunk_parser(chunk2)
        assert result2.get("tool_use") is not None
        assert "Then some text." in handler._buffer

    def test_streaming_existing_tool_calls_normalized(self):
        """Test streaming with existing OpenAI format tool_calls that need normalization"""
        from litellm.llms.chutes.chat.kimi_k2_streaming import KimiK2StreamingHandler
        import json

        handler = KimiK2StreamingHandler(
            streaming_response=[],
            sync_stream=True,
        )

        # Chunk with existing OpenAI format tool_calls requiring normalization
        chunk = {
            "model": "kimi-k2-thinking",
            "choices": [{
                "delta": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "functions.Read:3",
                            "arguments": '{"file_path": "/path/to/file"}'
                        }
                    }]
                }
            }]
        }

        result = handler.chunk_parser(chunk)

        # Should return normalized tool calls
        assert result.get("tool_use") is not None
        assert result["tool_use"]["function"]["name"] == "Read"
        assert result["tool_use"]["id"] == "call_0"

    def test_streaming_existing_tool_calls_no_normalization_needed(self):
        """Test streaming with already normalized tool_calls"""
        from litellm.llms.chutes.chat.kimi_k2_streaming import KimiK2StreamingHandler

        handler = KimiK2StreamingHandler(
            streaming_response=[],
            sync_stream=True,
        )

        chunk = {
            "model": "kimi-k2-thinking",
            "choices": [{
                "delta": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": '{"file_path": "/path"}'
                        }
                    }]
                }
            }]
        }

        result = handler.chunk_parser(chunk)

        assert result.get("tool_use") is not None
        assert result["tool_use"]["function"]["name"] == "Read"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
