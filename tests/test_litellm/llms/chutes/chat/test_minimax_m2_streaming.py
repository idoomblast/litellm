"""
Tests for MiniMax M2 streaming transformation.
"""

import pytest

from litellm.llms.chutes.chat.minimax_m2_streaming import MiniMaxM2StreamingHandler


class TestMiniMaxM2StreamingHandler:
    """Test MiniMax M2 streaming handler functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MiniMaxM2StreamingHandler(
            streaming_response=[],
            sync_stream=True,
            json_mode=False,
        )

    def test_is_minimax_m2_model(self):
        """Test MiniMax M2 model detection."""
        test_cases = [
            ("minimax-m2", True),
            ("minimax-m2.1", True),
            ("minimax_m2", True),
            ("minimax_m2.1-lightning", True),
            ("MiniMax M2", True),
            ("MINIMAX-M2", True),
            ("minimax-m1", False),
            ("gpt-4", False),
            ("kimi-k2", False),
        ]

        for model, expected in test_cases:
            result = self.handler._is_minimax_m2_model(model)
            assert result == expected, f"Failed for model: {model}"

    def test_extract_text_from_chunk(self):
        """Test text extraction from chunks."""
        # Test with content field
        chunk = {
            "choices": [{"delta": {"content": "Hello world"}}],
        }
        result = self.handler._extract_text_from_chunk(chunk)
        assert result == "Hello world"

        # Test with reasoning_content field
        chunk = {
            "choices": [{"delta": {"reasoning_content": "Thinking..."}}],
        }
        result = self.handler._extract_text_from_chunk(chunk)
        assert result == "Thinking..."

        # Test with empty chunk
        chunk = {"choices": []}
        result = self.handler._extract_text_from_chunk(chunk)
        assert result == ""

        # Test with missing fields
        chunk = {}
        result = self.handler._extract_text_from_chunk(chunk)
        assert result == ""

    def test_extract_tool_calls_from_delta(self):
        """Test tool_calls extraction from delta."""
        # Test with tool_calls in delta
        chunk = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "test"}'
                        }
                    }]
                }
            }],
        }
        result = self.handler._extract_tool_calls_from_delta(chunk)
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "search"

        # Test without tool_calls
        chunk = {
            "choices": [{"delta": {"content": "Hello"}}],
        }
        result = self.handler._extract_tool_calls_from_delta(chunk)
        assert result is None

    def test_convert_param_value(self):
        """Test parameter value conversion."""
        # Test null
        assert self.handler._convert_param_value("null") is None

        # Test boolean
        assert self.handler._convert_param_value("true") is True
        assert self.handler._convert_param_value("false") is False
        assert self.handler._convert_param_value("True") is True
        assert self.handler._convert_param_value("False") is False

        # Test integer
        assert self.handler._convert_param_value("42") == 42
        assert self.handler._convert_param_value("0") == 0
        assert self.handler._convert_param_value("-123") == -123

        # Test float
        assert self.handler._convert_param_value("3.14") == 3.14
        assert self.handler._convert_param_value("2.0") == 2

        # Test JSON object
        result = self.handler._convert_param_value('{"key": "value"}')
        assert isinstance(result, dict)
        assert result["key"] == "value"

        # Test JSON array
        result = self.handler._convert_param_value('[1, 2, 3]')
        assert isinstance(result, list)
        assert result == [1, 2, 3]

        # Test string (default)
        assert self.handler._convert_param_value("hello") == "hello"
        assert self.handler._convert_param_value("123abc") == "123abc"

    def test_parse_xml_tool_call(self):
        """Test XML tool call parsing."""
        # Test simple tool call
        xml = '<invoke name="search"><parameter name="query">python</parameter></invoke>'
        result = self.handler._parse_xml_tool_call(xml)
        assert result is not None
        assert result["id"] == "call_0"
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"
        assert '"query": "python"' in result["function"]["arguments"]

        # Test multiple parameters
        xml = """
        <invoke name="calculate">
            <parameter name="x">10</parameter>
            <parameter name="y">20</parameter>
            <parameter name="operation">add</parameter>
        </invoke>
        """
        result = self.handler._parse_xml_tool_call(xml)
        assert result is not None
        assert result["function"]["name"] == "calculate"
        import json
        args = json.loads(result["function"]["arguments"])
        assert args["x"] == 10
        assert args["y"] == 20
        assert args["operation"] == "add"

        # Test with parameter values that need type conversion
        xml = '<invoke name="test"><parameter name="count">42</parameter><parameter name="flag">true</parameter></invoke>'
        result = self.handler._parse_xml_tool_call(xml)
        assert result is not None
        args = json.loads(result["function"]["arguments"])
        assert args["count"] == 42
        assert args["flag"] is True

        # Test invalid XML
        result = self.handler._parse_xml_tool_call("")
        assert result is None

        result = self.handler._parse_xml_tool_call("no xml here")
        assert result is None

    def test_chunk_parser_non_minimax_model(self):
        """Test chunk parser with non-MiniMax model."""
        chunk = {
            "model": "gpt-4",
            "choices": [{"delta": {"content": "Hello"}}],
        }
        result = self.handler.chunk_parser(chunk)
        assert result["text"] == "Hello"
        assert result["tool_use"] is None

    def test_chunk_parser_with_openai_format_tool_calls(self):
        """Test chunk parser when provider returns OpenAI format tool_calls."""
        chunk = {
            "model": "minimax-m2",
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "test"}'
                        }
                    }]
                }
            }],
        }
        result = self.handler.chunk_parser(chunk)
        assert result["tool_use"] is not None
        assert result["tool_use"]["id"] == "call_123"
        assert result["tool_use"]["type"] == "function"
        assert result["tool_use"]["function"]["name"] == "search"

    def test_chunk_parser_with_xml_tool_call(self):
        """Test chunk parser with XML tool call."""
        chunk = {
            "model": "minimax-m2",
            "choices": [{
                "delta": {
                    "content": "<minimax:tool_call><invoke name=\"search\"><parameter name=\"query\">test</parameter></invoke></minimax:tool_call>"
                }
            }],
        }
        result = self.handler.chunk_parser(chunk)
        assert result["tool_use"] is not None
        assert result["tool_use"]["type"] == "function"
        assert result["tool_use"]["function"]["name"] == "search"

    def test_chunk_parser_partial_xml(self):
        """Test chunk parser with partial XML (streaming scenario)."""
        # First chunk with opening tag only
        chunk1 = {
            "model": "minimax-m2",
            "choices": [{
                "delta": {
                    "content": "Some text <minimax:tool_call>"
                }
            }],
        }
        result1 = self.handler.chunk_parser(chunk1)
        # Should buffer and return text before tool call
        assert "Some text " in result1["text"] or result1["text"] == ""

        # Second chunk with closing tag
        chunk2 = {
            "model": "minimax-m2",
            "choices": [{
                "delta": {
                    "content": "<invoke name=\"search\"><parameter name=\"query\">test</parameter></invoke></minimax:tool_call>"
                }
            }],
        }
        result2 = self.handler.chunk_parser(chunk2)
        assert result2["tool_use"] is not None
        assert result2["tool_use"]["function"]["name"] == "search"

    def test_extract_complete_tool_call(self):
        """Test complete tool call extraction from buffer."""
        # Set buffer with complete tool call
        self.handler._buffer = '<minimax:tool_call><invoke name="test"><parameter name="arg">value</parameter></invoke></minimax:tool_call>'
        result = self.handler._extract_complete_tool_call()
        assert result is not None
        assert result["function"]["name"] == "test"

    def test_extract_complete_tool_call_incomplete(self):
        """Test complete tool call extraction with incomplete buffer."""
        # Set buffer without closing tag
        self.handler._buffer = '<minimax:tool_call><invoke name="test">'
        result = self.handler._extract_complete_tool_call()
        assert result is None

    def test_create_empty_chunk(self):
        """Test empty chunk creation."""
        chunk = {"index": 5}
        result = self.handler._create_empty_chunk(chunk)
        assert result["text"] == ""
        assert result["is_finished"] is False
        assert result["tool_use"] is None
        assert result["index"] == 5

    def test_process_text_chunk_no_tool_call(self):
        """Test text chunk processing without tool calls."""
        chunk = {"index": 0}
        result = self.handler._process_text_chunk("Hello world", chunk)
        assert result["text"] == "Hello world"
        assert result["tool_use"] is None


class TestMiniMaxM2StreamingIntegration:
    """Integration tests for MiniMax M2 streaming scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MiniMaxM2StreamingHandler(
            streaming_response=[],
            sync_stream=True,
            json_mode=False,
        )

    def test_full_streaming_scenario_single_tool_call(self):
        """Test full streaming scenario with a single tool call."""
        chunks = [
            {"model": "minimax-m2", "choices": [{"delta": {"content": "Let me search"}}]},
            {"model": "minimax-m2", "choices": [{"delta": {"content": " that for you."}}]},
            {"model": "minimax-m2", "choices": [{"delta": {"content": "<minimax:tool_call><invoke name=\"search\"><parameter name=\"query\">python tutorial</parameter></invoke></minimax:tool_call>"}}]},
        ]

        results = []
        for chunk in chunks:
            result = self.handler.chunk_parser(chunk)
            results.append(result)

        # First two chunks should have text
        assert results[0]["text"] == "Let me search"
        assert results[1]["text"] == " that for you."

        # Third chunk should have tool call
        assert results[2]["tool_use"] is not None
        assert results[2]["tool_use"]["function"]["name"] == "search"

    def test_full_streaming_scenario_split_tool_call(self):
        """Test full streaming scenario with tool call split across chunks."""
        chunks = [
            {"model": "minimax-m2", "choices": [{"delta": {"content": "<minimax:tool_call>"}}]},
            {"model": "minimax-m2", "choices": [{"delta": {"content": "<invoke name=\"search\">"}}]},
            {"model": "minimax-m2", "choices": [{"delta": {"content": "<parameter name=\"query\">python</parameter>"}}]},
            {"model": "minimax-m2", "choices": [{"delta": {"content": "</invoke></minimax:tool_call>"}}]},
        ]

        results = []
        for chunk in chunks:
            result = self.handler.chunk_parser(chunk)
            results.append(result)

        # Only last chunk should have tool call
        assert all(results[i]["tool_use"] is None for i in range(len(results) - 1))
        assert results[-1]["tool_use"] is not None
        assert results[-1]["tool_use"]["function"]["name"] == "search"

    def test_multiple_function_calls_consecutive(self):
        """Test streaming with multiple function calls consecutively."""
        chunks = [
            {"model": "minimax-m2", "choices": [{"delta": {"content": "<minimax:tool_call><invoke name=\"search\"><parameter name=\"query\">python</parameter></invoke></minimax:tool_call>"}}]},
            {"model": "minimax-m2", "choices": [{"delta": {"content": "<minimax:tool_call><invoke name=\"calculate\"><parameter name=\"x\">10</parameter><parameter name=\"y\">20</parameter></invoke></minimax:tool_call>"}}]},
        ]

        results = []
        for chunk in chunks:
            result = self.handler.chunk_parser(chunk)
            results.append(result)

        # Each chunk should have a tool call
        for i, result in enumerate(results):
            assert result["tool_use"] is not None, f"Chunk {i} should have tool call"

        assert results[0]["tool_use"]["function"]["name"] == "search"
        assert results[1]["tool_use"]["function"]["name"] == "calculate"

    def test_mixed_text_and_tool_calls(self):
        """Test streaming with mixed text and tool calls."""
        chunks = [
            {"model": "minimax-m2", "choices": [{"delta": {"content": "I'll help you find that information."}}]},
            {"model": "minimax-m2", "choices": [{"delta": {"content": " <minimax:tool_call><invoke name=\"search\"><parameter name=\"query\">answer</parameter></invoke></minimax:tool_call>"}}]},
        ]

        results = []
        for chunk in chunks:
            result = self.handler.chunk_parser(chunk)
            results.append(result)

        assert "I'll help you find that information." in results[0]["text"]
        assert results[1]["tool_use"] is not None
        assert results[1]["tool_use"]["function"]["name"] == "search"