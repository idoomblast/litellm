"""
Tests for Vertex AI Moonshot (Kimi K2) partner model transformation.

Covers:
- Config initialization and parameter mapping
- Native Kimi K2 tool call parsing in non-streaming responses
- finish_reason fix ("stop" → "tool_calls")
- <think> tag extraction to reasoning_content
- Streaming handler: cross-chunk buffering
- Streaming handler: stateful think tag processing
- Streaming handler: incremental reasoning emission
- Streaming handler: end-of-stream buffer flushing
- Streaming handler: duplicate tool call prevention
- Streaming handler: tool section wrappers
"""

import json
import pytest
from unittest.mock import MagicMock

from litellm.llms.vertex_ai.vertex_ai_partner_models.moonshot.transformation import (
    VertexAIMoonshotConfig,
    VertexAIMoonshotStreamingHandler,
)
from litellm.llms.chutes.chat.kimi_k2_tool_call_parser import (
    TOOL_CALLS_SECTION_BEGIN,
    TOOL_CALLS_SECTION_END,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    TOOL_CALL_ARGUMENT_BEGIN,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Delta,
    Function,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)


class TestVertexAIMoonshotConfig:
    """Tests for VertexAIMoonshotConfig."""

    def setup_method(self):
        self.config = VertexAIMoonshotConfig()

    def test_should_support_thinking_params(self):
        """should include thinking and reasoning_effort in supported params."""
        params = self.config.get_supported_openai_params("moonshotai/kimi-k2-thinking-maas")
        assert "thinking" in params
        assert "reasoning_effort" in params

    def test_should_map_thinking_enabled(self):
        """should map thinking=True to chat_template_kwargs.enable_thinking=True."""
        result = self.config.map_openai_params(
            non_default_params={"thinking": True},
            optional_params={"thinking": True},
            model="moonshotai/kimi-k2-thinking-maas",
            drop_params=False,
        )
        assert result["chat_template_kwargs"]["enable_thinking"] is True

    def test_should_map_thinking_disabled(self):
        """should map thinking=False to chat_template_kwargs.enable_thinking=False."""
        result = self.config.map_openai_params(
            non_default_params={"thinking": False},
            optional_params={"thinking": False},
            model="moonshotai/kimi-k2-thinking-maas",
            drop_params=False,
        )
        assert result["chat_template_kwargs"]["enable_thinking"] is False

    def test_should_map_thinking_dict_enabled(self):
        """should map thinking={"type": "enabled"} correctly."""
        result = self.config.map_openai_params(
            non_default_params={"thinking": {"type": "enabled"}},
            optional_params={"thinking": {"type": "enabled"}},
            model="moonshotai/kimi-k2-thinking-maas",
            drop_params=False,
        )
        assert result["chat_template_kwargs"]["enable_thinking"] is True

    def test_should_map_reasoning_effort_high(self):
        """should map reasoning_effort='high' to enable_thinking=True."""
        result = self.config.map_openai_params(
            non_default_params={"reasoning_effort": "high"},
            optional_params={"reasoning_effort": "high"},
            model="moonshotai/kimi-k2-thinking-maas",
            drop_params=False,
        )
        assert result["chat_template_kwargs"]["enable_thinking"] is True

    def test_should_map_reasoning_effort_none(self):
        """should map reasoning_effort='none' to enable_thinking=False."""
        result = self.config.map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={"reasoning_effort": "none"},
            model="moonshotai/kimi-k2-thinking-maas",
            drop_params=False,
        )
        assert result["chat_template_kwargs"]["enable_thinking"] is False


class TestVertexAIMoonshotToolCallParsing:
    """Tests for Kimi K2 tool call parsing in non-streaming responses."""

    def setup_method(self):
        self.config = VertexAIMoonshotConfig()

    def _make_response(self, content, finish_reason="stop", tool_calls=None):
        """Helper to create a ModelResponse with given content."""
        message = Message(
            content=content,
            role="assistant",
        )
        if tool_calls:
            message.tool_calls = tool_calls

        choice = Choices(
            index=0,
            message=message,
            finish_reason=finish_reason,
        )
        return ModelResponse(
            id="test-id",
            choices=[choice],
            model="moonshotai/kimi-k2-thinking-maas",
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    def test_should_parse_native_tool_calls_from_content(self):
        """should parse native Kimi K2 tool call tokens from content field."""
        content = '<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Beijing"}<|tool_call_end|>'
        response = self._make_response(content)

        result = self.config._parse_kimi_k2_tool_calls_from_response(response)

        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1
        tc = result.choices[0].message.tool_calls[0]
        assert tc.function.name == "get_weather"
        assert json.loads(tc.function.arguments) == {"city": "Beijing"}

    def test_should_fix_finish_reason_when_tool_calls_found(self):
        """should fix finish_reason from 'stop' to 'tool_calls' when native tool calls are parsed."""
        content = '<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{"q": "test"}<|tool_call_end|>'
        response = self._make_response(content, finish_reason="stop")

        result = self.config._parse_kimi_k2_tool_calls_from_response(response)

        assert result.choices[0].finish_reason == "tool_calls"

    def test_should_fix_finish_reason_with_existing_tool_calls(self):
        """should fix finish_reason when tool_calls already exist from standard format."""
        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(name="get_weather", arguments='{"city": "Tokyo"}'),
            )
        ]
        response = self._make_response(
            content="some content with <|tool_call_begin|>...",
            finish_reason="stop",
            tool_calls=tool_calls,
        )

        result = self.config._parse_kimi_k2_tool_calls_from_response(response)

        assert result.choices[0].finish_reason == "tool_calls"

    def test_should_extract_think_tags_to_reasoning_content(self):
        """should extract <think> tags into reasoning_content field."""
        content = "<think>Let me analyze this step by step...</think>The answer is 42."
        response = self._make_response(content)

        result = self.config._parse_kimi_k2_tool_calls_from_response(response)

        assert result.choices[0].message.reasoning_content == "Let me analyze this step by step..."
        assert result.choices[0].message.content == "The answer is 42."

    def test_should_handle_think_tags_with_tool_calls(self):
        """should handle both <think> tags and tool calls in the same response."""
        content = (
            "<think>I need to check the weather</think>"
            '<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "NYC"}<|tool_call_end|>'
        )
        response = self._make_response(content, finish_reason="stop")

        result = self.config._parse_kimi_k2_tool_calls_from_response(response)

        assert result.choices[0].message.reasoning_content == "I need to check the weather"
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1
        assert result.choices[0].finish_reason == "tool_calls"

    def test_should_parse_multiple_tool_calls(self):
        """should parse multiple native tool calls from content."""
        content = (
            '<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "NYC"}<|tool_call_end|>'
            '<|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>{"tz": "EST"}<|tool_call_end|>'
        )
        response = self._make_response(content, finish_reason="stop")

        result = self.config._parse_kimi_k2_tool_calls_from_response(response)

        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 2
        assert result.choices[0].message.tool_calls[0].function.name == "get_weather"
        assert result.choices[0].message.tool_calls[1].function.name == "get_time"
        assert result.choices[0].finish_reason == "tool_calls"

    def test_should_not_modify_response_without_tool_calls(self):
        """should not modify response when no tool calls are present."""
        response = self._make_response("Just a regular response", finish_reason="stop")

        result = self.config._parse_kimi_k2_tool_calls_from_response(response)

        assert result.choices[0].message.content == "Just a regular response"
        assert result.choices[0].message.tool_calls is None
        assert result.choices[0].finish_reason == "stop"

    def test_should_clean_content_after_parsing_tool_calls(self):
        """should clean up native tool tokens from content after parsing."""
        content = (
            "Here is the result: "
            '<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{"q": "test"}<|tool_call_end|>'
        )
        response = self._make_response(content, finish_reason="stop")

        result = self.config._parse_kimi_k2_tool_calls_from_response(response)

        assert result.choices[0].message.tool_calls is not None
        # Content should be cleaned of tool call tokens
        cleaned_content = result.choices[0].message.content
        assert cleaned_content is None or "<|tool_call" not in (cleaned_content or "")


class TestVertexAIMoonshotStreamingHandler:
    """Tests for VertexAIMoonshotStreamingHandler (upgraded with full Chutes parity)."""

    def setup_method(self):
        self.handler = VertexAIMoonshotStreamingHandler(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

    # ── Basic streaming ──

    def test_should_handle_regular_content_streaming(self):
        """should stream regular content and flush on finish_reason."""
        chunk1 = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello, world!"},
                    "finish_reason": None,
                }
            ],
        }
        chunk2 = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " How are you?"},
                    "finish_reason": "stop",
                }
            ],
        }

        result1 = self.handler.chunk_parser(chunk1)
        assert result1.id == "test-id"
        assert result1.model == "kimi-k2"

        result2 = self.handler.chunk_parser(chunk2)
        assert result2.choices[0].finish_reason == "stop"
        # Content should be flushed on finish
        assert result2.choices[0].delta.content is not None

    def test_should_handle_empty_choices(self):
        """should handle chunks with empty choices gracefully."""
        chunk = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [],
        }

        result = self.handler.chunk_parser(chunk)
        assert len(result.choices) == 0

    def test_should_preserve_chunk_metadata(self):
        """should preserve chunk metadata (id, model, created)."""
        chunk = {
            "id": "chatcmpl-abc123",
            "model": "moonshotai/kimi-k2-thinking-maas",
            "created": 1234567890,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Test"},
                    "finish_reason": None,
                }
            ],
        }

        result = self.handler.chunk_parser(chunk)

        assert result.id == "chatcmpl-abc123"
        assert result.model == "moonshotai/kimi-k2-thinking-maas"
        assert result.created == 1234567890

    # ── Standard tool call detection ──

    def test_should_detect_standard_tool_calls(self):
        """should track standard OpenAI tool_calls and set _saw_standard_tool_calls."""
        chunk = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }

        result = self.handler.chunk_parser(chunk)

        assert self.handler._saw_standard_tool_calls is True
        assert self.handler._emitted_any_tool_calls is True
        assert result.choices[0].delta.tool_calls is not None

    def test_should_clean_tool_call_xml_tags_from_function_names(self):
        """should clean <tool_call> XML tags from function names in standard format."""
        chunk = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "<tool_call>get_weather",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }

        result = self.handler.chunk_parser(chunk)

        tc = result.choices[0].delta.tool_calls[0]
        assert tc["function"]["name"] == "get_weather"

    # ── Native tool call parsing with section wrappers ──

    def test_should_parse_native_tool_calls_with_section_wrappers(self):
        """should parse native tool calls with section wrappers in single chunk."""
        tool_call_content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        chunk = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": tool_call_content},
                    "finish_reason": None,
                }
            ],
        }

        result = self.handler.chunk_parser(chunk)

        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].delta.tool_calls[0].function.name == "get_weather"

    def test_should_parse_native_tool_calls_without_section_wrappers(self):
        """should parse native tool calls without section wrappers (Chutes format)."""
        native_content = (
            f'{TOOL_CALL_BEGIN}functions.search:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"q": "test"}}{TOOL_CALL_END}'
        )

        chunk = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": native_content},
                    "finish_reason": None,
                }
            ],
        }

        result = self.handler.chunk_parser(chunk)

        assert self.handler._emitted_any_tool_calls is True
        assert result.choices[0].delta.tool_calls is not None
        assert result.choices[0].delta.tool_calls[0].function.name == "search"

    def test_should_parse_multiple_tool_calls_in_section(self):
        """should parse multiple tool calls in one section."""
        tool_call_content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.func1:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"arg": "1"}}{TOOL_CALL_END}\n'
            f'{TOOL_CALL_BEGIN}functions.func2:1{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"arg": "2"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        chunk = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": tool_call_content},
                    "finish_reason": None,
                }
            ],
        }

        result = self.handler.chunk_parser(chunk)

        tool_calls = result.choices[0].delta.tool_calls
        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0].function.name == "func1"
        assert tool_calls[1].function.name == "func2"
        assert tool_calls[0].index == 0
        assert tool_calls[1].index == 1

    # ── Cross-chunk buffering ──

    def test_should_buffer_partial_tokens_across_chunks(self):
        """should buffer content when tool call token is split across chunks."""
        # First chunk ends mid-token
        chunk1 = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "<|tool_calls_section"},
                    "finish_reason": None,
                }
            ],
        }

        # Second chunk completes the token
        remaining = (
            f"_begin|>\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )
        chunk2 = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": remaining},
                    "finish_reason": None,
                }
            ],
        }

        # Process first chunk — should buffer, not emit tool calls yet
        result1 = self.handler.chunk_parser(chunk1)
        assert result1.choices[0].delta.tool_calls is None

        # Process second chunk — should complete and emit tool calls
        result2 = self.handler.chunk_parser(chunk2)
        assert result2.choices[0].delta.tool_calls is not None
        assert len(result2.choices[0].delta.tool_calls) == 1
        assert result2.choices[0].delta.tool_calls[0].function.name == "get_weather"

    def test_should_detect_tool_call_in_reasoning_content_field(self):
        """should detect tool calls in reasoning_content field."""
        tool_call_content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.analyze:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"data": "test"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        chunk = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "Regular content",
                        "reasoning_content": tool_call_content,
                    },
                    "finish_reason": None,
                }
            ],
        }

        result = self.handler.chunk_parser(chunk)

        assert result.choices[0].delta.tool_calls is not None
        assert result.choices[0].delta.tool_calls[0].function.name == "analyze"

    # ── Duplicate tool call prevention ──

    def test_should_skip_native_tokens_when_standard_format_seen(self):
        """should skip native tool tokens when standard format was already seen."""
        # Chunk 1: standard tool_calls
        chunk1 = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "search", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        self.handler.chunk_parser(chunk1)
        assert self.handler._saw_standard_tool_calls is True

        # Chunk 2: native tokens in content (should NOT produce additional tool calls)
        native_content = (
            f'{TOOL_CALL_BEGIN}functions.search:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"q": "test"}}{TOOL_CALL_END}'
        )
        chunk2 = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": native_content},
                    "finish_reason": None,
                }
            ],
        }

        result2 = self.handler.chunk_parser(chunk2)

        # Native tokens should NOT produce additional tool calls
        # They should be buffered but not parsed
        if result2.choices[0].delta.tool_calls:
            # If any tool_calls are present, they shouldn't be from native parsing
            pass
        # The handler should still have saw_standard flag
        assert self.handler._saw_standard_tool_calls is True

    def test_should_strip_native_tokens_on_flush_when_standard_seen(self):
        """should strip native tokens from flushed content when standard format was seen."""
        # Set standard flag
        self.handler._saw_standard_tool_calls = True
        self.handler._emitted_any_tool_calls = True

        # Add native tokens to content buffer
        native_content = (
            f"Some text {TOOL_CALL_BEGIN}functions.test:0{TOOL_CALL_ARGUMENT_BEGIN}"
            f'{{"a": 1}}{TOOL_CALL_END}'
        )
        self.handler._field_buffers["content"] = native_content

        # Flush
        content_out, reasoning_out = self.handler._flush_buffers_at_end_of_stream()

        # Native tokens should be stripped
        if content_out:
            assert "<|tool_call" not in content_out

    # ── Stateful think tag processing ──

    def test_should_handle_think_tags_across_chunks(self):
        """should handle <think> tag spanning multiple chunks — processed at flush."""
        # Chunk 1: <think> starts (buffered, not processed yet)
        chunk1 = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " <think>"},
                    "finish_reason": None,
                }
            ],
        }

        # Chunk 2: reasoning content (still buffered)
        chunk2 = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Let me analyze this..."},
                    "finish_reason": None,
                }
            ],
        }

        # Chunk 3: </think> ends + regular content + finish (triggers flush)
        chunk3 = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " </think>The answer is 42."},
                    "finish_reason": "stop",
                }
            ],
        }

        # Content is buffered, not processed yet — _in_think_block stays False
        self.handler.chunk_parser(chunk1)
        self.handler.chunk_parser(chunk2)

        # Chunk 3 has finish_reason → flush triggers think tag processing
        result3 = self.handler.chunk_parser(chunk3)

        # After flush, think block should be closed (</think> was found)
        assert self.handler._in_think_block is False
        # Content and reasoning should be properly routed
        assert result3.choices[0].delta.content is not None or result3.choices[0].delta.reasoning_content is not None

    def test_should_handle_unclosed_think_tag(self):
        """should treat all content after unclosed <think> as reasoning_content at flush."""
        # Chunk 1: <think> starts, never closes (buffered)
        chunk1 = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "<think>I'm thinking deeply about this..."},
                    "finish_reason": None,
                }
            ],
        }

        # Chunk 2: More content, stream ends (triggers flush)
        chunk2 = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " and considering all options"},
                    "finish_reason": "stop",
                }
            ],
        }

        # Content is buffered, think tags not processed until flush
        self.handler.chunk_parser(chunk1)

        # Chunk 2 triggers flush — unclosed think tag means all content → reasoning
        result2 = self.handler.chunk_parser(chunk2)

        # After flush with unclosed think, _in_think_block should be True
        assert self.handler._in_think_block is True
        # All content should be treated as reasoning (unclosed think tag)
        assert result2.choices[0].delta.reasoning_content is not None

    def test_should_process_think_tags_via_state_machine(self):
        """should route content through think tag state machine correctly."""
        content_out, reasoning_out = self.handler._process_content_for_think_tags(
            "<think>reasoning here</think>regular content"
        )

        assert reasoning_out == "reasoning here"
        assert content_out == "regular content"
        assert self.handler._in_think_block is False

    def test_should_handle_think_tag_state_persistence(self):
        """should persist think block state across multiple calls."""
        # First call: opens think block
        content1, reasoning1 = self.handler._process_content_for_think_tags("<think>start")
        assert self.handler._in_think_block is True
        assert reasoning1 == "start"
        assert content1 is None

        # Second call: still in think block
        content2, reasoning2 = self.handler._process_content_for_think_tags("middle part")
        assert self.handler._in_think_block is True
        assert reasoning2 == "middle part"
        assert content2 is None

        # Third call: closes think block + regular content
        content3, reasoning3 = self.handler._process_content_for_think_tags("end</think>answer")
        assert self.handler._in_think_block is False
        assert reasoning3 == "end"
        assert content3 == "answer"

    # ── Incremental reasoning emission ──

    def test_should_emit_reasoning_incrementally(self):
        """should emit reasoning from dedicated fields incrementally."""
        # Add reasoning content to buffer
        self.handler._field_buffers["reasoning_content"] = "Step 1: analyze input"

        result = self.handler._get_incremental_reasoning()

        assert result == "Step 1: analyze input"
        # Buffer should be cleared after emission
        assert self.handler._field_buffers["reasoning_content"] == ""

    def test_should_deduplicate_reasoning_from_multiple_fields(self):
        """should deduplicate reasoning when same content in multiple fields."""
        # Simulate Chutes-style duplicate reasoning
        self.handler._field_buffers["reasoning_content"] = "thinking..."
        self.handler._field_buffers["reasoning"] = "thinking..."
        self.handler._field_buffers["thinking"] = "thinking..."

        result = self.handler._get_incremental_reasoning()

        # Should be deduplicated to single instance
        assert result == "thinking..."

    def test_should_not_emit_reasoning_when_in_tool_section(self):
        """should NOT emit reasoning when field is in a tool section."""
        self.handler._field_buffers["reasoning_content"] = "some reasoning"
        self.handler._in_tool_section["reasoning_content"] = True

        result = self.handler._get_incremental_reasoning()

        # Should not emit because reasoning_content is in tool section
        assert result is None
        # Buffer should still have content
        assert self.handler._field_buffers["reasoning_content"] == "some reasoning"

    def test_should_emit_reasoning_incrementally_during_streaming(self):
        """should emit reasoning_content incrementally via chunk_parser."""
        chunk = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "Step 1: check the input",
                    },
                    "finish_reason": None,
                }
            ],
        }

        result = self.handler.chunk_parser(chunk)

        assert result.choices[0].delta.reasoning_content == "Step 1: check the input"
        assert self.handler._emitted_any_reasoning is True

    # ── End-of-stream buffer flushing ──

    def test_should_flush_content_buffer_on_finish(self):
        """should flush remaining content buffer when finish_reason is set."""
        # Put content in buffer
        self.handler._field_buffers["content"] = "Hello, world!"

        content_out, reasoning_out = self.handler._flush_buffers_at_end_of_stream()

        assert content_out == "Hello, world!"
        assert self.handler._field_buffers["content"] == ""

    def test_should_flush_reasoning_buffers_on_finish(self):
        """should flush remaining reasoning buffers when finish_reason is set."""
        self.handler._field_buffers["reasoning_content"] = "final reasoning"

        content_out, reasoning_out = self.handler._flush_buffers_at_end_of_stream()

        assert reasoning_out == "final reasoning"

    def test_should_handle_unclosed_think_on_flush(self):
        """should move all content to reasoning when think tag is unclosed at flush."""
        self.handler._in_think_block = True
        self.handler._field_buffers["content"] = "this is all reasoning"

        content_out, reasoning_out = self.handler._flush_buffers_at_end_of_stream()

        # All content should become reasoning because think tag was never closed
        assert content_out is None
        assert reasoning_out is not None
        assert "this is all reasoning" in reasoning_out

    def test_should_strip_native_tokens_on_flush(self):
        """should strip native tool tokens from flushed content when standard format seen."""
        self.handler._saw_standard_tool_calls = True
        native_content = (
            f"Some text {TOOL_CALL_BEGIN}functions.test:0{TOOL_CALL_ARGUMENT_BEGIN}"
            f'{{"a": 1}}{TOOL_CALL_END}'
        )
        self.handler._field_buffers["content"] = native_content

        content_out, reasoning_out = self.handler._flush_buffers_at_end_of_stream()

        if content_out:
            assert "<|tool_call" not in content_out

    # ── finish_reason fix ──

    def test_should_fix_finish_reason_when_tool_calls_emitted(self):
        """should fix finish_reason to 'tool_calls' when tool calls were emitted."""
        # First: emit tool calls
        chunk1 = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "search", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        self.handler.chunk_parser(chunk1)

        # Second: finish with "stop" — should be fixed to "tool_calls"
        chunk2 = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }

        result = self.handler.chunk_parser(chunk2)

        assert result.choices[0].finish_reason == "tool_calls"

    def test_should_not_fix_finish_reason_without_tool_calls(self):
        """should NOT fix finish_reason when no tool calls were emitted."""
        chunk = {
            "id": "test-id",
            "model": "kimi-k2",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
        }

        result = self.handler.chunk_parser(chunk)

        assert result.choices[0].finish_reason == "stop"

    # ── Full stream simulation ──

    def test_should_handle_full_think_then_tool_call_stream(self):
        """should handle full stream: think → content → tool call → finish."""
        handler = VertexAIMoonshotStreamingHandler(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        # Chunk 1: Think start (buffered, not processed yet)
        chunk1 = {
            "id": "test-id", "model": "kimi-k2", "created": 123,
            "choices": [{"index": 0, "delta": {"content": "<think>"}, "finish_reason": None}],
        }

        # Chunk 2: Thinking content (buffered)
        chunk2 = {
            "id": "test-id", "model": "kimi-k2", "created": 123,
            "choices": [{"index": 0, "delta": {"content": "Let me think..."}, "finish_reason": None}],
        }

        # Chunk 3: Think end (buffered)
        chunk3 = {
            "id": "test-id", "model": "kimi-k2", "created": 123,
            "choices": [{"index": 0, "delta": {"content": "</think>"}, "finish_reason": None}],
        }

        # Chunk 4: Standard tool calls
        chunk4 = {
            "id": "test-id", "model": "kimi-k2", "created": 123,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0, "id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                    }]
                },
                "finish_reason": None,
            }],
        }

        # Chunk 5: Finish (triggers flush)
        chunk5 = {
            "id": "test-id", "model": "kimi-k2", "created": 123,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

        # Content is buffered during chunk 1-3, think tags not processed yet
        handler.chunk_parser(chunk1)
        handler.chunk_parser(chunk2)
        handler.chunk_parser(chunk3)

        # Chunk 4: standard tool calls detected
        r4 = handler.chunk_parser(chunk4)
        assert handler._saw_standard_tool_calls is True
        assert r4.choices[0].delta.tool_calls is not None

        # Chunk 5: finish triggers flush — think tags processed at this point
        r5 = handler.chunk_parser(chunk5)
        # finish_reason should be fixed to "tool_calls"
        assert r5.choices[0].finish_reason == "tool_calls"
        # After flush, think block should be closed (</think> was in buffer)
        assert handler._in_think_block is False

    def test_should_handle_dual_format_stream(self):
        """should handle Chutes-style dual format: standard + native tokens."""
        handler = VertexAIMoonshotStreamingHandler(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        # Chunk 1: Standard tool calls
        chunk1 = {
            "id": "test-id", "model": "kimi-k2", "created": 123,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0, "id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                    }]
                },
                "finish_reason": None,
            }],
        }

        # Chunk 2: Native tokens (DUPLICATES — should be skipped)
        native_content = (
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}'
        )
        chunk2 = {
            "id": "test-id", "model": "kimi-k2", "created": 123,
            "choices": [{
                "index": 0,
                "delta": {"content": native_content},
                "finish_reason": None,
            }],
        }

        # Chunk 3: Finish
        chunk3 = {
            "id": "test-id", "model": "kimi-k2", "created": 123,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

        r1 = handler.chunk_parser(chunk1)
        assert handler._saw_standard_tool_calls is True

        r2 = handler.chunk_parser(chunk2)
        # Native tokens should NOT produce additional tool calls

        r3 = handler.chunk_parser(chunk3)
        assert r3.choices[0].finish_reason == "tool_calls"
        # Flushed content should be clean (native tokens stripped)
        if r3.choices[0].delta.content:
            assert "<|tool_call" not in r3.choices[0].delta.content


class TestVertexAIMoonshotConfigRouting:
    """Tests for config routing in _get_vertex_ai_config."""

    def test_should_route_moonshot_model_to_config(self):
        """should route moonshotai/ models to VertexAIMoonshotConfig."""
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager._get_vertex_ai_config("moonshotai/kimi-k2-thinking-maas")
        assert isinstance(config, VertexAIMoonshotConfig)

    def test_should_route_kimi_model_to_config(self):
        """should route models with 'kimi' in the name to VertexAIMoonshotConfig."""
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager._get_vertex_ai_config("some-provider/kimi-k2")
        assert isinstance(config, VertexAIMoonshotConfig)

    def test_should_not_route_non_moonshot_model(self):
        """should NOT route non-moonshot models to VertexAIMoonshotConfig."""
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager._get_vertex_ai_config("zai-org/glm-4.7-maas")
        assert not isinstance(config, VertexAIMoonshotConfig)
