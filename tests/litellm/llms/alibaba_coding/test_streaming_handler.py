"""
Unit tests for Alibaba Coding streaming handler with Kimi K2 tool call support.

Tests the AlibabaCodingStreamingHandler (subclass of ChutesChatCompletionStreamingHandler)
for streaming tool call parsing, think tag handling, and duplicate prevention.
"""

import pytest

from litellm.llms.alibaba_coding.chat.streaming_handler import (
    AlibabaCodingStreamingHandler,
)
from litellm.llms.chutes.chat.streaming_handler import (
    ChutesChatCompletionStreamingHandler,
)
from litellm.llms.chutes.chat.kimi_k2_tool_call_parser import (
    TOOL_CALLS_SECTION_BEGIN,
    TOOL_CALLS_SECTION_END,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    TOOL_CALL_ARGUMENT_BEGIN,
)


class MockStreamingResponse:
    """Mock streaming response for testing."""

    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.chunks):
            raise StopIteration
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk


def _make_handler():
    """Create a fresh AlibabaCodingStreamingHandler instance."""
    return AlibabaCodingStreamingHandler(
        streaming_response=MockStreamingResponse([]),
        sync_stream=True,
    )


def _make_chunk(content=None, finish_reason=None, tool_calls=None,
                reasoning=None, reasoning_content=None, thinking=None,
                role=None, chunk_id="test-id", model="kimi-k2.5"):
    """Helper to build a streaming chunk dict."""
    delta = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    if reasoning is not None:
        delta["reasoning"] = reasoning
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    if thinking is not None:
        delta["thinking"] = thinking

    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


class TestAlibabaCodingStreamingHandlerInheritance:
    """Test that AlibabaCodingStreamingHandler is a proper subclass of ChutesChatCompletionStreamingHandler."""

    def test_should_be_subclass_of_chutes(self):
        assert issubclass(AlibabaCodingStreamingHandler, ChutesChatCompletionStreamingHandler)

    def test_should_instantiate_correctly(self):
        handler = _make_handler()
        assert isinstance(handler, AlibabaCodingStreamingHandler)
        assert isinstance(handler, ChutesChatCompletionStreamingHandler)

    def test_should_have_all_state_attributes(self):
        handler = _make_handler()
        assert hasattr(handler, "_field_buffers")
        assert hasattr(handler, "_in_tool_section")
        assert hasattr(handler, "_tool_call_index")
        assert hasattr(handler, "_saw_any_standard_tool_calls")
        assert hasattr(handler, "_emitted_any_tool_calls")
        assert hasattr(handler, "_in_think_block")
        assert hasattr(handler, "_emitted_any_reasoning")


class TestRegularContentStreaming:
    """Test streaming regular content without tool calls."""

    def test_should_stream_regular_content(self):
        handler = _make_handler()
        result = handler.chunk_parser(_make_chunk(role="assistant", content="Hello, world!"))

        assert result.id == "test-id"
        assert result.model == "kimi-k2.5"
        assert len(result.choices) == 1

    def test_should_flush_content_on_finish_reason(self):
        handler = _make_handler()
        handler.chunk_parser(_make_chunk(content="Hello"))
        result = handler.chunk_parser(_make_chunk(content=", world!", finish_reason="stop"))

        assert result.choices[0].finish_reason == "stop"
        content = result.choices[0].delta.content or ""
        assert "Hello" in content or ", world!" in content

    def test_should_handle_empty_choices(self):
        handler = _make_handler()
        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2.5",
            "choices": [],
        }
        result = handler.chunk_parser(chunk)
        assert len(result.choices) == 0

    def test_should_preserve_chunk_metadata(self):
        handler = _make_handler()
        result = handler.chunk_parser(_make_chunk(
            content="Test", chunk_id="chatcmpl-abc", model="kimi-k2.5-test"
        ))

        assert result.id == "chatcmpl-abc"
        assert result.model == "kimi-k2.5-test"
        assert result.object == "chat.completion.chunk"
        assert result.created == 1234567890


class TestToolCallDetection:
    """Test tool call detection with native Kimi K2 tokens."""

    def test_should_detect_tool_call_with_section_wrappers(self):
        handler = _make_handler()
        tool_content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        result = handler.chunk_parser(_make_chunk(content=tool_content))

        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].delta.tool_calls[0].function.name == "get_weather"
        assert result.choices[0].delta.tool_calls[0].function.arguments == '{"city": "Beijing"}'

    def test_should_detect_tool_call_without_section_wrappers(self):
        """Test Chutes format: no section wrappers, spaces around tokens."""
        handler = _make_handler()
        tool_content = (
            f"{TOOL_CALL_BEGIN} functions.read_file:0 {TOOL_CALL_ARGUMENT_BEGIN} "
            f'{{"filePath": "/test.txt"}} {TOOL_CALL_END}'
        )

        result = handler.chunk_parser(_make_chunk(content=tool_content))

        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].delta.tool_calls[0].function.name == "read_file"

    def test_should_detect_multiple_tool_calls(self):
        handler = _make_handler()
        tool_content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f'{TOOL_CALL_BEGIN}functions.get_time:1{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"timezone": "UTC"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        result = handler.chunk_parser(_make_chunk(content=tool_content))

        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 2
        assert result.choices[0].delta.tool_calls[0].function.name == "get_weather"
        assert result.choices[0].delta.tool_calls[1].function.name == "get_time"

    def test_should_increment_tool_call_indices(self):
        handler = _make_handler()
        tool_content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.func1:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"arg": "1"}}{TOOL_CALL_END}\n'
            f'{TOOL_CALL_BEGIN}functions.func2:1{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"arg": "2"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        result = handler.chunk_parser(_make_chunk(content=tool_content))

        tool_calls = result.choices[0].delta.tool_calls
        assert tool_calls is not None
        assert tool_calls[0].index == 0
        assert tool_calls[1].index == 1

    def test_should_detect_tool_calls_in_reasoning_content_field(self):
        handler = _make_handler()
        tool_content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.analyze:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"data": "test"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        result = handler.chunk_parser(_make_chunk(
            content="Regular content",
            reasoning_content=tool_content,
        ))

        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].delta.tool_calls[0].function.name == "analyze"


class TestBufferingPartialTokens:
    """Test cross-chunk buffering of partial tokens."""

    def test_should_buffer_partial_section_begin_token(self):
        handler = _make_handler()

        # First chunk ends mid-token
        result1 = handler.chunk_parser(_make_chunk(content="<|tool_calls_section"))

        # No tool calls yet
        assert result1.choices[0].delta.tool_calls is None

        # Second chunk completes the token
        remaining = (
            f"_begin|>\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )
        result2 = handler.chunk_parser(_make_chunk(content=remaining))

        assert result2.choices[0].delta.tool_calls is not None
        assert len(result2.choices[0].delta.tool_calls) == 1


class TestDuplicateToolCallPrevention:
    """Test duplicate tool call prevention when API sends both formats."""

    def test_should_prioritize_standard_format(self):
        handler = _make_handler()

        # Chunk with standard format tool call
        result1 = handler.chunk_parser(_make_chunk(
            tool_calls=[{
                "index": 0,
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
            }],
        ))

        assert result1.choices[0].delta.tool_calls is not None
        assert handler._saw_any_standard_tool_calls is True

        # Chunk with native tokens (duplicates — should be skipped)
        native_content = (
            f"{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}"
            f'{{"city": "Beijing"}}{TOOL_CALL_END}'
        )
        result2 = handler.chunk_parser(_make_chunk(content=native_content))

        # Native tokens should NOT produce additional tool calls
        native_tool_calls = result2.choices[0].delta.tool_calls
        assert native_tool_calls is None

    def test_should_skip_native_tokens_after_standard_seen(self):
        """Full flow: think → standard tool_calls → native tokens (skip)."""
        handler = _make_handler()

        # Think tag start
        handler.chunk_parser(_make_chunk(content=" <think>"))
        # Think tag end
        handler.chunk_parser(_make_chunk(content="Let me think... </think>"))
        # Standard tool calls
        result3 = handler.chunk_parser(_make_chunk(
            tool_calls=[{
                "index": 0,
                "id": "functions.get_weather:0",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
            }],
        ))

        assert handler._saw_any_standard_tool_calls is True
        assert result3.choices[0].delta.tool_calls is not None

        # Native tokens (duplicates)
        native_content = (
            f"{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}"
            f'{{"city": "Beijing"}}{TOOL_CALL_END}'
        )
        result4 = handler.chunk_parser(_make_chunk(
            content=native_content,
            finish_reason="stop",
        ))

        # Should not have native tool calls
        assert result4.choices[0].delta.tool_calls is None


class TestThinkTagHandling:
    """Test handling of <think>...</think> tags in streaming."""

    def test_should_transform_think_content_to_reasoning(self):
        handler = _make_handler()
        result = handler.chunk_parser(_make_chunk(
            content="<think>Let me analyze this</think>Here is my answer",
            finish_reason="stop",
        ))

        reasoning = result.choices[0].delta.reasoning_content or ""
        assert "Let me analyze this" in reasoning

        content = result.choices[0].delta.content or ""
        assert "Here is my answer" in content
        assert "<think>" not in content
        assert "</think>" not in content

    def test_should_handle_unclosed_think_tag(self):
        """Unclosed <think> routes all subsequent content to reasoning."""
        handler = _make_handler()

        handler.chunk_parser(_make_chunk(content="<think>Starting to think..."))
        handler.chunk_parser(_make_chunk(content="Still thinking about this..."))

        result = handler.chunk_parser(_make_chunk(
            tool_calls=[{
                "index": 0,
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{}'},
            }],
            finish_reason="tool_calls",
        ))

        reasoning = result.choices[0].delta.reasoning_content or ""
        assert "Starting to think..." in reasoning
        assert "Still thinking about this..." in reasoning

        content = result.choices[0].delta.content
        assert content is None or content.strip() == ""

    def test_should_handle_text_before_and_after_think_tags(self):
        handler = _make_handler()
        result = handler.chunk_parser(_make_chunk(
            content="Hello <think>internal thought</think> world!",
            finish_reason="stop",
        ))

        content = result.choices[0].delta.content or ""
        assert "Hello" in content
        assert "world!" in content

        reasoning = result.choices[0].delta.reasoning_content or ""
        assert "internal thought" in reasoning

    def test_should_handle_multiple_think_blocks(self):
        handler = _make_handler()
        result = handler.chunk_parser(_make_chunk(
            content="<think>first thought</think>middle<think>second thought</think>end",
            finish_reason="stop",
        ))

        content = result.choices[0].delta.content or ""
        assert "middle" in content
        assert "end" in content

        reasoning = result.choices[0].delta.reasoning_content or ""
        assert "first thought" in reasoning
        assert "second thought" in reasoning

    def test_should_handle_empty_think_block(self):
        handler = _make_handler()
        result = handler.chunk_parser(_make_chunk(
            content="<think></think>answer",
            finish_reason="stop",
        ))

        content = result.choices[0].delta.content or ""
        assert "answer" in content

        reasoning = getattr(result.choices[0].delta, "reasoning_content", None)
        assert reasoning is None or reasoning.strip() == ""

    def test_should_handle_think_tag_spanning_chunks(self):
        handler = _make_handler()

        handler.chunk_parser(_make_chunk(content="Before <think>thinking"))
        result = handler.chunk_parser(_make_chunk(
            content=" continues</think>after",
            finish_reason="stop",
        ))

        content = result.choices[0].delta.content or ""
        assert "<think>" not in content
        assert "</think>" not in content

        reasoning = result.choices[0].delta.reasoning_content or ""
        assert "thinking" in reasoning or "continues" in reasoning

    def test_should_preserve_content_without_think_tags(self):
        handler = _make_handler()
        result = handler.chunk_parser(_make_chunk(
            content="Just regular content.",
            finish_reason="stop",
        ))

        content = result.choices[0].delta.content or ""
        assert "Just regular content." in content


class TestFinishReasonFix:
    """Test that finish_reason is fixed to 'tool_calls' when tool calls are emitted."""

    def test_should_fix_finish_reason_for_standard_tool_calls(self):
        handler = _make_handler()

        handler.chunk_parser(_make_chunk(
            tool_calls=[{
                "index": 0,
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{}'},
            }],
        ))

        result = handler.chunk_parser(_make_chunk(finish_reason="stop"))

        assert result.choices[0].finish_reason == "tool_calls"

    def test_should_fix_finish_reason_for_native_tool_calls(self):
        handler = _make_handler()

        tool_content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )
        result1 = handler.chunk_parser(_make_chunk(content=tool_content))
        assert result1.choices[0].delta.tool_calls is not None

        result2 = handler.chunk_parser(_make_chunk(finish_reason="stop"))
        assert result2.choices[0].finish_reason == "tool_calls"

    def test_should_not_change_correct_finish_reason(self):
        handler = _make_handler()

        handler.chunk_parser(_make_chunk(
            tool_calls=[{
                "index": 0,
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{}'},
            }],
        ))

        result = handler.chunk_parser(_make_chunk(finish_reason="tool_calls"))
        assert result.choices[0].finish_reason == "tool_calls"

    def test_should_keep_stop_when_no_tool_calls(self):
        handler = _make_handler()
        handler.chunk_parser(_make_chunk(content="Just text"))
        result = handler.chunk_parser(_make_chunk(content=".", finish_reason="stop"))
        assert result.choices[0].finish_reason == "stop"


class TestReasoningFieldNormalization:
    """Test normalization of reasoning/thinking fields to reasoning_content in streaming."""

    def test_should_normalize_reasoning_field(self):
        handler = _make_handler()
        result = handler.chunk_parser(_make_chunk(
            content="answer",
            reasoning="I need to think carefully",
            finish_reason="stop",
        ))

        reasoning = result.choices[0].delta.reasoning_content or ""
        assert "I need to think carefully" in reasoning

        content = result.choices[0].delta.content or ""
        assert "answer" in content

    def test_should_normalize_thinking_field(self):
        handler = _make_handler()
        result = handler.chunk_parser(_make_chunk(
            content="the answer",
            thinking="Let me work through this step by step",
            finish_reason="stop",
        ))

        reasoning = result.choices[0].delta.reasoning_content or ""
        assert "Let me work through this step by step" in reasoning

    def test_should_deduplicate_same_reasoning_across_fields(self):
        handler = _make_handler()
        same_reasoning = "Let me analyze the problem"
        result = handler.chunk_parser(_make_chunk(
            content="answer",
            reasoning=same_reasoning,
            reasoning_content=same_reasoning,
            thinking=same_reasoning,
            finish_reason="stop",
        ))

        reasoning = result.choices[0].delta.reasoning_content or ""
        assert reasoning.count("Let me analyze the problem") == 1

    def test_should_combine_different_reasoning_across_fields(self):
        handler = _make_handler()
        result = handler.chunk_parser(_make_chunk(
            content="answer",
            reasoning_content="first reasoning",
            reasoning="second reasoning",
            finish_reason="stop",
        ))

        reasoning = result.choices[0].delta.reasoning_content or ""
        assert "first reasoning" in reasoning
        assert "second reasoning" in reasoning

    def test_should_stream_reasoning_incrementally(self):
        handler = _make_handler()

        result1 = handler.chunk_parser(_make_chunk(reasoning="First part. "))
        reasoning1 = result1.choices[0].delta.reasoning_content or ""
        assert "First part." in reasoning1

        result2 = handler.chunk_parser(_make_chunk(
            content="the answer",
            reasoning="Second part.",
            finish_reason="stop",
        ))
        reasoning2 = result2.choices[0].delta.reasoning_content or ""
        assert "Second part." in reasoning2

    def test_should_deduplicate_reasoning_across_chunks_incrementally(self):
        handler = _make_handler()

        result1 = handler.chunk_parser(_make_chunk(
            reasoning="thinking",
            reasoning_content="thinking",
            thinking="thinking",
        ))
        reasoning1 = result1.choices[0].delta.reasoning_content or ""
        assert reasoning1 == "thinking"

        result2 = handler.chunk_parser(_make_chunk(
            content="answer",
            reasoning=" more",
            reasoning_content=" more",
            thinking=" more",
            finish_reason="stop",
        ))
        reasoning2 = result2.choices[0].delta.reasoning_content or ""
        assert reasoning2 == " more"

    def test_should_stream_reasoning_content_token_by_token(self):
        handler = _make_handler()
        tokens = ["this", " is", " reasoning", " example"]
        results = []

        for i, token in enumerate(tokens):
            is_last = i == len(tokens) - 1
            results.append(handler.chunk_parser(_make_chunk(
                reasoning_content=token,
                finish_reason="stop" if is_last else None,
            )))

        assert results[0].choices[0].delta.reasoning_content == "this"
        assert results[1].choices[0].delta.reasoning_content == " is"
        assert results[2].choices[0].delta.reasoning_content == " reasoning"
        assert results[3].choices[0].delta.reasoning_content == " example"

        full = "".join(r.choices[0].delta.reasoning_content or "" for r in results)
        assert full == "this is reasoning example"


class TestStandardToolCallNameStripping:
    """Test that function names are stripped of whitespace in standard format."""

    def test_should_strip_whitespace_from_function_names(self):
        handler = _make_handler()
        result = handler.chunk_parser(_make_chunk(
            tool_calls=[{
                "index": 0,
                "id": "call_123",
                "type": "function",
                "function": {"name": "  get_weather  ", "arguments": '{}'},
            }],
        ))

        tc = result.choices[0].delta.tool_calls
        assert tc is not None
        assert tc[0]["function"]["name"] == "get_weather"
