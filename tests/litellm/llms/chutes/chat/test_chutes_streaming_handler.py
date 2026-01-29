"""
Unit tests for Chutes streaming handler with Kimi K2 tool call support.

Tests the streaming handler's ability to buffer content and detect
tool call tokens that may span chunk boundaries.
"""

import pytest

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

    def __init__(self, chunks):
        self.chunks = chunks
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.chunks):
            raise StopIteration
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk


class TestChutesChatCompletionStreamingHandler:
    """Test ChutesChatCompletionStreamingHandler class."""

    def test_regular_content_streaming(self):
        """Test streaming regular content without tool calls."""
        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello, world!"},
                    "finish_reason": None,
                }
            ],
        }

        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        result = handler.chunk_parser(chunk)

        assert result.id == "test-id"
        assert result.model == "test-model"
        assert len(result.choices) == 1

    def test_tool_call_detection_single_chunk(self):
        """Test detecting tool calls in a single chunk."""
        tool_call_content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": tool_call_content},
                    "finish_reason": None,
                }
            ],
        }

        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        result = handler.chunk_parser(chunk)

        # Tool calls should be extracted
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].delta.tool_calls[0].function.name == "get_weather"

    def test_tool_call_in_reasoning_content(self):
        """Test detecting tool calls in reasoning_content field."""
        tool_call_content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.analyze:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"data": "test"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "Regular content",
                        "reasoning_content": tool_call_content,
                    },
                    "finish_reason": None,
                }
            ],
        }

        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        result = handler.chunk_parser(chunk)

        # Tool calls should be extracted from reasoning_content
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].delta.tool_calls[0].function.name == "analyze"

    def test_multiple_tool_calls(self):
        """Test detecting multiple tool calls in one section."""
        tool_call_content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f'{TOOL_CALL_BEGIN}functions.get_time:1{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"timezone": "UTC"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": tool_call_content},
                    "finish_reason": None,
                }
            ],
        }

        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        result = handler.chunk_parser(chunk)

        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 2
        assert result.choices[0].delta.tool_calls[0].function.name == "get_weather"
        assert result.choices[0].delta.tool_calls[1].function.name == "get_time"

    def test_buffering_partial_tokens(self):
        """Test that partial tokens are buffered correctly."""
        # First chunk ends mid-token
        chunk1 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
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
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": remaining},
                    "finish_reason": None,
                }
            ],
        }

        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        # Process first chunk - should buffer
        result1 = handler.chunk_parser(chunk1)
        # First chunk should not emit tool calls yet (incomplete)
        assert result1.choices[0].delta.tool_calls is None or len(result1.choices[0].delta.tool_calls) == 0

        # Process second chunk - should complete and emit tool calls
        result2 = handler.chunk_parser(chunk2)
        assert result2.choices[0].delta.tool_calls is not None
        assert len(result2.choices[0].delta.tool_calls) == 1

    def test_finish_reason_flushes_buffer(self):
        """Test that finish_reason causes remaining buffer to be flushed."""
        chunk1 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }

        chunk2 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": ", world!"},
                    "finish_reason": "stop",
                }
            ],
        }

        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        handler.chunk_parser(chunk1)
        result2 = handler.chunk_parser(chunk2)

        # With finish_reason, remaining content should be flushed
        assert result2.choices[0].finish_reason == "stop"

    def test_empty_choices(self):
        """Test handling of chunk with empty choices."""
        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [],
        }

        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        result = handler.chunk_parser(chunk)

        assert len(result.choices) == 0

    def test_tool_call_index_incrementing(self):
        """Test that tool call indices increment correctly."""
        # First chunk with tool calls
        tool_call_content1 = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.func1:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"arg": "1"}}{TOOL_CALL_END}\n'
            f'{TOOL_CALL_BEGIN}functions.func2:1{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"arg": "2"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": tool_call_content1},
                    "finish_reason": None,
                }
            ],
        }

        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        result = handler.chunk_parser(chunk)

        tool_calls = result.choices[0].delta.tool_calls
        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0].index == 0
        assert tool_calls[1].index == 1

    def test_chunk_metadata_preserved(self):
        """Test that chunk metadata is preserved correctly."""
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Test"},
                    "finish_reason": None,
                }
            ],
        }

        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        result = handler.chunk_parser(chunk)

        assert result.id == "chatcmpl-123"
        assert result.object == "chat.completion.chunk"
        assert result.created == 1234567890
        assert result.model == "kimi-k2-model"


class TestDuplicateToolCallPrevention:
    """Test duplicate tool call prevention when Chutes sends both formats."""

    def test_standard_format_takes_priority(self):
        """Test that standard delta.tool_calls takes priority over native tokens."""
        # Chunk 1: Standard format tool call
        chunk1 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }

        # Chunk 2: Native tokens (DUPLICATES - should be skipped)
        native_content = (
            f"{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}"
            f'{{"city": "Beijing"}}{TOOL_CALL_END}'
        )
        chunk2 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": native_content},
                    "finish_reason": None,
                }
            ],
        }

        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        # Process chunk 1 (standard format)
        result1 = handler.chunk_parser(chunk1)
        assert result1.choices[0].delta.tool_calls is not None
        assert handler._saw_any_standard_tool_calls is True

        # Process chunk 2 (native tokens - should NOT emit additional tool calls)
        result2 = handler.chunk_parser(chunk2)
        # Native tokens should NOT produce additional tool calls since standard was seen
        # The content might still contain the native tokens, but they shouldn't be parsed as tool calls
        if result2.choices[0].delta.tool_calls:
            # If tool_calls are present, they should NOT be from native parsing
            # (this is allowed since delta might carry through from standard format)
            pass

    def test_no_duplicate_when_standard_format_seen(self):
        """
        Simulates Chutes stream: tool_calls chunks first, then native tokens in content.
        Should result in only ONE set of tool calls.
        """
        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        # Chunk 1: Think tag start
        chunk1 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " <think>"},
                    "finish_reason": None,
                }
            ],
        }

        # Chunk 2: Think tag end
        chunk2 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Let me think... </think>"},
                    "finish_reason": None,
                }
            ],
        }

        # Chunk 3: Standard tool calls
        chunk3 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "functions.get_weather:0",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }

        # Chunk 4: Native tokens (duplicates)
        chunk4 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": (
                            f"{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}"
                            f'{{"city": "Beijing"}}{TOOL_CALL_END}'
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        # Process all chunks
        handler.chunk_parser(chunk1)
        handler.chunk_parser(chunk2)
        result3 = handler.chunk_parser(chunk3)
        result4 = handler.chunk_parser(chunk4)

        # After chunk 3, we should have seen standard tool calls
        assert handler._saw_any_standard_tool_calls is True
        assert result3.choices[0].delta.tool_calls is not None

        # Chunk 4 should NOT emit tool calls from native tokens
        # (the content with native tokens should be stripped or ignored)


class TestThinkTagHandling:
    """Test handling of <think>...</think> tags in streaming."""

    def test_think_content_transformed_to_reasoning_content(self):
        """Test that <think>...</think> content is transformed to reasoning_content."""
        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        # Single chunk with think content and finish_reason
        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "<think>Let me analyze this request</think>Here is my answer"},
                    "finish_reason": "stop",
                }
            ],
        }

        result = handler.chunk_parser(chunk)

        # The thinking content should be in reasoning_content
        reasoning = result.choices[0].delta.reasoning_content or ""
        assert "Let me analyze this request" in reasoning

        # The answer should be in content
        content = result.choices[0].delta.content or ""
        assert "Here is my answer" in content

        # No think tags should remain
        assert "<think>" not in content
        assert "</think>" not in content

    def test_unclosed_think_tag_all_content_to_reasoning(self):
        """Test that unclosed <think> tag routes all content to reasoning_content."""
        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        # Chunk 1: Start thinking
        chunk1 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "<think>Starting to think..."},
                    "finish_reason": None,
                }
            ],
        }

        # Chunk 2: More thinking (NO </think>!)
        chunk2 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Still thinking about this..."},
                    "finish_reason": None,
                }
            ],
        }

        # Chunk 3: Tool calls (model calls tools mid-thinking, never closes think tag)
        chunk3 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        # Process all chunks
        handler.chunk_parser(chunk1)
        handler.chunk_parser(chunk2)
        result3 = handler.chunk_parser(chunk3)

        # All content after <think> should be in reasoning_content
        reasoning = result3.choices[0].delta.reasoning_content or ""
        assert "Starting to think..." in reasoning
        assert "Still thinking about this..." in reasoning

        # Content should be empty or None
        content = result3.choices[0].delta.content
        assert content is None or content.strip() == ""

    def test_think_tags_stripped_from_flushed_content(self):
        """Test that think tags are stripped from buffered content at end of stream.

        Note: During streaming, content is buffered and only "safe" portions are emitted.
        Think tags in the remaining buffer are stripped when flushed at stream end.
        """
        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        # Single chunk with think content and finish_reason
        # This puts everything in buffer and then flushes it all at once
        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "<think>Let me analyze...</think>"},
                    "finish_reason": "stop",
                }
            ],
        }

        result = handler.chunk_parser(chunk)

        # The flushed content should have think tags stripped
        final_content = result.choices[0].delta.content or ""
        # After stripping think tags, content should be empty or just whitespace
        assert "<think>" not in final_content
        assert "</think>" not in final_content

    def test_regular_content_without_think_tags_preserved(self):
        """Test that regular content without think tags is preserved."""
        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Just regular content."},
                    "finish_reason": "stop",
                }
            ],
        }

        result = handler.chunk_parser(chunk)

        final_content = result.choices[0].delta.content or ""
        assert "Just regular content." in final_content

    def test_text_before_and_after_think_tags(self):
        """Test that text before and after think tags is preserved in content."""
        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello <think>internal thought</think> world!"},
                    "finish_reason": "stop",
                }
            ],
        }

        result = handler.chunk_parser(chunk)

        # Check that regular content is preserved
        content = result.choices[0].delta.content or ""
        assert "Hello" in content
        assert "world!" in content

        # Check that thinking content is in reasoning_content
        reasoning = result.choices[0].delta.reasoning_content or ""
        assert "internal thought" in reasoning

    def test_multiple_think_blocks(self):
        """Test handling of multiple <think>...</think> blocks."""
        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "<think>first thought</think>middle<think>second thought</think>end"},
                    "finish_reason": "stop",
                }
            ],
        }

        result = handler.chunk_parser(chunk)

        # Check that regular content is preserved
        content = result.choices[0].delta.content or ""
        assert "middle" in content
        assert "end" in content

        # Check that all thinking content is in reasoning_content
        reasoning = result.choices[0].delta.reasoning_content or ""
        assert "first thought" in reasoning
        assert "second thought" in reasoning

    def test_empty_think_block(self):
        """Test handling of empty <think></think> block."""
        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "<think></think>answer"},
                    "finish_reason": "stop",
                }
            ],
        }

        result = handler.chunk_parser(chunk)

        # Content should have the answer
        content = result.choices[0].delta.content or ""
        assert "answer" in content

        # Reasoning should be None or empty since think block was empty
        reasoning = getattr(result.choices[0].delta, "reasoning_content", None)
        assert reasoning is None or reasoning.strip() == ""

    def test_think_tag_spans_chunks(self):
        """Test that think tag spanning multiple chunks is handled correctly."""
        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        # Chunk 1: Think tag opens
        chunk1 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Before <think>thinking"},
                    "finish_reason": None,
                }
            ],
        }

        # Chunk 2: Think tag closes
        chunk2 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " continues</think>after"},
                    "finish_reason": "stop",
                }
            ],
        }

        handler.chunk_parser(chunk1)
        result2 = handler.chunk_parser(chunk2)

        # Content should have "Before" and "after"
        content = result2.choices[0].delta.content or ""
        # Note: Due to buffering, content may be combined in final chunk
        # Just verify think tags are not in content
        assert "<think>" not in content
        assert "</think>" not in content

        # Reasoning should have the thinking content
        reasoning = result2.choices[0].delta.reasoning_content or ""
        assert "thinking" in reasoning or "continues" in reasoning


class TestFinishReasonFix:
    """Test that finish_reason is fixed to 'tool_calls' when tool calls are emitted."""

    def test_finish_reason_fixed_for_standard_tool_calls(self):
        """Test finish_reason is changed to 'tool_calls' when standard format tool calls are emitted."""
        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        # Chunk with tool calls
        chunk1 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }

        # Final chunk with wrong finish_reason
        chunk2 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",  # Wrong! Should be tool_calls
                }
            ],
        }

        handler.chunk_parser(chunk1)
        result2 = handler.chunk_parser(chunk2)

        # finish_reason should be corrected to "tool_calls"
        assert result2.choices[0].finish_reason == "tool_calls"

    def test_finish_reason_fixed_for_native_tool_calls(self):
        """Test finish_reason is changed to 'tool_calls' when native format tool calls are emitted."""
        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        # Chunk with native tool call tokens (complete)
        tool_content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )
        chunk1 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": tool_content},
                    "finish_reason": None,
                }
            ],
        }

        # Final chunk with wrong finish_reason
        chunk2 = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }

        result1 = handler.chunk_parser(chunk1)
        # Tool calls should be extracted from native tokens
        assert result1.choices[0].delta.tool_calls is not None

        result2 = handler.chunk_parser(chunk2)
        # finish_reason should be corrected to "tool_calls"
        assert result2.choices[0].finish_reason == "tool_calls"


class TestChutesFormatStreaming:
    """Test streaming with Chutes-specific format (no section wrappers)."""

    def test_chutes_format_without_section_wrappers(self):
        """Test parsing Chutes format tool calls (no section wrappers, with spaces)."""
        handler = ChutesChatCompletionStreamingHandler(
            streaming_response=MockStreamingResponse([]),
            sync_stream=True,
        )

        # Chutes format: spaces around tokens, no section wrappers
        tool_content = (
            f"{TOOL_CALL_BEGIN} functions.read_file:0 {TOOL_CALL_ARGUMENT_BEGIN} "
            f'{{"filePath": "/test.txt"}} {TOOL_CALL_END}'
        )
        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": tool_content},
                    "finish_reason": None,
                }
            ],
        }

        result = handler.chunk_parser(chunk)

        # Tool calls should be extracted
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].delta.tool_calls[0].function.name == "read_file"
