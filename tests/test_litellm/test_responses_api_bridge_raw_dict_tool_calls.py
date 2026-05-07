"""
Tests for _convert_response_output_to_choices with raw dict items.

This covers the GitHub Copilot scenario where Responses API returns output
items as raw dicts instead of typed objects (ResponseFunctionToolCall, etc.).

Key regression tests:
1. Raw dict function_call items should be accumulated (not individual choices)
2. When message + tool calls exist, they should merge into one choice
3. finish_reason must be "tool_calls" when tool calls are present
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)


class TestRawDictFunctionCallAccumulation:
    """Test that raw dict function_call items are accumulated like typed ones."""

    def test_should_accumulate_single_raw_dict_function_call(self):
        """Single raw dict function_call should produce one choice with finish_reason=tool_calls."""
        handler = LiteLLMResponsesTransformationHandler()

        output_items = [
            {
                "type": "function_call",
                "call_id": "call_abc123",
                "name": "search_files",
                "arguments": '{"query": "hello"}',
            }
        ]

        choices = handler._convert_response_output_to_choices(
            output_items=output_items,
            handle_raw_dict_callback=handler._handle_raw_dict_response_item,
        )

        assert len(choices) == 1
        assert choices[0].finish_reason == "tool_calls"
        assert choices[0].message.tool_calls is not None
        assert len(choices[0].message.tool_calls) == 1
        assert choices[0].message.tool_calls[0]["id"] == "call_abc123"
        assert choices[0].message.tool_calls[0]["function"]["name"] == "search_files"
        assert (
            choices[0].message.tool_calls[0]["function"]["arguments"]
            == '{"query": "hello"}'
        )

    def test_should_accumulate_multiple_raw_dict_function_calls(self):
        """Multiple raw dict function_calls should be merged into ONE choice."""
        handler = LiteLLMResponsesTransformationHandler()

        output_items = [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "search_files",
                "arguments": '{"query": "foo"}',
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "read_file",
                "arguments": '{"path": "/tmp/test.py"}',
            },
            {
                "type": "function_call",
                "call_id": "call_3",
                "name": "write_file",
                "arguments": '{"path": "/tmp/out.py", "content": "print(1)"}',
            },
        ]

        choices = handler._convert_response_output_to_choices(
            output_items=output_items,
            handle_raw_dict_callback=handler._handle_raw_dict_response_item,
        )

        # Should be ONE choice, not three
        assert len(choices) == 1
        assert choices[0].finish_reason == "tool_calls"
        assert len(choices[0].message.tool_calls) == 3
        assert choices[0].message.tool_calls[0]["id"] == "call_1"
        assert choices[0].message.tool_calls[1]["id"] == "call_2"
        assert choices[0].message.tool_calls[2]["id"] == "call_3"


class TestMessageAndToolCallsMerge:
    """Test that message + tool calls merge into one choice with finish_reason=tool_calls."""

    def test_should_merge_message_and_function_calls_into_single_choice(self):
        """
        GitHub Copilot typical response: message + function_calls.
        Should produce ONE choice with finish_reason='tool_calls', not separate choices.

        Before fix: choices[0]=message(stop), choices[1]=tool(tool_calls) → app stops!
        After fix:  choices[0]=message+tools(tool_calls) → app continues tool loop!
        """
        handler = LiteLLMResponsesTransformationHandler()

        output_items = [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I'll search for that file.",
                    }
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "search_files",
                "arguments": '{"query": "main.py"}',
            },
        ]

        choices = handler._convert_response_output_to_choices(
            output_items=output_items,
            handle_raw_dict_callback=handler._handle_raw_dict_response_item,
        )

        # Must be ONE choice
        assert len(choices) == 1

        choice = choices[0]
        # finish_reason MUST be tool_calls (not stop!)
        assert choice.finish_reason == "tool_calls"

        # Message content should be preserved
        assert choice.message.content == "I'll search for that file."

        # Tool calls should be merged into the same message
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) == 1
        assert choice.message.tool_calls[0]["id"] == "call_abc"
        assert choice.message.tool_calls[0]["function"]["name"] == "search_files"

    def test_should_merge_message_with_multiple_function_calls(self):
        """Message + multiple function_calls → one choice with all tool calls."""
        handler = LiteLLMResponsesTransformationHandler()

        output_items = [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Let me check multiple files.",
                    }
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": '{"path": "src/main.py"}',
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "read_file",
                "arguments": '{"path": "src/utils.py"}',
            },
        ]

        choices = handler._convert_response_output_to_choices(
            output_items=output_items,
            handle_raw_dict_callback=handler._handle_raw_dict_response_item,
        )

        assert len(choices) == 1
        assert choices[0].finish_reason == "tool_calls"
        assert choices[0].message.content == "Let me check multiple files."
        assert len(choices[0].message.tool_calls) == 2

    def test_should_handle_reasoning_message_and_function_calls(self):
        """Reasoning + message + function_calls → one choice with reasoning_content."""
        handler = LiteLLMResponsesTransformationHandler()

        output_items = [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "Thinking about this..."}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I need to search.",
                    }
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_xyz",
                "name": "grep_search",
                "arguments": '{"pattern": "def main"}',
            },
        ]

        choices = handler._convert_response_output_to_choices(
            output_items=output_items,
            handle_raw_dict_callback=handler._handle_raw_dict_response_item,
        )

        assert len(choices) == 1
        assert choices[0].finish_reason == "tool_calls"
        assert choices[0].message.content == "I need to search."
        assert len(choices[0].message.tool_calls) == 1


class TestMessageOnlyStillWorks:
    """Ensure message-only responses still return finish_reason=stop."""

    def test_should_return_stop_when_no_tool_calls(self):
        """Message without tool calls should still have finish_reason=stop."""
        handler = LiteLLMResponsesTransformationHandler()

        output_items = [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "The answer is 42.",
                    }
                ],
            },
        ]

        choices = handler._convert_response_output_to_choices(
            output_items=output_items,
            handle_raw_dict_callback=handler._handle_raw_dict_response_item,
        )

        assert len(choices) == 1
        assert choices[0].finish_reason == "stop"
        assert choices[0].message.content == "The answer is 42."
        assert choices[0].message.tool_calls is None


class TestProviderSpecificFieldsPreserved:
    """Test that provider_specific_fields are preserved in raw dict tool calls."""

    def test_should_preserve_provider_specific_fields(self):
        """provider_specific_fields should pass through in accumulated tool calls."""
        handler = LiteLLMResponsesTransformationHandler()

        output_items = [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "test_tool",
                "arguments": "{}",
                "provider_specific_fields": {"encrypted_content": "abc123"},
            },
        ]

        choices = handler._convert_response_output_to_choices(
            output_items=output_items,
            handle_raw_dict_callback=handler._handle_raw_dict_response_item,
        )

        assert len(choices) == 1
        tc = choices[0].message.tool_calls[0]
        assert tc["provider_specific_fields"]["encrypted_content"] == "abc123"


class TestFunctionCallOnlyNoMessage:
    """Test function_call items without a preceding message."""

    def test_should_create_choice_with_null_content_when_no_message(self):
        """Function calls without message → new choice with content=None."""
        handler = LiteLLMResponsesTransformationHandler()

        output_items = [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "do_something",
                "arguments": '{"x": 1}',
            },
        ]

        choices = handler._convert_response_output_to_choices(
            output_items=output_items,
            handle_raw_dict_callback=handler._handle_raw_dict_response_item,
        )

        assert len(choices) == 1
        assert choices[0].finish_reason == "tool_calls"
        assert choices[0].message.content is None
        assert len(choices[0].message.tool_calls) == 1


class TestStreamingFinishReason:
    """Test that streaming chunks have correct finish_reason for tool calls."""

    def test_should_return_tool_calls_finish_reason_on_completed_with_function_calls(
        self,
    ):
        """response.completed with function_call outputs → finish_reason='tool_calls'."""
        from litellm.completion_extras.litellm_responses_transformation.transformation import (
            OpenAiResponsesToChatCompletionStreamIterator,
        )

        chunk = {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Let me search."}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "search",
                        "arguments": '{"q": "test"}',
                    },
                ],
            },
        }

        result = OpenAiResponsesToChatCompletionStreamIterator.translate_responses_chunk_to_openai_stream(
            chunk
        )

        assert result.choices[0].finish_reason == "tool_calls"

    def test_should_return_stop_finish_reason_on_completed_without_function_calls(self):
        """response.completed without function_call outputs → finish_reason='stop'."""
        from litellm.completion_extras.litellm_responses_transformation.transformation import (
            OpenAiResponsesToChatCompletionStreamIterator,
        )

        chunk = {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "The answer is 42."}],
                    },
                ],
            },
        }

        result = OpenAiResponsesToChatCompletionStreamIterator.translate_responses_chunk_to_openai_stream(
            chunk
        )

        assert result.choices[0].finish_reason == "stop"

    def test_should_return_tool_calls_finish_reason_with_multiple_function_calls(self):
        """response.completed with multiple function_calls → finish_reason='tool_calls'."""
        from litellm.completion_extras.litellm_responses_transformation.transformation import (
            OpenAiResponsesToChatCompletionStreamIterator,
        )

        chunk = {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": '{"path": "a.py"}',
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "read_file",
                        "arguments": '{"path": "b.py"}',
                    },
                ],
            },
        }

        result = OpenAiResponsesToChatCompletionStreamIterator.translate_responses_chunk_to_openai_stream(
            chunk
        )

        assert result.choices[0].finish_reason == "tool_calls"

    def test_should_handle_completed_with_empty_response(self):
        """response.completed with no response data → finish_reason='stop'."""
        from litellm.completion_extras.litellm_responses_transformation.transformation import (
            OpenAiResponsesToChatCompletionStreamIterator,
        )

        chunk = {
            "type": "response.completed",
        }

        result = OpenAiResponsesToChatCompletionStreamIterator.translate_responses_chunk_to_openai_stream(
            chunk
        )

        assert result.choices[0].finish_reason == "stop"
