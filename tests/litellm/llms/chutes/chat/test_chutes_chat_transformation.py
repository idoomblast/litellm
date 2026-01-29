"""
Unit tests for Chutes chat transformation.

Tests the configuration and parameter handling for Chutes models,
including Kimi K2 native tool call support.
"""

from typing import List, cast

import pytest
from litellm.llms.chutes.chat.transformation import ChutesChatConfig
from litellm.llms.chutes.chat.streaming_handler import ChutesChatCompletionStreamingHandler
from litellm.llms.chutes.chat.kimi_k2_tool_call_parser import (
    TOOL_CALLS_SECTION_BEGIN,
    TOOL_CALLS_SECTION_END,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    TOOL_CALL_ARGUMENT_BEGIN,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Choices, Message


class TestChutesChatConfig:
    """Test Chutes chat configuration."""

    def setup_method(self):
        self.config = ChutesChatConfig()
        self.model = "chutes/model-name"

    def test_get_supported_openai_params(self):
        """Test that standard OpenAI params are supported including thinking."""
        params = self.config.get_supported_openai_params(self.model)
        # Chutes supports standard OpenAI optional params
        assert "temperature" in params
        assert "max_tokens" in params
        assert "top_p" in params
        assert "stream" in params
        # Chutes supports thinking parameter
        assert "thinking" in params

    def test_thinking_parameter_enabled(self):
        """Test that thinking parameter with enabled type maps to chat_template_kwargs."""
        non_default_params = {
            "thinking": {"type": "enabled"},
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Verify the new mapping to chat_template_kwargs
        assert "chat_template_kwargs" in result
        assert result["chat_template_kwargs"]["enable_thinking"] is True

    def test_thinking_parameter_disabled(self):
        """Test that thinking parameter with disabled type maps to chat_template_kwargs."""
        non_default_params = {
            "thinking": {"type": "disabled"},
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Verify the new mapping to chat_template_kwargs
        assert "chat_template_kwargs" in result
        assert result["chat_template_kwargs"]["enable_thinking"] is False

    def test_map_openai_params_passes_through(self):
        """Test that standard params are passed through correctly."""
        non_default_params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1000
        assert result["top_p"] == 0.9

    def test_transform_messages_passes_through(self):
        """Test that messages are passed through without transformation."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, how can I help?"},
        ]

        # Cast messages to AllMessageValues type for type checking
        from litellm.types.llms.openai import AllMessageValues
        from typing import cast
        typed_messages = cast(List[AllMessageValues], messages)

        result = self.config._transform_messages(
            messages=typed_messages, model=self.model, is_async=False
        )

        # Messages should be unchanged (OpenAI format)
        assert result[0]["role"] == "user"
        assert result[0].get("content") == "Hello, how are you?"
        assert result[1]["role"] == "assistant"
        assert result[1].get("content") == "I'm doing well, how can I help?"

    def test_get_openai_compatible_provider_info_default_api_base(self):
        """Test that default API base is correctly set."""
        api_base, api_key = self.config._get_openai_compatible_provider_info(
            api_base=None, api_key=None
        )

        assert api_base == "https://llm.chutes.ai/v1"
        assert api_key == ""

    def test_get_openai_compatible_provider_info_custom_api_base(self):
        """Test that custom API base is used when provided."""
        custom_base = "https://custom.chutes.ai/v1"
        api_base, api_key = self.config._get_openai_compatible_provider_info(
            api_base=custom_base, api_key="test-key"
        )

        assert api_base == custom_base
        assert api_key == "test-key"

    def test_get_openai_compatible_provider_info_custom_api_key(self):
        """Test that custom API key is used when provided."""
        custom_key = "my-custom-api-key"
        api_base, api_key = self.config._get_openai_compatible_provider_info(
            api_base=None, api_key=custom_key
        )

        assert api_base == "https://llm.chutes.ai/v1"
        assert api_key == custom_key

    def test_reasoning_effort_low_medium_high(self):
        """Test that reasoning_effort values 'low', 'medium', 'high' map to enable_thinking=True."""
        for effort in ["low", "medium", "high"]:
            non_default_params = {
                "reasoning_effort": effort,
            }
            optional_params = {}

            result = self.config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=self.model,
                drop_params=False,
            )

            assert "chat_template_kwargs" in result
            assert result["chat_template_kwargs"]["enable_thinking"] is True

    def test_reasoning_effort_none_minimal(self):
        """Test that reasoning_effort values 'none', 'minimal' map to enable_thinking=False."""
        for effort in ["none", "minimal"]:
            non_default_params = {
                "reasoning_effort": effort,
            }
            optional_params = {}

            result = self.config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=self.model,
                drop_params=False,
            )

            assert "chat_template_kwargs" in result
            assert result["chat_template_kwargs"]["enable_thinking"] is False

    def test_thinking_with_existing_chat_template_kwargs(self):
        """Test that thinking parameter properly merges with existing chat_template_kwargs."""
        non_default_params = {
            "thinking": {"type": "enabled"},
        }
        optional_params = {
            "chat_template_kwargs": {"existing_param": "value"},
        }

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should merge, not overwrite
        assert "chat_template_kwargs" in result
        assert result["chat_template_kwargs"]["enable_thinking"] is True
        assert result["chat_template_kwargs"]["existing_param"] == "value"

    def test_no_chat_template_kwargs_when_no_thinking(self):
        """Test that no chat_template_kwargs is created when no thinking/reasoning_effort provided."""
        non_default_params = {
            "temperature": 0.5,
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should not have chat_template_kwargs when no thinking params
        assert "chat_template_kwargs" not in result

    def test_thinking_direct_boolean(self):
        """Test that direct boolean thinking values work correctly."""
        # Test thinking=True
        non_default_params_true = {
            "thinking": True,
        }
        result_true = self.config.map_openai_params(
            non_default_params=non_default_params_true,
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result_true["chat_template_kwargs"]["enable_thinking"] is True

        # Test thinking=False
        non_default_params_false = {
            "thinking": False,
        }
        result_false = self.config.map_openai_params(
            non_default_params=non_default_params_false,
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result_false["chat_template_kwargs"]["enable_thinking"] is False


class TestChutesChatConfigKimiK2:
    """Test Kimi K2 native tool call support."""

    def setup_method(self):
        self.config = ChutesChatConfig()

    def test_is_kimi_k2_model(self):
        """Test Kimi K2 model detection."""
        assert self.config._is_kimi_k2_model("moonshotai/Kimi-K2") is True
        assert self.config._is_kimi_k2_model("chutes/moonshotai/Kimi-K2") is True
        assert self.config._is_kimi_k2_model("kimi-k2-instruct") is True
        assert self.config._is_kimi_k2_model("gpt-4") is False
        assert self.config._is_kimi_k2_model("claude-3") is False

    def test_parse_kimi_k2_tool_calls_from_response(self):
        """Test parsing Kimi K2 tool calls from response content."""
        tool_call_content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        # Create a mock ModelResponse
        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=tool_call_content,
                    ),
                    finish_reason="stop",
                )
            ],
            model="kimi-k2",
        )

        # Parse tool calls
        parsed_response = self.config._parse_kimi_k2_tool_calls_from_response(response)

        # Verify tool calls were extracted
        assert parsed_response.choices[0].message.tool_calls is not None
        assert len(parsed_response.choices[0].message.tool_calls) == 1
        assert parsed_response.choices[0].message.tool_calls[0].id == "functions.get_weather:0"
        assert parsed_response.choices[0].message.tool_calls[0].function.name == "get_weather"
        assert parsed_response.choices[0].message.tool_calls[0].function.arguments == '{"city": "Beijing"}'

        # Content should be cleaned (empty -> None)
        assert parsed_response.choices[0].message.content is None

    def test_parse_tool_calls_preserves_mixed_content(self):
        """Test that text before/after tool calls is preserved."""
        tool_call_content = (
            "Here is some analysis.\n"
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}\n"
            "Let me know if you need more help."
        )

        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=tool_call_content,
                    ),
                    finish_reason="stop",
                )
            ],
            model="kimi-k2",
        )

        parsed_response = self.config._parse_kimi_k2_tool_calls_from_response(response)

        # Tool calls should be extracted
        assert parsed_response.choices[0].message.tool_calls is not None
        assert len(parsed_response.choices[0].message.tool_calls) == 1

        # Surrounding text should be preserved
        assert "Here is some analysis." in parsed_response.choices[0].message.content
        assert "Let me know if you need more help." in parsed_response.choices[0].message.content

    def test_parse_tool_calls_from_reasoning_content(self):
        """Test parsing tool calls from reasoning_content field."""
        tool_call_content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.analyze:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"data": "test"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content="Regular response",
                        reasoning_content=tool_call_content,
                    ),
                    finish_reason="stop",
                )
            ],
            model="kimi-k2",
        )

        parsed_response = self.config._parse_kimi_k2_tool_calls_from_response(response)

        # Tool calls should be extracted from reasoning_content
        assert parsed_response.choices[0].message.tool_calls is not None
        assert len(parsed_response.choices[0].message.tool_calls) == 1
        assert parsed_response.choices[0].message.tool_calls[0].function.name == "analyze"

        # Regular content should be preserved
        assert parsed_response.choices[0].message.content == "Regular response"
        # reasoning_content should be cleaned
        assert parsed_response.choices[0].message.reasoning_content is None

    def test_no_tool_calls_passes_through(self):
        """Test that responses without tool calls pass through unchanged."""
        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content="Just a regular response without tool calls.",
                    ),
                    finish_reason="stop",
                )
            ],
            model="kimi-k2",
        )

        parsed_response = self.config._parse_kimi_k2_tool_calls_from_response(response)

        # Should remain unchanged
        assert parsed_response.choices[0].message.content == "Just a regular response without tool calls."
        assert parsed_response.choices[0].message.tool_calls is None

    def test_existing_tool_calls_not_overwritten(self):
        """Test that existing tool_calls are not overwritten."""
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        existing_tool_call = ChatCompletionMessageToolCall(
            id="existing-id",
            type="function",
            function=Function(name="existing_func", arguments="{}"),
        )

        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content="Some content",
                        tool_calls=[existing_tool_call],
                    ),
                    finish_reason="stop",
                )
            ],
            model="kimi-k2",
        )

        parsed_response = self.config._parse_kimi_k2_tool_calls_from_response(response)

        # Existing tool calls should be preserved
        assert parsed_response.choices[0].message.tool_calls is not None
        assert len(parsed_response.choices[0].message.tool_calls) == 1
        assert parsed_response.choices[0].message.tool_calls[0].id == "existing-id"

    def test_get_model_response_iterator(self):
        """Test that custom streaming handler is returned."""
        handler = self.config.get_model_response_iterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

        assert isinstance(handler, ChutesChatCompletionStreamingHandler)

    def test_skip_native_parsing_when_tool_calls_exist(self):
        """Test that native token parsing is skipped when tool_calls already exist."""
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        existing_tool_call = ChatCompletionMessageToolCall(
            id="existing-id",
            type="function",
            function=Function(name="existing_func", arguments='{"arg": "value"}'),
        )

        # Content has both tool calls (existing) AND native tokens
        content_with_native_tokens = (
            "Some text before. "
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
            " Some text after."
        )

        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=content_with_native_tokens,
                        tool_calls=[existing_tool_call],
                    ),
                    finish_reason="stop",
                )
            ],
            model="kimi-k2",
        )

        parsed_response = self.config._parse_kimi_k2_tool_calls_from_response(response)

        # Existing tool calls should be preserved (not overwritten)
        assert parsed_response.choices[0].message.tool_calls is not None
        assert len(parsed_response.choices[0].message.tool_calls) == 1
        assert parsed_response.choices[0].message.tool_calls[0].id == "existing-id"

        # Content should have native tokens stripped
        content = parsed_response.choices[0].message.content
        assert content is not None
        assert "Some text before." in content
        assert "Some text after." in content
        assert TOOL_CALLS_SECTION_BEGIN not in content
        assert TOOL_CALL_BEGIN not in content

    def test_finish_reason_fixed_when_tool_calls_exist(self):
        """Test that finish_reason is changed to 'tool_calls' when tool_calls exist."""
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        existing_tool_call = ChatCompletionMessageToolCall(
            id="existing-id",
            type="function",
            function=Function(name="existing_func", arguments="{}"),
        )

        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content="Some content",
                        tool_calls=[existing_tool_call],
                    ),
                    finish_reason="stop",  # Wrong - should be tool_calls
                )
            ],
            model="kimi-k2",
        )

        parsed_response = self.config._parse_kimi_k2_tool_calls_from_response(response)

        # finish_reason should be corrected to "tool_calls"
        assert parsed_response.choices[0].finish_reason == "tool_calls"

    def test_finish_reason_fixed_when_native_tool_calls_parsed(self):
        """Test that finish_reason is fixed when native tool calls are parsed."""
        tool_call_content = (
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=tool_call_content,
                    ),
                    finish_reason="stop",  # Wrong - should be tool_calls
                )
            ],
            model="kimi-k2",
        )

        parsed_response = self.config._parse_kimi_k2_tool_calls_from_response(response)

        # Tool calls should be extracted
        assert parsed_response.choices[0].message.tool_calls is not None
        # finish_reason should be corrected to "tool_calls"
        assert parsed_response.choices[0].finish_reason == "tool_calls"

    def test_think_tags_stripped_from_content(self):
        """Test that <think>...</think> tags are stripped from content."""
        content_with_think = (
            "<think>Let me analyze this request...</think> "
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=content_with_think,
                    ),
                    finish_reason="stop",
                )
            ],
            model="kimi-k2",
        )

        parsed_response = self.config._parse_kimi_k2_tool_calls_from_response(response)

        # Tool calls should be extracted
        assert parsed_response.choices[0].message.tool_calls is not None

        # Content should be cleaned (either None or empty after stripping)
        content = parsed_response.choices[0].message.content
        if content:
            assert "<think>" not in content
            assert "</think>" not in content

    def test_think_tags_stripped_when_tool_calls_exist(self):
        """Test that think tags are stripped even when tool_calls already exist."""
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        existing_tool_call = ChatCompletionMessageToolCall(
            id="existing-id",
            type="function",
            function=Function(name="existing_func", arguments="{}"),
        )

        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content="<think>Some internal reasoning</think> Final answer.",
                        tool_calls=[existing_tool_call],
                    ),
                    finish_reason="stop",
                )
            ],
            model="kimi-k2",
        )

        parsed_response = self.config._parse_kimi_k2_tool_calls_from_response(response)

        content = parsed_response.choices[0].message.content
        assert content is not None
        assert "<think>" not in content
        assert "</think>" not in content
        assert "Final answer." in content

        # Thinking content should be in reasoning_content
        reasoning = parsed_response.choices[0].message.reasoning_content
        assert reasoning is not None
        assert "Some internal reasoning" in reasoning


class TestThinkTagToReasoningContent:
    """Test transformation of <think> tags to reasoning_content in non-streaming."""

    def setup_method(self):
        self.config = ChutesChatConfig()

    def test_think_content_to_reasoning_content(self):
        """Test that <think>...</think> content is extracted to reasoning_content."""
        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content="<think>Let me analyze this</think>Here is my answer",
                    ),
                    finish_reason="stop",
                )
            ],
            model="kimi-k2",
        )

        parsed_response = self.config._parse_kimi_k2_tool_calls_from_response(response)

        # Reasoning content should have the thinking
        reasoning = parsed_response.choices[0].message.reasoning_content
        assert reasoning is not None
        assert "Let me analyze this" in reasoning

        # Content should have the answer
        content = parsed_response.choices[0].message.content
        assert content is not None
        assert "Here is my answer" in content
        assert "<think>" not in content
        assert "</think>" not in content

    def test_unclosed_think_tag_non_streaming(self):
        """Test that unclosed <think> tag treats all following content as reasoning."""
        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content="<think>Thinking but never finished closing the tag...",
                    ),
                    finish_reason="stop",
                )
            ],
            model="kimi-k2",
        )

        parsed_response = self.config._parse_kimi_k2_tool_calls_from_response(response)

        # All content after <think> should be in reasoning_content
        reasoning = parsed_response.choices[0].message.reasoning_content
        assert reasoning is not None
        assert "Thinking but never finished closing the tag..." in reasoning

        # Content should be None (nothing before the think tag)
        content = parsed_response.choices[0].message.content
        assert content is None

    def test_mixed_content_with_think_tags(self):
        """Test content before, inside, and after think tags."""
        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content="Hello <think>internal thought</think> world!",
                    ),
                    finish_reason="stop",
                )
            ],
            model="kimi-k2",
        )

        parsed_response = self.config._parse_kimi_k2_tool_calls_from_response(response)

        # Content should have text before and after
        content = parsed_response.choices[0].message.content
        assert content is not None
        assert "Hello" in content
        assert "world!" in content
        assert "<think>" not in content

        # Reasoning should have the thinking
        reasoning = parsed_response.choices[0].message.reasoning_content
        assert reasoning is not None
        assert "internal thought" in reasoning

    def test_multiple_think_blocks(self):
        """Test multiple <think>...</think> blocks are all extracted."""
        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content="<think>first</think>middle<think>second</think>end",
                    ),
                    finish_reason="stop",
                )
            ],
            model="kimi-k2",
        )

        parsed_response = self.config._parse_kimi_k2_tool_calls_from_response(response)

        # Content should have middle and end
        content = parsed_response.choices[0].message.content
        assert content is not None
        assert "middle" in content
        assert "end" in content

        # Reasoning should have both thoughts
        reasoning = parsed_response.choices[0].message.reasoning_content
        assert reasoning is not None
        assert "first" in reasoning
        assert "second" in reasoning

    def test_empty_think_block(self):
        """Test empty <think></think> block is handled."""
        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content="<think></think>answer",
                    ),
                    finish_reason="stop",
                )
            ],
            model="kimi-k2",
        )

        parsed_response = self.config._parse_kimi_k2_tool_calls_from_response(response)

        # Content should have the answer
        content = parsed_response.choices[0].message.content
        assert content is not None
        assert "answer" in content

        # Reasoning should be None (empty think block)
        reasoning = getattr(parsed_response.choices[0].message, "reasoning_content", None)
        assert reasoning is None

    def test_append_to_existing_reasoning_content(self):
        """Test that think content is appended to existing reasoning_content."""
        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content="<think>new thought</think>answer",
                        reasoning_content="existing reasoning",
                    ),
                    finish_reason="stop",
                )
            ],
            model="kimi-k2",
        )

        parsed_response = self.config._parse_kimi_k2_tool_calls_from_response(response)

        # Reasoning should have both existing and new
        reasoning = parsed_response.choices[0].message.reasoning_content
        assert reasoning is not None
        assert "existing reasoning" in reasoning
        assert "new thought" in reasoning

        # Content should have the answer
        content = parsed_response.choices[0].message.content
        assert content is not None
        assert "answer" in content

    def test_think_tags_with_tool_calls(self):
        """Test think tags with tool calls (think content extracted, then tool calls parsed)."""
        tool_call_content = (
            "<think>Let me call a function</think>"
            f"{TOOL_CALLS_SECTION_BEGIN}\n"
            f'{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}'
            f'{{"city": "Beijing"}}{TOOL_CALL_END}\n'
            f"{TOOL_CALLS_SECTION_END}"
        )

        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=tool_call_content,
                    ),
                    finish_reason="stop",
                )
            ],
            model="kimi-k2",
        )

        parsed_response = self.config._parse_kimi_k2_tool_calls_from_response(response)

        # Tool calls should be extracted
        assert parsed_response.choices[0].message.tool_calls is not None
        assert len(parsed_response.choices[0].message.tool_calls) == 1
        assert parsed_response.choices[0].message.tool_calls[0].function.name == "get_weather"

        # Reasoning should have the thinking
        reasoning = parsed_response.choices[0].message.reasoning_content
        assert reasoning is not None
        assert "Let me call a function" in reasoning

        # Content should be cleaned
        content = parsed_response.choices[0].message.content
        assert content is None or "<think>" not in content