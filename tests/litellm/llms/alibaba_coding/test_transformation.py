"""
Unit tests for Alibaba Coding chat transformation.

Tests configuration, parameter handling, and native tool call parsing
for Kimi K2 models on Alibaba Coding.
"""

import pytest

from litellm.llms.alibaba_coding.chat.transformation import (
    AlibabaCodingChatConfig,
    DEFAULT_USER_AGENT,
)
from litellm.llms.alibaba_coding.chat.tool_call_parser import (
    TOOL_CALLS_SECTION_BEGIN,
    TOOL_CALLS_SECTION_END,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    TOOL_CALL_ARGUMENT_BEGIN,
)
from litellm.types.utils import (
    Choices,
    Message,
    ModelResponse,
    ChatCompletionMessageToolCall,
    Function,
)


class TestAlibabaCodingChatConfig:
    """Test Alibaba Coding chat configuration."""

    def setup_method(self):
        self.config = AlibabaCodingChatConfig()
        self.model = "alibaba_coding/kimi-k2.5"

    def test_should_instantiate(self):
        assert self.config is not None

    def test_should_return_default_provider_info(self):
        api_base, api_key = self.config._get_openai_compatible_provider_info(
            api_base=None, api_key=None
        )
        assert api_base == "https://coding-intl.dashscope.aliyuncs.com/v1"
        assert api_key == ""

    def test_should_use_custom_provider_info(self):
        api_base, api_key = self.config._get_openai_compatible_provider_info(
            api_base="https://custom.api.com/v1", api_key="test-key-123"
        )
        assert api_base == "https://custom.api.com/v1"
        assert api_key == "test-key-123"

    def test_should_support_thinking_and_reasoning_effort_params(self):
        params = self.config.get_supported_openai_params(self.model)
        assert "thinking" in params
        assert "reasoning_effort" in params
        assert "temperature" in params

    def test_should_map_thinking_enabled_dict(self):
        result = self.config.map_openai_params(
            non_default_params={"thinking": {"type": "enabled"}},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["chat_template_kwargs"]["enable_thinking"] is True

    def test_should_map_thinking_disabled_dict(self):
        result = self.config.map_openai_params(
            non_default_params={"thinking": {"type": "disabled"}},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["chat_template_kwargs"]["enable_thinking"] is False

    def test_should_map_thinking_bool_true(self):
        result = self.config.map_openai_params(
            non_default_params={"thinking": True},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["chat_template_kwargs"]["enable_thinking"] is True

    def test_should_map_thinking_bool_false(self):
        result = self.config.map_openai_params(
            non_default_params={"thinking": False},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["chat_template_kwargs"]["enable_thinking"] is False

    def test_should_map_reasoning_effort_high(self):
        result = self.config.map_openai_params(
            non_default_params={"reasoning_effort": "high"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["chat_template_kwargs"]["enable_thinking"] is True

    def test_should_map_reasoning_effort_none(self):
        result = self.config.map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["chat_template_kwargs"]["enable_thinking"] is False

    def test_should_pass_through_standard_params(self):
        result = self.config.map_openai_params(
            non_default_params={"temperature": 0.7, "max_tokens": 1000},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1000

    def test_should_detect_native_tool_call_model(self):
        assert self.config._is_native_tool_call_model("kimi-k2.5") is True
        assert self.config._is_native_tool_call_model("kimi-K2-Instruct") is True
        assert self.config._is_native_tool_call_model("gpt-4o") is False


class TestParseToolCallsFromResponse:
    """Test _parse_tool_calls_from_response on non-streaming responses."""

    def setup_method(self):
        self.config = AlibabaCodingChatConfig()

    def _make_response(self, content=None, tool_calls=None, finish_reason="stop",
                       reasoning_content=None) -> ModelResponse:
        """Create a ModelResponse with given content."""
        message = Message(content=content, role="assistant")
        if tool_calls:
            message.tool_calls = tool_calls
        if reasoning_content:
            message.reasoning_content = reasoning_content
        choice = Choices(
            message=message,
            index=0,
            finish_reason=finish_reason,
        )
        resp = ModelResponse()
        resp.choices = [choice]
        return resp

    def test_should_parse_native_tool_calls_from_content(self):
        content = (
            f"{TOOL_CALLS_SECTION_BEGIN}"
            f"{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}"
            f'{{"city": "Tokyo"}}{TOOL_CALL_END}'
            f"{TOOL_CALLS_SECTION_END}"
        )
        response = self._make_response(content=content)
        parsed = self.config._parse_tool_calls_from_response(response)

        message = parsed.choices[0].message
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].function.name == "get_weather"
        assert message.tool_calls[0].function.arguments == '{"city": "Tokyo"}'

    def test_should_fix_finish_reason_when_tool_calls_found(self):
        content = (
            f"{TOOL_CALL_BEGIN}functions.search:0{TOOL_CALL_ARGUMENT_BEGIN}"
            f'{{"q": "test"}}{TOOL_CALL_END}'
        )
        response = self._make_response(content=content, finish_reason="stop")
        parsed = self.config._parse_tool_calls_from_response(response)

        assert parsed.choices[0].finish_reason == "tool_calls"

    def test_should_extract_think_tags_to_reasoning_content(self):
        content = "<think>I need to analyze this carefully</think>Here is the answer"
        response = self._make_response(content=content)
        parsed = self.config._parse_tool_calls_from_response(response)

        message = parsed.choices[0].message
        assert message.reasoning_content is not None
        assert "I need to analyze this carefully" in message.reasoning_content
        assert message.content == "Here is the answer"

    def test_should_handle_unclosed_think_tag(self):
        content = "<think>Still thinking about this problem..."
        response = self._make_response(content=content)
        parsed = self.config._parse_tool_calls_from_response(response)

        message = parsed.choices[0].message
        assert message.reasoning_content is not None
        assert "Still thinking about this problem..." in message.reasoning_content

    def test_should_deduplicate_reasoning_across_fields(self):
        """Test that duplicate reasoning from multiple fields is deduplicated."""
        content = "<think>Same reasoning</think>Answer"
        response = self._make_response(content=content)
        # Manually set reasoning_content to test dedup
        response.choices[0].message.reasoning_content = "Same reasoning"
        parsed = self.config._parse_tool_calls_from_response(response)

        message = parsed.choices[0].message
        assert message.reasoning_content is not None
        assert message.reasoning_content.count("Same reasoning") == 1

    def test_should_cleanup_content_when_tool_calls_already_exist(self):
        """When tool_calls already exist, just clean up content fields."""
        tool_content = (
            f"Some text {TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}"
            f'{{"city": "Tokyo"}}{TOOL_CALL_END}'
        )
        existing_tc = [
            ChatCompletionMessageToolCall(
                id="call_123",
                type="function",
                function=Function(name="  get_weather  ", arguments='{"city": "Tokyo"}'),
            )
        ]
        response = self._make_response(content=tool_content, tool_calls=existing_tc)
        parsed = self.config._parse_tool_calls_from_response(response)

        message = parsed.choices[0].message
        # Function name should be stripped
        assert message.tool_calls[0].function.name == "get_weather"
        # Native tokens should be stripped from content
        content = message.content or ""
        assert TOOL_CALL_BEGIN not in content

    def test_should_fix_finish_reason_for_existing_tool_calls(self):
        existing_tc = [
            ChatCompletionMessageToolCall(
                id="call_123",
                type="function",
                function=Function(name="get_weather", arguments='{}'),
            )
        ]
        response = self._make_response(
            content="text", tool_calls=existing_tc, finish_reason="stop"
        )
        parsed = self.config._parse_tool_calls_from_response(response)

        assert parsed.choices[0].finish_reason == "tool_calls"

    def test_should_handle_empty_response(self):
        resp = ModelResponse()
        resp.choices = []
        result = self.config._parse_tool_calls_from_response(resp)
        assert len(result.choices) == 0

    def test_should_handle_content_without_tool_calls(self):
        response = self._make_response(content="Just regular text")
        parsed = self.config._parse_tool_calls_from_response(response)

        message = parsed.choices[0].message
        assert message.tool_calls is None
        assert message.content == "Just regular text"

    def test_should_parse_multiple_tool_calls(self):
        content = (
            f"{TOOL_CALLS_SECTION_BEGIN}"
            f"{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}"
            f'{{"city": "Tokyo"}}{TOOL_CALL_END}'
            f"{TOOL_CALL_BEGIN}functions.get_time:1{TOOL_CALL_ARGUMENT_BEGIN}"
            f'{{"timezone": "JST"}}{TOOL_CALL_END}'
            f"{TOOL_CALLS_SECTION_END}"
        )
        response = self._make_response(content=content)
        parsed = self.config._parse_tool_calls_from_response(response)

        message = parsed.choices[0].message
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 2
        assert message.tool_calls[0].function.name == "get_weather"
        assert message.tool_calls[1].function.name == "get_time"
        assert parsed.choices[0].finish_reason == "tool_calls"

    def test_should_handle_think_tags_with_tool_calls(self):
        """Think tags + native tool calls in same content."""
        content = (
            "<think>I should check the weather</think>"
            f"{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}"
            f'{{"city": "Tokyo"}}{TOOL_CALL_END}'
        )
        response = self._make_response(content=content)
        parsed = self.config._parse_tool_calls_from_response(response)

        message = parsed.choices[0].message
        assert message.reasoning_content is not None
        assert "I should check the weather" in message.reasoning_content
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1
