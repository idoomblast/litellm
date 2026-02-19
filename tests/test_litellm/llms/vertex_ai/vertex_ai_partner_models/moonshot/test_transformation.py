"""
Tests for Vertex AI Moonshot (Kimi K2) partner model transformation.

Covers:
- Config initialization and parameter mapping
- Native Kimi K2 tool call parsing in non-streaming responses
- finish_reason fix ("stop" → "tool_calls")
- <think> tag extraction to reasoning_content
- Streaming handler chunk cleaning
- Duplicate tool call prevention
"""

import json
import pytest
from unittest.mock import MagicMock

from litellm.llms.vertex_ai.vertex_ai_partner_models.moonshot.transformation import (
    VertexAIMoonshotConfig,
    VertexAIMoonshotStreamingHandler,
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
    """Tests for VertexAIMoonshotStreamingHandler."""

    def setup_method(self):
        self.handler = VertexAIMoonshotStreamingHandler(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )

    def test_should_detect_standard_tool_calls(self):
        """should track standard OpenAI tool_calls and set _saw_standard_tool_calls."""
        chunk = {
            "choices": [
                {
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
            ]
        }

        result = self.handler._clean_chunk_dict(chunk)

        assert self.handler._saw_standard_tool_calls is True
        assert self.handler._emitted_any_tool_calls is True

    def test_should_parse_native_tool_calls_in_streaming(self):
        """should parse native Kimi K2 tool call tokens from streaming content."""
        chunk = {
            "choices": [
                {
                    "delta": {
                        "content": '<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{"q": "test"}<|tool_call_end|>'
                    },
                    "finish_reason": None,
                }
            ]
        }

        result = self.handler._clean_chunk_dict(chunk)

        assert self.handler._emitted_any_tool_calls is True
        assert "tool_calls" in result["choices"][0]["delta"]
        tc = result["choices"][0]["delta"]["tool_calls"][0]
        assert tc["function"]["name"] == "search"

    def test_should_skip_native_tokens_when_standard_format_seen(self):
        """should skip native tool tokens when standard format was already seen."""
        # First chunk: standard tool_calls
        chunk1 = {
            "choices": [
                {
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
            ]
        }
        self.handler._clean_chunk_dict(chunk1)
        assert self.handler._saw_standard_tool_calls is True

        # Second chunk: native tokens in content (should be stripped, not parsed)
        chunk2 = {
            "choices": [
                {
                    "delta": {
                        "content": '<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{"q": "test"}<|tool_call_end|>'
                    },
                    "finish_reason": None,
                }
            ]
        }

        result = self.handler._clean_chunk_dict(chunk2)

        # Content should be stripped, no tool_calls added
        content = result["choices"][0]["delta"].get("content", "")
        assert "<|tool_call" not in content

    def test_should_fix_finish_reason_in_streaming(self):
        """should fix finish_reason to 'tool_calls' when tool calls were emitted."""
        # First: emit tool calls
        chunk1 = {
            "choices": [
                {
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
            ]
        }
        self.handler._clean_chunk_dict(chunk1)

        # Second: finish with "stop" — should be fixed to "tool_calls"
        chunk2 = {
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop",
                }
            ]
        }

        result = self.handler._clean_chunk_dict(chunk2)

        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_should_not_fix_finish_reason_without_tool_calls(self):
        """should NOT fix finish_reason when no tool calls were emitted."""
        chunk = {
            "choices": [
                {
                    "delta": {"content": "Hello!"},
                    "finish_reason": "stop",
                }
            ]
        }

        result = self.handler._clean_chunk_dict(chunk)

        assert result["choices"][0]["finish_reason"] == "stop"

    def test_should_clean_tool_call_xml_tags_from_function_names(self):
        """should clean <tool_call> XML tags from function names in standard format."""
        chunk = {
            "choices": [
                {
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
            ]
        }

        result = self.handler._clean_chunk_dict(chunk)

        tc = result["choices"][0]["delta"]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"

    def test_should_handle_think_tags_in_streaming(self):
        """should extract <think> content to reasoning_content in streaming."""
        chunk = {
            "choices": [
                {
                    "delta": {
                        "content": "<think>Step by step analysis</think>The answer is 42."
                    },
                    "finish_reason": None,
                }
            ]
        }

        result = self.handler._clean_chunk_dict(chunk)

        delta = result["choices"][0]["delta"]
        assert delta.get("reasoning_content") == "Step by step analysis"
        assert delta.get("content") == "The answer is 42."

    def test_should_handle_empty_choices(self):
        """should handle chunks with empty choices gracefully."""
        chunk = {"choices": []}

        result = self.handler._clean_chunk_dict(chunk)

        assert result == {"choices": []}

    def test_should_handle_non_dict_input(self):
        """should return input as-is for non-dict chunks."""
        chunk = "not a dict"
        result = self.handler._clean_chunk_dict(chunk)
        assert result == "not a dict"


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
