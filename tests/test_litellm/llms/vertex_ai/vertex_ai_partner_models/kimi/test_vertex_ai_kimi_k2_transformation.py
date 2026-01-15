"""
Unit tests for Vertex AI Kimi-K2 transformation.

Tests the configuration and parameter handling for Vertex AI Kimi-K2 models
with native special token tool call format.
"""

import json
from unittest.mock import Mock, patch, MagicMock

import httpx
import pytest

from litellm.llms.vertex_ai.vertex_ai_partner_models.kimi.transformation import (
    VertexAIKimiK2Config,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage


class TestVertexAIKimiK2Config:
    """Test Vertex AI Kimi-K2 configuration."""

    def setup_method(self):
        self.config = VertexAIKimiK2Config()

    def test_is_kimi_k2_model_detection(self):
        """Test Kimi-K2 model detection."""
        # Should detect Kimi-K2 models
        assert self.config.is_kimi_k2_model("vertex_ai/moonshotai/kimi-k2-instruct") is True
        assert self.config.is_kimi_k2_model("moonshotai/kimi-k2-instruct") is True
        assert self.config.is_kimi_k2_model("kimi-k2-instruct") is True
        assert self.config.is_kimi_k2_model("Kimi-K2-Thinking") is True
        assert self.config.is_kimi_k2_model("kimi_k2") is True

        # Should not detect non-Kimi-K2 models
        assert self.config.is_kimi_k2_model("vertex_ai/meta/llama-3") is False
        assert self.config.is_kimi_k2_model("gpt-4") is False
        assert self.config.is_kimi_k2_model("kimi-v1") is False

    def test_has_native_kimi_format_in_reasoning_content(self):
        """Test detection of native format in reasoning_content."""
        raw_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "I'll use a tool. <|tool_calls_section_begin|> <|tool_call_end|> <|tool_calls_section_end|>"
                }
            }]
        }

        assert self.config._has_native_kimi_format(raw_response) is True

    def test_has_native_kimi_format_in_content(self):
        """Test detection of native format in content field."""
        raw_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<|tool_calls_section_begin|> <|tool_call_end|> <|tool_calls_section_end|>",
                    "reasoning_content": None
                }
            }]
        }

        assert self.config._has_native_kimi_format(raw_response) is True

    def test_has_native_kimi_format_false(self):
        """Test detection returns False for standard format."""
        raw_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "This is standard text.",
                    "reasoning_content": "Some reasoning."
                }
            }]
        }

        assert self.config._has_native_kimi_format(raw_response) is False

    def test_has_native_kimi_format_no_choices(self):
        """Test detection with no choices returns False."""
        raw_response = {"choices": []}
        assert self.config._has_native_kimi_format(raw_response) is False

    @patch('litellm.llms.vertex_ai.vertex_ai_partner_models.kimi.transformation.verbose_logger')
    @patch('litellm.llms.vertex_ai.vertex_ai_partner_models.llama3.transformation.VertexAILlama3Config.transform_response')
    def test_parse_kimi_k2_tool_calls_in_reasoning(self, mock_parent_transform, mock_logger):
        """Test _parse_kimi_k2_tool_calls with tool calls in reasoning_content."""
        # Create ModelResponse with Kimi-K2 native format
        model_response = ModelResponse()
        choice = Choices(
            index=0,
            message=Message(
                role="assistant",
                content="",
                reasoning_content="""I'll help you with that.
<|tool_calls_section_begin|>
<|tool_call_begin|> get_weather
<|tool_call_argument_begin|> {"city": "Jakarta", "unit": "celsius"}
<|tool_call_end|>
<|tool_calls_section_end|>"""
            ),
            finish_reason="stop"
        )
        model_response.choices = [choice]
        model_response.usage = Usage(prompt_tokens=10, completion_tokens=20)

        # Mock parent transform_response to return the model_response unchanged
        mock_parent_transform.return_value = model_response

        # Raw response data
        raw_response_dict = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": """I'll help you with that.
<|tool_calls_section_begin|>
<|tool_call_begin|> get_weather
<|tool_call_argument_begin|> {"city": "Jakarta", "unit": "celsius"}
<|tool_call_end|>
<|tool_calls_section_end|>"""
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }

        # Mock raw response object
        raw_response = Mock(spec=httpx.Response)
        raw_response.json.return_value = raw_response_dict
        raw_response.text = json.dumps(raw_response_dict)
        raw_response.headers = {}
        raw_response.status_code = 200

        # Mock logging object
        logging_obj = Mock()

        # Parse directly using _parse_kimi_k2_tool_calls
        result = self.config._parse_kimi_k2_tool_calls(
            model_response=model_response,
            raw_response=raw_response_dict,
            request_data={}
        )

        # Verify tool calls were parsed
        assert len(result.choices) == 1
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1

        tool_call = result.choices[0].message.tool_calls[0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"
        assert tool_call["id"] == "call_0"

        args = json.loads(tool_call["function"]["arguments"])
        assert args["city"] == "Jakarta"
        assert args["unit"] == "celsius"

        # Verify finish reason was updated
        assert result.choices[0].finish_reason == "tool_calls"

    def test_parse_kimi_k2_tool_calls_in_content(self):
        """Test _parse_kimi_k2_tool_calls with tool calls in content field (not reasoning_content)."""
        # Create ModelResponse with tool calls in content
        model_response = ModelResponse()
        choice = Choices(
            index=0,
            message=Message(
                role="assistant",
                content="""<|tool_calls_section_begin|>
<|tool_call_begin|> search_web
<|tool_call_argument_begin|> {"query": "AI news"}
<|tool_call_end|>
<|tool_calls_section_end|>After searching, here's the information."""
            ),
            finish_reason="stop"
        )
        model_response.choices = [choice]

        # Raw response data
        raw_response_dict = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": """<|tool_calls_section_begin|>
<|tool_call_begin|> search_web
<|tool_call_argument_begin|> {"query": "AI news"}
<|tool_call_end|>
<|tool_calls_section_end|>After searching, here's the information."""
                },
                "finish_reason": "stop"
            }]
        }

        # Parse directly using _parse_kimi_k2_tool_calls
        result = self.config._parse_kimi_k2_tool_calls(
            model_response=model_response,
            raw_response=raw_response_dict,
            request_data={}
        )

        # Verify tool calls were parsed
        assert len(result.choices) == 1
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1

        tool_call = result.choices[0].message.tool_calls[0]
        assert tool_call["function"]["name"] == "search_web"

        # Verify content was cleaned (tool calls removed)
        assert "After searching, here's the information" in result.choices[0].message.content
        assert "<|tool_calls_section_begin|>" not in result.choices[0].message.content

    def test_parse_kimi_k2_tool_calls_multiple(self):
        """Test _parse_kimi_k2_tool_calls with multiple tool calls."""
        # Create ModelResponse with multiple tool calls
        model_response = ModelResponse()
        choice = Choices(
            index=0,
            message=Message(
                role="assistant",
                content="",
                reasoning_content="""I'll help you.
<|tool_calls_section_begin|>
<|tool_call_begin|> get_weather
<|tool_call_argument_begin|> {"city": "Jakarta"}
<|tool_call_end|>
<|tool_call_begin|> get_time
<|tool_call_argument_begin|> {"timezone": "Asia/Jakarta"}
<|tool_call_end|>
<|tool_calls_section_end|>"""
            ),
            finish_reason="stop"
        )
        model_response.choices = [choice]

        # Raw response data
        raw_response_dict = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": """I'll help you.
<|tool_calls_section_begin|>
<|tool_call_begin|> get_weather
<|tool_call_argument_begin|> {"city": "Jakarta"}
<|tool_call_end|>
<|tool_call_begin|> get_time
<|tool_call_argument_begin|> {"timezone": "Asia/Jakarta"}
<|tool_call_end|>
<|tool_calls_section_end|>"""
                },
                "finish_reason": "stop"
            }]
        }

        # Parse directly using _parse_kimi_k2_tool_calls
        result = self.config._parse_kimi_k2_tool_calls(
            model_response=model_response,
            raw_response=raw_response_dict,
            request_data={}
        )

        # Verify both tool calls were parsed
        assert len(result.choices) == 1
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 2

        # Check first tool call
        assert result.choices[0].message.tool_calls[0]["function"]["name"] == "get_weather"
        assert result.choices[0].message.tool_calls[0]["id"] == "call_0"

        # Check second tool call
        assert result.choices[0].message.tool_calls[1]["function"]["name"] == "get_time"
        assert result.choices[0].message.tool_calls[1]["id"] == "call_1"

    @patch('litellm.llms.vertex_ai.vertex_ai_partner_models.kimi.transformation.verbose_logger')
    @patch('litellm.llms.vertex_ai.vertex_ai_partner_models.llama3.transformation.VertexAILlama3Config.transform_response')
    def test_transform_response_with_standard_format(self, mock_parent_transform, mock_logger):
        """Test transform_response with standard OpenAI format (no special tokens)."""
        # Mock raw response with standard format
        raw_response = Mock(spec=httpx.Response)
        raw_response.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help?",
                    "reasoning_content": None
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }
        raw_response.text = json.dumps(raw_response.json.return_value)
        raw_response.headers = {}
        raw_response.status_code = 200

        # Create ModelResponse
        model_response = ModelResponse()
        choice = Choices(
            index=0,
            message=Message(
                role="assistant",
                content="Hello, how can I help?"
            ),
            finish_reason="stop"
        )
        model_response.choices = [choice]
        model_response.usage = Usage(prompt_tokens=10, completion_tokens=20)

        # Mock parent transform_response to return the model_response unchanged
        mock_parent_transform.return_value = model_response

        # Mock logging object
        logging_obj = Mock()

        # Transform response
        result = self.config.transform_response(
            model="vertex_ai/moonshotai/kimi-k2-instruct",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key="test-key"
        )

        # Verify response is unchanged (standard format)
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Hello, how can I help?"
        assert result.choices[0].message.tool_calls is None
        assert result.choices[0].finish_reason == "stop"

    def test_parse_kimi_k2_tool_calls_complex_json(self):
        """Test _parse_kimi_k2_tool_calls with complex (nested) JSON arguments."""
        # Create ModelResponse with complex JSON arguments
        model_response = ModelResponse()
        choice = Choices(
            index=0,
            message=Message(
                role="assistant",
                content="",
                reasoning_content="""<|tool_calls_section_begin|>
<|tool_call_begin|> search
<|tool_call_argument_begin|> {"query": "test", "filters": {"date": "2025", "lang": "en"}}
<|tool_call_end|>
<|tool_calls_section_end|>"""
            ),
            finish_reason="stop"
        )
        model_response.choices = [choice]

        # Raw response data
        raw_response_dict = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": """<|tool_calls_section_begin|>
<|tool_call_begin|> search
<|tool_call_argument_begin|> {"query": "test", "filters": {"date": "2025", "lang": "en"}}
<|tool_call_end|>
<|tool_calls_section_end|>"""
                },
                "finish_reason": "stop"
            }]
        }

        # Parse directly using _parse_kimi_k2_tool_calls
        result = self.config._parse_kimi_k2_tool_calls(
            model_response=model_response,
            raw_response=raw_response_dict,
            request_data={}
        )

        # Verify complex arguments were parsed
        assert len(result.choices) == 1
        tool_call = result.choices[0].message.tool_calls[0]
        args = json.loads(tool_call["function"]["arguments"])
        assert args["query"] == "test"
        assert args["filters"]["date"] == "2025"
        assert args["filters"]["lang"] == "en"

    @patch('litellm.llms.vertex_ai.vertex_ai_partner_models.kimi.transformation.verbose_logger')
    @patch('litellm.llms.vertex_ai.vertex_ai_partner_models.llama3.transformation.VertexAILlama3Config.transform_response')
    def test_transform_response_with_openai_tool_calls_already_present(self, mock_parent_transform, mock_logger):
        """Test that existing OpenAI-format tool calls are preserved."""
        raw_response = Mock(spec=httpx.Response)
        raw_response.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "test_func",
                            "arguments": '{"arg": "value"}'
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }
        raw_response.text = json.dumps(raw_response.json.return_value)
        raw_response.headers = {}
        raw_response.status_code = 200

        # Create ModelResponse with existing tool calls
        model_response = ModelResponse()
        choice = Choices(
            index=0,
            message=Message(
                role="assistant",
                content=None
            ),
            finish_reason="tool_calls"
        )
        choice.message["tool_calls"] = [{
            "id": "call_0",
            "type": "function",
            "function": {
                "name": "test_func",
                "arguments": '{"arg": "value"}'
            }
        }]
        model_response.choices = [choice]
        model_response.usage = Usage(prompt_tokens=10, completion_tokens=20)

        # Mock parent transform_response to return the model_response unchanged
        mock_parent_transform.return_value = model_response

        # Mock logging object
        logging_obj = Mock()

        # Transform response
        result = self.config.transform_response(
            model="kimi-k2-instruct",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key="test-key"
        )

        # Verify existing tool calls are preserved (no special tokens to parse)
        assert len(result.choices) == 1
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1
        assert result.choices[0].message.tool_calls[0]["function"]["name"] == "test_func"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
