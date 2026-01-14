"""
Unit tests for Vertex AI MiniMax transformation.

Tests the configuration and parameter handling for Vertex AI MiniMax models.
"""

import json
from typing import cast
from unittest.mock import Mock, patch

import httpx
import pytest

from litellm.llms.vertex_ai.vertex_ai_partner_models.minimax.transformation import (
    VertexAIMiniMaxConfig,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage


class TestVertexAIMiniMaxConfig:
    """Test Vertex AI MiniMax configuration."""

    def setup_method(self):
        self.config = VertexAIMiniMaxConfig()

    def test_is_minimax_model_detection(self):
        """Test MiniMax model detection."""
        # Should detect MiniMax models
        assert self.config.is_minimax_model("vertex_ai/minimaxai/minimax-m2-maas") is True
        assert self.config.is_minimax_model("minimaxai/minimax-m2-maas") is True
        assert self.config.is_minimax_model("minimax-m2-maas") is True

        # Should not detect non-MiniMax models
        assert self.config.is_minimax_model("vertex_ai/meta/llama-3") is False
        assert self.config.is_minimax_model("gpt-4") is False

    def test_extract_name_from_quoted_string(self):
        """Test name extraction from quoted strings."""
        assert self.config._extract_name('"function_name"') == "function_name"
        assert self.config._extract_name("'function_name'") == "function_name"
        assert self.config._extract_name("function_name") == "function_name"

    def test_convert_param_value_string(self):
        """Test parameter value conversion for string type."""
        assert self.config._convert_param_value("hello", "string") == "hello"
        assert self.config._convert_param_value("hello", "str") == "hello"

    def test_convert_param_value_integer(self):
        """Test parameter value conversion for integer type."""
        assert self.config._convert_param_value("42", "integer") == 42
        assert self.config._convert_param_value("42", "int") == 42
        assert self.config._convert_param_value("not_a_number", "int") == "not_a_number"

    def test_convert_param_value_float(self):
        """Test parameter value conversion for float type."""
        assert self.config._convert_param_value("3.14", "float") == 3.14
        assert self.config._convert_param_value("42.0", "float") == 42
        assert self.config._convert_param_value("not_a_number", "float") == "not_a_number"

    def test_convert_param_value_boolean(self):
        """Test parameter value conversion for boolean type."""
        assert self.config._convert_param_value("true", "boolean") is True
        assert self.config._convert_param_value("True", "boolean") is True
        assert self.config._convert_param_value("1", "bool") is True
        assert self.config._convert_param_value("false", "boolean") is False
        assert self.config._convert_param_value("False", "bool") is False
        assert self.config._convert_param_value("0", "bool") is False

    def test_convert_param_value_object(self):
        """Test parameter value conversion for object type."""
        json_str = '{"key": "value"}'
        result = self.config._convert_param_value(json_str, "object")
        assert result == {"key": "value"}

    def test_convert_param_value_json_object(self):
        """Test parameter value conversion for JSON object."""
        json_str = '{"name": "John", "age": 30, "active": true}'
        result = self.config._convert_param_value(json_str, "object")
        assert isinstance(result, dict)
        assert result["name"] == "John"
        assert result["age"] == 30
        assert result["active"] is True

    def test_convert_param_value_json_array(self):
        """Test parameter value conversion for JSON array."""
        json_str = '["apple", "banana", "orange"]'
        result = self.config._convert_param_value(json_str, "array")
        assert isinstance(result, list)
        assert result == ["apple", "banana", "orange"]

    def test_convert_param_value_null(self):
        """Test parameter value conversion for null value."""
        assert self.config._convert_param_value("null", "string") is None
        assert self.config._convert_param_value("NULL", "string") is None

    def test_get_tool_name_from_tool_definition(self):
        """Test tool name extraction from tool definitions."""
        # Test direct name field
        tool1 = {"name": "get_weather", "parameters": {}}
        assert self.config._get_tool_name(tool1) == "get_weather"

        # Test function.name field (OpenAI format)
        tool2 = {
            "type": "function",
            "function": {"name": "search_web", "parameters": {}}
        }
        assert self.config._get_tool_name(tool2) == "search_web"

        # Test missing name
        tool3 = {"parameters": {}}
        assert self.config._get_tool_name(tool3) is None

    def test_get_tool_parameters_from_tool_definition(self):
        """Test tool parameters extraction from tool definitions."""
        # Test direct parameters field
        tool1 = {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}
        assert self.config._get_tool_parameters(tool1) == {"type": "object", "properties": {}}

        # Test function.parameters field (OpenAI format)
        tool2 = {
            "type": "function",
            "function": {
                "name": "search_web",
                "parameters": {"type": "object", "properties": {}}
            }
        }
        assert self.config._get_tool_parameters(tool2) == {"type": "object", "properties": {}}

        # Test missing parameters
        tool3 = {"name": "get_weather"}
        assert self.config._get_tool_parameters(tool3) == {}

    def test_extract_tool_calls_from_xml_simple(self):
        """Test extraction of simple tool call from XML."""
        xml_content = """
        Let me help you query the weather.
        <minimax:tool_call>
        <invoke name="get_weather">
        <parameter name="location">San Francisco</parameter>
        <parameter name="unit">celsius</parameter>
        </invoke>
        </minimax:tool_call>
        """

        tools = [{
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string"}
                },
                "required": ["location", "unit"]
            }
        }]

        result = self.config._extract_tool_calls_from_xml(xml_content, tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"

        args = json.loads(result[0]["function"]["arguments"])
        assert args["location"] == "San Francisco"
        assert args["unit"] == "celsius"

    def test_extract_tool_calls_from_xml_multiple(self):
        """Test extraction of multiple tool calls from XML."""
        xml_content = """
        <minimax:tool_call>
        <invoke name="search_web">
        <parameter name="query_tag">["technology", "events"]</parameter>
        <parameter name="query_list">["OpenAI latest release"]</parameter>
        </invoke>
        <invoke name="search_web">
        <parameter name="query_tag">["technology", "events"]</parameter>
        <parameter name="query_list">["Gemini latest release"]</parameter>
        </invoke>
        </minimax:tool_call>
        """

        tools = [{
            "name": "search_web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_tag": {"type": "array", "items": {"type": "string"}},
                    "query_list": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["query_tag", "query_list"]
            }
        }]

        result = self.config._extract_tool_calls_from_xml(xml_content, tools)

        assert len(result) == 2
        assert result[0]["function"]["name"] == "search_web"
        assert result[1]["function"]["name"] == "search_web"

        args1 = json.loads(result[0]["function"]["arguments"])
        assert args1["query_list"] == ["OpenAI latest release"]

        args2 = json.loads(result[1]["function"]["arguments"])
        assert args2["query_list"] == ["Gemini latest release"]

    def test_extract_tool_calls_from_xml_with_json_params(self):
        """Test extraction of tool calls with JSON parameter values."""
        xml_content = """
        <minimax:tool_call>
        <invoke name="process_data">
        <parameter name="config">{"timeout": 30, "retries": 3}</parameter>
        <parameter name="items">["item1", "item2", "item3"]</parameter>
        </invoke>
        </minimax:tool_call>
        """

        tools = [{
            "name": "process_data",
            "parameters": {
                "type": "object",
                "properties": {
                    "config": {"type": "object"},
                    "items": {"type": "array"}
                }
            }
        }]

        result = self.config._extract_tool_calls_from_xml(xml_content, tools)

        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])

        # Verify JSON object was parsed correctly
        assert isinstance(args["config"], dict)
        assert args["config"]["timeout"] == 30
        assert args["config"]["retries"] == 3

        # Verify JSON array was parsed correctly
        assert isinstance(args["items"], list)
        assert args["items"] == ["item1", "item2", "item3"]

    @patch('litellm.llms.vertex_ai.vertex_ai_partner_models.minimax.transformation.verbose_logger')
    def test_transform_response_with_native_minimax_format(self, mock_logger):
        """Test transform_response with native MiniMax XML format."""
        # Mock raw response with native MiniMax format
        raw_response = Mock(spec=httpx.Response)
        raw_response.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": """Let me help you.
                    <minimax:tool_call>
                    <invoke name="get_weather">
                    <parameter name="location">San Francisco</parameter>
                    <parameter name="unit">celsius</parameter>
                    </invoke>
                    </minimax:tool_call>
                    """
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
                content="""Let me help you.
                    <minimax:tool_call>
                    <invoke name="get_weather">
                    <parameter name="location">San Francisco</parameter>
                    <parameter name="unit">celsius</parameter>
                    </invoke>
                    </minimax:tool_call>
                    """
            ),
            finish_reason="stop"
        )
        model_response.choices = [choice]
        model_response.usage = Usage(prompt_tokens=10, completion_tokens=20)

        # Mock logging object
        logging_obj = Mock()
        logging_obj.post_call = Mock()

        # Request data with tools
        request_data = {
            "tools": [{
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string"}
                    }
                }
            }]
        }

        # Transform response
        result = self.config.transform_response(
            model="vertex_ai/minimaxai/minimax-m2-maas",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key="test-key"
        )

        # Verify tool calls were parsed
        assert len(result.choices) == 1
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1

        tool_call = result.choices[0].message.tool_calls[0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"

        args = json.loads(tool_call["function"]["arguments"])
        assert args["location"] == "San Francisco"
        assert args["unit"] == "celsius"

        assert result.choices[0].finish_reason == "tool_calls"

    @patch('litellm.llms.vertex_ai.vertex_ai_partner_models.minimax.transformation.verbose_logger')
    def test_transform_response_with_standard_format(self, mock_logger):
        """Test transform_response with standard OpenAI format (no XML)."""
        # Mock raw response with standard OpenAI format
        raw_response = Mock(spec=httpx.Response)
        raw_response.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help?"
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

        # Mock logging object
        logging_obj = Mock()
        logging_obj.post_call = Mock()

        # Transform response (no tools in request)
        result = self.config.transform_response(
            model="vertex_ai/minimaxai/minimax-m2-maas",
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