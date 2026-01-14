"""
Unit tests for Chutes chat transformation.

Tests the configuration and parameter handling for Chutes models.
"""

from typing import List, cast
import json
from unittest.mock import Mock, patch

import pytest
import httpx
from litellm.llms.chutes.chat.transformation import ChutesChatConfig
from litellm.llms.chutes.chat.minimax_m2_transformation import MiniMaxM2Config
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Choices, Message, Usage


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


class TestMiniMaxM2Config:
    """Test MiniMax M2.1 specific transformations."""

    def setup_method(self):
        self.config = MiniMaxM2Config()

    def test_is_minimax_m2_model_detection(self):
        """Test MiniMax M2.1 model detection."""
        # Should detect MiniMax M2.1 models
        assert self.config.is_minimax_m2_model("chutes/MiniMaxAI/MiniMax-M2.1-TEE") is True
        assert self.config.is_minimax_m2_model("chutes/minimax/minimax-m2.1") is True
        assert self.config.is_minimax_m2_model("chutes/MiniMax-M2.1") is True
        
        # Should not detect non-MiniMax models
        assert self.config.is_minimax_m2_model("chutes/other/model") is False
        assert self.config.is_minimax_m2_model("gpt-4") is False

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
        assert self.config._convert_param_value("42.0", "float") == 42  # Should convert to int if no decimal
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

    def test_extract_tool_calls_from_xml_no_tools(self):
        """Test extraction when no tool calls in XML."""
        xml_content = """This is a normal response without tool calls."""
        tools = []
        
        result = self.config._extract_tool_calls_from_xml(xml_content, tools)
        assert result == []

    def test_extract_tool_calls_from_xml_with_json_array(self):
        """Test extraction with JSON array parameter."""
        xml_content = """
        <minimax:tool_call>
        <invoke name="process_items">
        <parameter name="items">[{"id": 1, "name": "item1"}, {"id": 2, "name": "item2"}]</parameter>
        </invoke>
        </minimax:tool_call>
        """
        
        tools = [{
            "name": "process_items",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {"type": "array"}
                }
            }
        }]
        
        result = self.config._extract_tool_calls_from_xml(xml_content, tools)
        
        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert isinstance(args["items"], list)
        assert len(args["items"]) == 2
        assert args["items"][0]["id"] == 1

    @patch('litellm.llms.chutes.chat.minimax_m2_transformation.verbose_logger')
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
            model="chutes/MiniMaxAI/MiniMax-M2.1-TEE",
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

    @patch('litellm.llms.chutes.chat.minimax_m2_transformation.verbose_logger')
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
            model="chutes/MiniMaxAI/MiniMax-M2.1-TEE",
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

    def test_convert_param_value_json_nested_object(self):
        """Test parameter value conversion for nested JSON object."""
        json_str = '{"user": {"name": "Alice", "address": {"city": "NYC"}}}'
        result = self.config._convert_param_value(json_str, "object")
        assert isinstance(result, dict)
        assert result["user"]["name"] == "Alice"
        assert result["user"]["address"]["city"] == "NYC"

    def test_convert_param_value_json_array_of_objects(self):
        """Test parameter value conversion for JSON array of objects."""
        json_str = '[{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]'
        result = self.config._convert_param_value(json_str, "array")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["name"] == "Item 2"

    def test_convert_param_value_json_with_special_characters(self):
        """Test parameter value conversion for JSON with special characters."""
        # Valid JSON with quotes and backslashes (properly escaped for Python string literal)
        json_str = '{"message": "Hello, \\"World\\"!", "path": "C:\\\\Users\\\\test"}'
        result = self.config._convert_param_value(json_str, "object")
        assert isinstance(result, dict)
        assert result["message"] == 'Hello, "World"!'
        assert result["path"] == "C:\\Users\\test"

    def test_convert_param_value_json_with_unicode(self):
        """Test parameter value conversion for JSON with unicode characters."""
        json_str = '{"emoji": "🚀", "chinese": "你好", "arabic": "مرحبا"}'
        result = self.config._convert_param_value(json_str, "object")
        assert isinstance(result, dict)
        assert result["emoji"] == "🚀"
        assert result["chinese"] == "你好"
        assert result["arabic"] == "مرحبا"

    def test_convert_param_value_json_malformed_fallback(self):
        """Test parameter value conversion for malformed JSON (should return as string)."""
        # Missing closing brace
        malformed_json = '{"name": "test"'
        result = self.config._convert_param_value(malformed_json, "object")
        assert result == malformed_json  # Should return as-is
        assert isinstance(result, str)

    def test_convert_param_value_json_numbers(self):
        """Test parameter value conversion for JSON with different number types."""
        json_str = '{"int": 42, "float": 3.14, "negative": -10, "zero": 0}'
        result = self.config._convert_param_value(json_str, "object")
        assert isinstance(result, dict)
        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["negative"] == -10
        assert result["zero"] == 0

    def test_convert_param_value_json_boolean_null(self):
        """Test parameter value conversion for JSON with boolean and null values."""
        json_str = '{"active": true, "deleted": false, "data": null}'
        result = self.config._convert_param_value(json_str, "object")
        assert isinstance(result, dict)
        assert result["active"] is True
        assert result["deleted"] is False
        assert result["data"] is None

    def test_convert_param_value_json_without_param_type(self):
        """Test parameter value conversion when param_type is unknown (should auto-detect JSON)."""
        json_str = '{"auto": "detected"}'
        result = self.config._convert_param_value(json_str, "unknown_type")
        assert isinstance(result, dict)
        assert result["auto"] == "detected"

    def test_convert_param_value_json_empty_values(self):
        """Test parameter value conversion for JSON with empty values."""
        json_str = '{"empty_string": "", "empty_array": [], "empty_object": {}}'
        result = self.config._convert_param_value(json_str, "object")
        assert isinstance(result, dict)
        assert result["empty_string"] == ""
        assert result["empty_array"] == []
        assert result["empty_object"] == {}

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