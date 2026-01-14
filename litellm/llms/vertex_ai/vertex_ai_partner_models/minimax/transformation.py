"""
Transformation for MiniMax model on Vertex AI platform.

MiniMax uses native XML format for tool calls that needs to be transformed
into standard OpenAI format.

XML Format:
<minimax:tool_call>
<invoke name="function_name">
<parameter name="param_name">value</parameter>
</invoke>
</minimax:tool_call>

Reference: https://github.com/MiniMax-AI/MiniMax-M2.1/blob/main/docs/tool_calling_guide.md
"""

import json
import re
from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

import httpx

from litellm import verbose_logger
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Usage

from ..llama3.transformation import VertexAILlama3Config

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class VertexAIMiniMaxConfig(VertexAILlama3Config):
    """
    Config class for MiniMax model on Vertex AI.
    
    Handles transformation of native XML tool calls to OpenAI format.
    """

    @classmethod
    def is_minimax_model(cls, model: str) -> bool:
        """
        Detect if the model is a MiniMax model.
        """
        return "minimax" in model.lower()

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform the response from Vertex AI MiniMax API.
        
        Handles both standard OpenAI format and native XML format.
        """
        # Call grandparent method directly to avoid recursion
        response = super(VertexAILlama3Config, self).transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )

        # Check if response needs MiniMax XML parsing
        try:
            raw_response_json = raw_response.json()
            if self._is_native_minimax_format(raw_response_json):
                verbose_logger.debug("Detected native MiniMax format, parsing XML tool calls")
                response = self._parse_minimax_tool_calls(response, raw_response_json, request_data)
        except Exception as e:
            verbose_logger.debug(f"Error checking/parsing MiniMax format: {e}")

        return response

    def _is_native_minimax_format(self, raw_response: dict) -> bool:
        """
        Detect if the response is in native MiniMax XML format.
        
        Checks if the response contains XML tool call markers in the content.
        """
        try:
            choices = raw_response.get("choices", [])
            if not choices:
                return False
            
            # Check first choice's message content
            message = choices[0].get("message", {})
            content = message.get("content", "")
            
            # Check for MiniMax XML tool call markers
            return "<minimax:tool_call>" in content or "<invoke name=" in content
        except Exception:
            return False

    def _parse_minimax_tool_calls(
        self, 
        model_response: ModelResponse, 
        raw_response: dict,
        request_data: dict
    ) -> ModelResponse:
        """
        Parse native MiniMax XML tool calls and convert to OpenAI format.
        """
        try:
            # Get tools from request data for parameter type validation
            tools = request_data.get("tools", [])
            
            # Parse each choice
            if hasattr(model_response, "choices") and model_response.choices:
                for choice_idx, choice in enumerate(model_response.choices):
                    if hasattr(choice, "message") and choice.message:
                        content = choice.message.get("content", "")
                        
                        # Parse XML tool calls
                        tool_calls = self._extract_tool_calls_from_xml(content, tools)
                        
                        if tool_calls:
                            # Update message with parsed tool calls
                            choice.message["tool_calls"] = tool_calls
                            choice.message["content"] = None  # Clear content when tool calls present
                            
                            # Set finish reason to tool_calls when tool calls are present
                            choice.finish_reason = "tool_calls"
            
            return model_response
        except Exception as e:
            verbose_logger.error(f"Error parsing MiniMax tool calls: {e}")
            return model_response

    def _extract_tool_calls_from_xml(self, content: str, tools: List[dict]) -> List[dict]:
        """
        Extract tool calls from MiniMax XML format.
        
        XML Format:
        <minimax:tool_call>
        <invoke name="function_name">
        <parameter name="param_name">value</parameter>
        </invoke>
        </minimax:tool_call>
        """
        tool_calls = []
        
        try:
            # Quick check if tool call marker is present
            if "<minimax:tool_call>" not in content:
                return []
            
            # Match all <minimax:tool_call> blocks
            tool_call_regex = re.compile(r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL)
            invoke_regex = re.compile(r"<invoke name=(.*?)</invoke>", re.DOTALL)
            parameter_regex = re.compile(r"<parameter name=(.*?)</parameter>", re.DOTALL)
            
            # Build tool config map for parameter type validation
            tool_config_map = {}
            for tool in tools:
                tool_name = self._get_tool_name(tool)
                if tool_name:
                    tool_config_map[tool_name] = tool
            
            # Iterate through all tool_call blocks
            for tool_call_match in tool_call_regex.findall(content):
                # Iterate through all invokes in this block
                for invoke_match in invoke_regex.findall(tool_call_match):
                    # Extract function name
                    name_match = re.search(r"^([^>]+)", invoke_match)
                    if not name_match:
                        continue
                    
                    function_name = self._extract_name(name_match.group(1))
                    
                    # Get parameter configuration
                    param_config = {}
                    if function_name in tool_config_map:
                        tool = tool_config_map[function_name]
                        params = self._get_tool_parameters(tool)
                        if isinstance(params, dict) and "properties" in params:
                            param_config = params["properties"]
                    
                    # Extract parameters
                    param_dict = {}
                    for match in parameter_regex.findall(invoke_match):
                        param_match = re.search(r"^([^>]+)>(.*)", match, re.DOTALL)
                        if param_match:
                            param_name = self._extract_name(param_match.group(1))
                            param_value = param_match.group(2).strip()
                            
                            # Remove leading and trailing newlines
                            if param_value.startswith("\n"):
                                param_value = param_value[1:]
                            if param_value.endswith("\n"):
                                param_value = param_value[:-1]
                            
                            # Get parameter type and convert
                            param_type = "string"
                            if param_name in param_config:
                                if isinstance(param_config[param_name], dict) and "type" in param_config[param_name]:
                                    param_type = param_config[param_name]["type"]
                            
                            param_dict[param_name] = self._convert_param_value(param_value, param_type)
                    
                    # Create tool call in OpenAI format
                    tool_call_id = f"call_{len(tool_calls)}"
                    tool_calls.append({
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": json.dumps(param_dict)
                        }
                    })
            
            return tool_calls
        except Exception as e:
            verbose_logger.error(f"Error extracting tool calls from XML: {e}")
            return []

    def _get_tool_name(self, tool: dict) -> Optional[str]:
        """Extract tool name from tool definition."""
        if "name" in tool:
            return tool["name"]
        elif "function" in tool and "name" in tool["function"]:
            return tool["function"]["name"]
        return None

    def _get_tool_parameters(self, tool: dict) -> dict:
        """Extract parameters from tool definition."""
        if "parameters" in tool:
            return tool["parameters"]
        elif "function" in tool and "parameters" in tool["function"]:
            return tool["function"]["parameters"]
        return {}

    def _extract_name(self, name_str: str) -> str:
        """Extract name from quoted string."""
        name_str = name_str.strip()
        if name_str.startswith('"') and name_str.endswith('"'):
            return name_str[1:-1]
        elif name_str.startswith("'") and name_str.endswith("'"):
            return name_str[1:-1]
        return name_str

    def _convert_param_value(self, value: str, param_type: str) -> Any:
        """Convert parameter value based on parameter type."""
        if value.lower() == "null":
            return None
        
        param_type = param_type.lower()
        
        if param_type in ["string", "str", "text"]:
            return value
        elif param_type in ["integer", "int"]:
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        elif param_type in ["number", "float"]:
            try:
                val = float(value)
                return val if val != int(val) else int(val)
            except (ValueError, TypeError):
                return value
        elif param_type in ["boolean", "bool"]:
            return value.lower() in ["true", "1"]
        elif param_type in ["object", "array"]:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        else:
            # Try JSON parsing, return string if failed
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value