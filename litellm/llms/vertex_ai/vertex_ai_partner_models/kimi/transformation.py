"""
Transformation for Kimi-K2 model on Vertex AI platform.

Kimi-K2 uses native special token format for tool calls that needs to be transformed
into standard OpenAI format.

Special Token Format:
<|tool_calls_section_begin|>
<|tool_call_begin|> function_name
<|tool_call_argument_begin|> {...}
<|tool_call_end|>
<|tool_calls_section_end|>

Reference: https://huggingface.co/unsloth/Kimi-K2-Thinking-GGUF
"""

import json
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union, cast

import httpx

from litellm import verbose_logger
from litellm.llms.chutes.chat.kimi_k2_parser import KimiK2ToolParser
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, ModelResponse, ModelResponseStream

from ..llama3.transformation import VertexAILlama3Config

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class VertexAIKimiK2Config(VertexAILlama3Config):
    """
    Config class for Kimi-K2 model on Vertex AI.

    Handles transformation of native special token tool calls to OpenAI format.
    """

    @classmethod
    def is_kimi_k2_model(cls, model: str) -> bool:
        """
        Detect if the model is a Kimi-K2 model.
        """
        return "kimi-k2" in model.lower() or "kimik2" in model.lower() or "kimi_k2" in model.lower()

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
        Transform the response from Vertex AI Kimi-K2 API.

        Handles both standard OpenAI format and native special token format.
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

        # Normalize Kimi-K2 function names and check for native format
        try:
            # Normalize function names in existing tool_calls
            if hasattr(response, "choices") and response.choices:
                for choice in response.choices:
                    if hasattr(choice, "message") and choice.message:
                        message = choice.message
                        if "tool_calls" in message and message["tool_calls"]:
                            message["tool_calls"] = KimiK2ToolParser.normalize_tool_calls(
                                message["tool_calls"]
                            )

            # Check if response needs Kimi-K2 special token parsing
            raw_response_json = raw_response.json()
            if self._has_native_kimi_format(raw_response_json):
                verbose_logger.debug("Detected Kimi-K2 native format, parsing tool calls")
                response = self._parse_kimi_k2_tool_calls(response, raw_response_json, request_data)

                # Normalize function names after parsing native format as well
                if hasattr(response, "choices") and response.choices:
                    for choice in response.choices:
                        if hasattr(choice, "message") and choice.message:
                            message = choice.message
                            if "tool_calls" in message and message["tool_calls"]:
                                message["tool_calls"] = KimiK2ToolParser.normalize_tool_calls(
                                    message["tool_calls"]
                                )
        except Exception as e:
            verbose_logger.debug(f"Error checking/parsing Kimi-K2 format: {e}")

        return response

    def _has_native_kimi_format(self, raw_response: dict) -> bool:
        """
        Detect if the response contains Kimi-K2 native tool call format.

        Checks for special tool call markers in reasoning_content or content.
        """
        try:
            choices = raw_response.get("choices", [])
            if not choices:
                return False

            # Check first choice's message
            message = choices[0].get("message", {})

            # Check reasoning_content
            reasoning_content = message.get("reasoning_content", "")
            if reasoning_content and "<|tool_calls_section_begin|>" in reasoning_content:
                return True

            # Check content
            content = message.get("content", "")
            if content and "<|tool_calls_section_begin|>" in content:
                return True

            return False
        except Exception:
            return False

    def _parse_kimi_k2_tool_calls(
        self,
        model_response: ModelResponse,
        raw_response: dict,
        request_data: dict,
    ) -> ModelResponse:
        """
        Parse Kimi-K2 native special token tool calls and convert to OpenAI format.
        """
        try:
            parser = KimiK2ToolParser()

            # Parse each choice in response
            if isinstance(model_response, dict) or not hasattr(model_response, "choices"):
                return model_response

            for choice in model_response.choices:
                # Only process Choices (not StreamingChoices)
                if not isinstance(choice, Choices) or not hasattr(choice, "message"):
                    continue

                message = choice.message

                # Parse tool calls from reasoning_content first
                tool_calls = []
                reasoning_content = message.get("reasoning_content", "")
                if reasoning_content:
                    tool_calls.extend(parser.extract_tool_calls(reasoning_content))

                # Also check content field
                content = message.get("content", "")
                if content and not tool_calls:
                    tool_calls.extend(parser.extract_tool_calls(content))

                # If tool calls found, update message
                if tool_calls:
                    message["tool_calls"] = tool_calls

                    # Extract reasoning and main content properly
                    if reasoning_content:
                        parsed_reasoning, main_content = parser.extract_reasoning_content(reasoning_content)
                        message["reasoning_content"] = parsed_reasoning
                        if main_content:
                            message["content"] = main_content
                        else:
                            # Clear content if no main content, only tool calls
                            message["content"] = None
                    elif content:
                        # Tool calls were in content, clean it up
                        cleaned_content = parser.clean_tool_calls_from_content(content)
                        message["content"] = cleaned_content

                    # Set finish reason to tool_calls when tool calls are present
                    choice.finish_reason = "tool_calls"

            return model_response

        except Exception as e:
            verbose_logger.error(f"Error parsing Kimi-K2 tool calls: {e}")
            return model_response

    def get_model_response_iterator(
        self,
        streaming_response: Union[Any, List[ModelResponseStream]],
        sync_stream: bool,
        json_mode: Optional[bool] = None,
    ) -> Any:
        """
        Get model response iterator for streaming responses.

        Routes to KimiK2StreamingHandler for Kimi-K2 models to handle
        native special token format during streaming.

        The streaming handler detects the model from the first chunk's
        "model" field and appropriately handles the native special
        token format for Kimi-K2 models.

        Args:
            streaming_response: The streaming response from the API
            sync_stream: Whether this is a sync or async stream
            json_mode: Whether JSON mode is enabled

        Returns:
            Streaming response iterator
        """
        from litellm.llms.chutes.chat.kimi_k2_streaming import KimiK2StreamingHandler

        # Use KimiK2StreamingHandler for Kimi-K2 streaming
        # It will detect the model from the first chunk and use
        # special token parsing only for Kimi-K2 models
        return KimiK2StreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
