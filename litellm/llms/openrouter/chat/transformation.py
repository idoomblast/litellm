"""
Support for OpenAI's `/v1/chat/completions` endpoint.

Calls done in OpenAI/openai.py as OpenRouter is openai-compatible.

Docs: https://openrouter.ai/docs/parameters
"""

from enum import Enum
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple, Union, cast

import httpx
import litellm

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam
from litellm.types.llms.openrouter import OpenRouterErrorMessage
from litellm.types.utils import ModelResponse, ModelResponseStream

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import OpenRouterException


class CacheControlSupportedModels(str, Enum):
    """Models that support cache_control in content blocks."""
    CLAUDE = "claude"
    GEMINI = "gemini"


class OpenrouterConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        Allow reasoning parameters for models flagged as reasoning-capable.
        """
        supported_params = super().get_supported_openai_params(model=model)
        try:
            if litellm.supports_reasoning(
                model=model, custom_llm_provider="openrouter"
            ) or litellm.supports_reasoning(model=model):
                supported_params.append("reasoning_effort")
        except Exception:
            pass
        return list(dict.fromkeys(supported_params))

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped_openai_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        # OpenRouter-only parameters
        extra_body = {}
        transforms = non_default_params.pop("transforms", None)
        models = non_default_params.pop("models", None)
        route = non_default_params.pop("route", None)
        if transforms is not None:
            extra_body["transforms"] = transforms
        if models is not None:
            extra_body["models"] = models
        if route is not None:
            extra_body["route"] = route
        mapped_openai_params["extra_body"] = (
            extra_body  # openai client supports `extra_body` param
        )
        return mapped_openai_params

    def _supports_cache_control_in_content(self, model: str) -> bool:
        """
        Check if the model supports cache_control in content blocks.
        
        Returns:
            bool: True if model supports cache_control (Claude or Gemini models)
        """
        model_lower = model.lower()
        return any(
            supported_model.value in model_lower
            for supported_model in CacheControlSupportedModels
        )

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: List[AllMessageValues],
        tools: Optional[List["ChatCompletionToolParam"]] = None,
    ) -> Tuple[List[AllMessageValues], Optional[List["ChatCompletionToolParam"]]]:
        if self._supports_cache_control_in_content(model):
            return messages, tools
        else:
            return super().remove_cache_control_flag_from_messages_and_tools(
                model, messages, tools
            )

    def _move_cache_control_to_content(
        self, messages: List[AllMessageValues]
    ) -> List[AllMessageValues]:
        """
        Move cache_control from message level to content blocks.
        OpenRouter requires cache_control to be inside content blocks, not at message level.
        
        To avoid exceeding Anthropic's limit of 4 cache breakpoints, cache_control is only
        added to the LAST content block in each message.
        """
        transformed_messages: List[AllMessageValues] = []
        for message in messages:
            message_dict = dict(message)
            cache_control = message_dict.pop("cache_control", None)
            
            if cache_control is not None:
                content = message_dict.get("content")
                
                if isinstance(content, list):
                    # Content is already a list, add cache_control only to the last block
                    if len(content) > 0:
                        content_copy = []
                        for i, block in enumerate(content):
                            block_dict = dict(block)
                            # Only add cache_control to the last content block
                            if i == len(content) - 1:
                                block_dict["cache_control"] = cache_control
                            content_copy.append(block_dict)
                        message_dict["content"] = content_copy
                else:
                    # Content is a string, convert to structured format
                    message_dict["content"] = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": cache_control,
                        }
                    ]
            
            # Cast back to AllMessageValues after modification
            transformed_messages.append(cast(AllMessageValues, message_dict))
        
        return transformed_messages

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the overall request to be sent to the API.

        Returns:
            dict: The transformed request. Sent as the body of the API call.
        """
        if self._supports_cache_control_in_content(model):
            messages = self._move_cache_control_to_content(messages)
        
        extra_body = optional_params.pop("extra_body", {})
        response = super().transform_request(
            model, messages, optional_params, litellm_params, headers
        )
        response.update(extra_body)

        # ALWAYS add usage parameter to get cost data from OpenRouter
        # This ensures cost tracking works for all OpenRouter models
        if "usage" not in response:
            response["usage"] = {"include": True}

        return response

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform the response from OpenRouter API.

        Extracts cost information from response headers if available.
        Parses Kimi-K2 native tool call format if present.

        Returns:
            ModelResponse: The transformed response with cost information.
        """
        # Call parent transform_response to get the standard ModelResponse
        model_response = super().transform_response(
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

        # Extract cost from OpenRouter response body
        # OpenRouter returns cost information in the usage object when usage.include=true
        try:
            response_json = raw_response.json()
            if "usage" in response_json and response_json["usage"]:
                response_cost = response_json["usage"].get("cost")
                if response_cost is not None:
                    # Store cost in hidden params for the cost calculator to use
                    if not hasattr(model_response, "_hidden_params"):
                        model_response._hidden_params = {}
                    if "additional_headers" not in model_response._hidden_params:
                        model_response._hidden_params["additional_headers"] = {}
                    model_response._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] = float(response_cost)
        except Exception:
            # If we can't extract cost, continue without it - don't fail the response
            pass

        # Parse Kimi-K2 native tool calls if present
        # Also normalize function names for Kimi-K2 models
        if self._is_kimi_k2_model(model):
            try:
                # Import here to avoid circular dependency
                from ...chutes.chat.kimi_k2_parser import KimiK2ToolParser

                if hasattr(model_response, "choices") and model_response.choices:
                    for choice in model_response.choices:
                        if hasattr(choice, "message") and choice.message:
                            # Normalize function names in existing tool_calls
                            if "tool_calls" in choice.message and choice.message["tool_calls"]:
                                choice.message["tool_calls"] = KimiK2ToolParser.normalize_tool_calls(
                                    choice.message["tool_calls"]
                                )

                            # Check reasoning_content
                            reasoning_content = choice.message.get("reasoning_content", "")
                            if reasoning_content and "<|tool_calls_section_begin|>" in reasoning_content:
                                verbose_logger.debug("Detected Kimi-K2 native tool calls in reasoning_content, parsing...")
                                self._parse_kimi_k2_tool_calls(choice.message, reasoning_content)
                                # Normalize function names after parsing native format
                                if "tool_calls" in choice.message and choice.message["tool_calls"]:
                                    choice.message["tool_calls"] = KimiK2ToolParser.normalize_tool_calls(
                                        choice.message["tool_calls"]
                                    )
                            else:
                                # Check content
                                content = choice.message.get("content", "")
                                if content and "<|tool_calls_section_begin|>" in content:
                                    verbose_logger.debug("Detected Kimi-K2 native tool calls in content, parsing...")
                                    self._parse_kimi_k2_tool_calls(choice.message, content)
                                    # Normalize function names after parsing native format
                                    if "tool_calls" in choice.message and choice.message["tool_calls"]:
                                        choice.message["tool_calls"] = KimiK2ToolParser.normalize_tool_calls(
                                            choice.message["tool_calls"]
                                        )
            except Exception as e:
                verbose_logger.debug(f"Error parsing Kimi-K2 native format: {e}")

        return model_response

    def _is_kimi_k2_model(self, model: str) -> bool:
        """
        Check if the model is a Kimi-K2 model.

        Args:
            model: Model name

        Returns:
            True if this is a Kimi-K2 model
        """
        model_lower = model.lower()
        return "kimi-k2" in model_lower or "kimik2" in model_lower

    def _parse_kimi_k2_tool_calls(self, message: Any, content: str) -> None:
        """
        Parse Kimi-K2 native tool calls from content.

        Kimi-K2 uses special tokens for tool calls:
        <|tool_calls_section_begin|>
        <|tool_call_begin|> function_name
        <|tool_call_argument_begin|> {...}
        <|tool_call_end|>
        <|tool_calls_section_end|>

        Args:
            message: The message object to update
            content: Content string containing tool calls
        """
        try:
            # Import here to avoid circular dependency
            from ...chutes.chat.kimi_k2_parser import KimiK2ToolParser

            parser = KimiK2ToolParser()

            # Extract tool calls
            tool_calls = parser.extract_tool_calls(content)

            if tool_calls:
                message["tool_calls"] = tool_calls

                # Extract reasoning and main content
                reasoning, main_content = parser.extract_reasoning_content(content)

                if reasoning:
                    message["reasoning_content"] = reasoning

                # Update content based on where tool calls were found
                if content == message.get("reasoning_content"):
                    message["reasoning_content"] = reasoning
                elif content == message.get("content"):
                    message["content"] = main_content

        except Exception as e:
            verbose_logger.error(f"Error parsing Kimi-K2 tool calls: {e}")

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        return OpenRouterChatCompletionStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class OpenRouterChatCompletionStreamingHandler(BaseModelResponseIterator):
    """
    Streaming handler for OpenRouter chat completions.

    Handles error chunks, transforms reasoning to reasoning_content,
    and parses Kimi-K2 native special token format for tool calls.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model: Optional[str] = None
        self._should_parse_kimi_tool_calls = False
        self._buffer = ""
        self._tool_calls_sent = False

    @classmethod
    def _is_kimi_k2_model(cls, model: str) -> bool:
        """Check if the model is a Kimi-K2 model."""
        model_lower = model.lower()
        return "kimi-k2" in model_lower or "kimi_k2" in model_lower or "kimik2" in model_lower

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        try:
            ## HANDLE ERROR IN CHUNK ##
            if "error" in chunk:
                error_chunk = chunk["error"]
                error_message = OpenRouterErrorMessage(
                    message="Message: {}, Metadata: {}, User ID: {}".format(
                        error_chunk["message"],
                        error_chunk.get("metadata", {}),
                        error_chunk.get("user_id", ""),
                    ),
                    code=error_chunk["code"],
                    metadata=error_chunk.get("metadata", {}),
                )
                raise OpenRouterException(
                    message=error_message["message"],
                    status_code=error_message["code"],
                    headers=error_message["metadata"].get("headers", {}),
                )

            # Detect model from first chunk for Kimi-K2 handling
            if self._model is None:
                self._model = chunk.get("model", "")
                self._should_parse_kimi_tool_calls = self._is_kimi_k2_model(self._model)

            new_choices = []
            for choice in chunk["choices"]:
                delta = choice["delta"]
                # Handle reasoning to reasoning_content transformation
                if "reasoning" in delta and delta["reasoning"]:
                    delta["reasoning_content"] = delta["reasoning"]

                # Process Kimi-K2 native special tokens if applicable
                if self._should_parse_kimi_tool_calls:
                    result = self._process_kimi_k2_chunk(delta, choice)
                    if result is not None:
                        # Delta was modified, use the processed version
                        pass

                new_choices.append(choice)
            return ModelResponseStream(
                id=chunk["id"],
                object="chat.completion.chunk",
                created=chunk["created"],
                usage=chunk.get("usage"),
                model=chunk["model"],
                choices=new_choices,
            )
        except KeyError as e:
            raise OpenRouterException(
                message=f"KeyError: {e}, Got unexpected response from OpenRouter: {chunk}",
                status_code=400,
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            raise e

    def _process_kimi_k2_chunk(self, delta: dict, choice: dict) -> Optional[bool]:
        """
        Process a delta chunk for Kimi-K2 native special tokens.

        Returns True if the chunk was processed (content may be modified).
        """
        # Handle openai-style tool_calls first (already parsed)
        if "tool_calls" in delta and delta["tool_calls"]:
            return True

        # Process native special tokens
        from ...chutes.chat.kimi_k2_parser import KimiK2ToolParser

        text_content = delta.get("content", "")
        reasoning_content = delta.get("reasoning_content", "")

        # Check for native special tokens
        has_tool_markers = (
            "<|tool_calls_section_begin|>" in text_content or
            "<|tool_calls_section_begin|>" in reasoning_content
        )

        if not has_tool_markers:
            # No special tokens, process as normal
            return False

        # Determine which content to process
        content_to_parse = reasoning_content if reasoning_content and "<|tool_calls_section_begin|>" in reasoning_content else (
            text_content if text_content and "<|tool_calls_section_begin|>" in text_content else ""
        )

        if not content_to_parse:
            return False

        # Accumulate buffer for special token detection
        self._buffer += content_to_parse

        # Check if we have a complete tool calls section
        if "<|tool_calls_section_end|>" in self._buffer:
            # We have a complete tool calls section
            tool_calls = self._parse_complete_kimi_tool_calls()
            if tool_calls and not self._tool_calls_sent:
                # Set tool_calls on delta
                delta["tool_calls"] = tool_calls
                # Clear the content that contained the special tokens
                if "reasoning_content" in delta and "<|tool_calls_section_begin|>" in reasoning_content:
                    parsed_reasoning, main_content = KimiK2ToolParser().extract_reasoning_content(content_to_parse)
                    delta["reasoning_content"] = parsed_reasoning or ""
                    if main_content:
                        delta["content"] = main_content
                elif "<|tool_calls_section_begin|>" in text_content:
                    cleaned = KimiK2ToolParser().clean_tool_calls_from_content(content_to_parse)
                    delta["content"] = cleaned or ""
                self._tool_calls_sent = True

        return True

    def _parse_complete_kimi_tool_calls(self) -> List[dict]:
        """Parse complete tool calls from accumulated buffer."""
        from ...chutes.chat.kimi_k2_parser import KimiK2ToolParser

        parser = KimiK2ToolParser()
        tool_calls = parser.extract_tool_calls(self._buffer)

        # Update buffer to only keep content after tool calls
        end_pos = self._buffer.find("<|tool_calls_section_end|>")
        if end_pos != -1:
            remaining = self._buffer[end_pos + len("<|tool_calls_section_end|>"):].strip()
            self._buffer = remaining

        return tool_calls
