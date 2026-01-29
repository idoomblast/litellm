"""
Translates from OpenAI's `/v1/chat/completions` to Chutes's `/v1/chat/completions`

Includes support for Kimi K2 native tool call format with special tokens.
"""

from typing import (
    Any,
    AsyncIterator,
    Coroutine,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import httpx

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from .kimi_k2_tool_call_parser import (
    extract_think_content_complete,
    has_think_start_tag,
    is_kimi_k2_model,
    parse_tool_calls_from_content,
    strip_native_tool_tokens,
    TOOL_CALL_FIELDS,
)
from .streaming_handler import ChutesChatCompletionStreamingHandler


class ChutesChatConfig(OpenAIGPTConfig):
    """
    Config class for Chutes LLM provider.

    Chutes is a cloud-native AI deployment platform that provides
    OpenAI-compatible APIs. It supports chat completions, embeddings,
    streaming, function calling, and all standard OpenAI parameters.

    Base URL: https://llm.chutes.ai/v1/
    Website: https://chutes.ai
    """

    def get_supported_openai_params(self, model: str) -> list:
        """
        Chutes supports most OpenAI parameters including thinking and reasoning_effort.

        Reference: https://chutes.ai/docs
        """
        params = super().get_supported_openai_params(model)
        params.extend(["thinking", "reasoning_effort"])
        return params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to Chutes params.

        Handles `thinking` and `reasoning_effort` parameters for Chutes models.
        Maps implementing `chat_template_kwargs.enable_thinking` format.

        Reference: https://chutes.ai/docs
        """
        # Let parent handle standard params first
        optional_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        # Pop thinking and reasoning_effort from optional_params
        thinking_value = optional_params.pop("thinking", None)
        reasoning_effort = optional_params.pop("reasoning_effort", None)

        # Get or initialize chat_template_kwargs
        chat_template_kwargs = optional_params.get("chat_template_kwargs", {})
        
        # Determine enable_thinking value
        enable_thinking = None
        
        # Process thinking parameter first
        if thinking_value is not None:
            if isinstance(thinking_value, bool):
                # Direct boolean format
                enable_thinking = thinking_value
            elif isinstance(thinking_value, dict):
                # Handle dict format with "type" field
                thinking_type = thinking_value.get("type", "").lower()
                enable_thinking = thinking_type == "enabled"
        
        # Process reasoning_effort if thinking not provided
        elif reasoning_effort is not None:
            if reasoning_effort in ["low", "medium", "high"]:
                enable_thinking = True
            elif reasoning_effort in ["none", "minimal"]:
                enable_thinking = False
        
        # Update chat_template_kwargs only if we have a value
        if enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = enable_thinking
            optional_params["chat_template_kwargs"] = chat_template_kwargs

        return optional_params

    @overload
    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]:
        ...

    @overload
    def _transform_messages(
        self,
        messages: List[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> List[AllMessageValues]:
        ...

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """
        Chutes uses standard OpenAI message format, no transformation needed.
        """
        if is_async:
            return super()._transform_messages(
                messages=messages, model=model, is_async=True
            )
        else:
            return super()._transform_messages(
                messages=messages, model=model, is_async=False
            )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get Chutes API base and API key.

        Default API base: https://llm.chutes.ai/v1/
        API key from CHUTES_API_KEY environment variable
        """
        api_base = api_base or get_secret_str("CHUTES_API_BASE") or "https://llm.chutes.ai/v1"
        dynamic_api_key = api_key or get_secret_str("CHUTES_API_KEY") or ""
        return api_base, dynamic_api_key

    def _is_kimi_k2_model(self, model: str) -> bool:
        """
        Check if the model is a Kimi K2 model that uses native tool call format.

        Args:
            model: Model name/identifier

        Returns:
            True if the model is a Kimi K2 model
        """
        return is_kimi_k2_model(model)

    def _parse_kimi_k2_tool_calls_from_response(
        self, response: ModelResponse
    ) -> ModelResponse:
        """
        Parse Kimi K2 native tool call format from response content.

        This method:
        1. Extracts <think>...</think> content to reasoning_content
        2. Checks all possible fields (content, reasoning_content, etc.)
           for tool call tokens and extracts them into the proper tool_calls format

        CRITICAL: If message.tool_calls already exists (from standard OpenAI format),
        we skip native token parsing but still clean up the content fields.

        Handles unclosed <think> tags by treating all content after <think> as
        reasoning_content.

        Args:
            response: The ModelResponse to process

        Returns:
            ModelResponse with tool_calls populated if found and
            <think> content transformed to reasoning_content
        """
        if not response.choices:
            return response

        for choice in response.choices:
            # Only process Choices objects (non-streaming), not StreamingChoices
            if not hasattr(choice, "message"):
                continue
            message = choice.message  # type: ignore[union-attr]
            if message is None:
                continue

            # First, extract <think> content from message.content to reasoning_content
            if message.content and has_think_start_tag(message.content):
                thinking, cleaned = extract_think_content_complete(message.content)

                if thinking:
                    # Append to existing reasoning_content or create new
                    existing_reasoning = getattr(message, "reasoning_content", None)
                    if existing_reasoning:
                        message.reasoning_content = existing_reasoning + thinking
                    else:
                        message.reasoning_content = thinking

                # Update content with cleaned version (think tags removed)
                message.content = cleaned

            # CRITICAL: If tool_calls already exist (from standard format), just clean up content
            if message.tool_calls:
                # Strip native tokens from content fields (they're duplicates)
                for field in TOOL_CALL_FIELDS:
                    field_value = getattr(message, field, None)
                    if field_value and isinstance(field_value, str):
                        # Strip native tool tokens (they're duplicates)
                        cleaned = strip_native_tool_tokens(field_value)
                        # Update field
                        if cleaned:
                            setattr(message, field, cleaned)
                        else:
                            setattr(message, field, None)

                # Fix finish_reason if needed
                if hasattr(choice, "finish_reason") and choice.finish_reason != "tool_calls":
                    choice.finish_reason = "tool_calls"

                continue

            # No existing tool_calls - parse from native tokens
            all_tool_calls = []

            for field in TOOL_CALL_FIELDS:
                field_value = getattr(message, field, None)
                if field_value and isinstance(field_value, str):
                    tool_calls, cleaned_content = parse_tool_calls_from_content(
                        field_value
                    )
                    if tool_calls:
                        all_tool_calls.extend(tool_calls)
                    # Update the field with cleaned content
                    if cleaned_content:
                        setattr(message, field, cleaned_content)
                    else:
                        # Set to None if content is empty after removing tool calls
                        setattr(message, field, None)

            # Set tool_calls on message if any were found
            if all_tool_calls:
                message.tool_calls = all_tool_calls
                # Fix finish_reason if needed
                if hasattr(choice, "finish_reason") and choice.finish_reason != "tool_calls":
                    choice.finish_reason = "tool_calls"

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
        Transform the response from the API.

        For Kimi K2 models, this also parses native tool call tokens from content.

        Returns:
            ModelResponse: The transformed response.
        """
        # Get base response from parent
        response = super().transform_response(
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

        # Check if model is Kimi K2 and parse tool calls from content
        if self._is_kimi_k2_model(model):
            response = self._parse_kimi_k2_tool_calls_from_response(
                cast(ModelResponse, response)
            )

        return cast(ModelResponse, response)

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        """
        Return custom streaming handler with Kimi K2 tool call support.

        The ChutesChatCompletionStreamingHandler handles buffering and parsing
        of Kimi K2 native tool call tokens that may span chunk boundaries.

        Args:
            streaming_response: The streaming response iterator
            sync_stream: Whether this is a synchronous stream
            json_mode: Whether JSON mode is enabled

        Returns:
            ChutesChatCompletionStreamingHandler instance
        """
        return ChutesChatCompletionStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )