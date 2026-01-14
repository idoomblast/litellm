"""
Translates from OpenAI's `/v1/chat/completions` to Chutes's `/v1/chat/completions`
"""

from typing import TYPE_CHECKING, Any, Coroutine, Dict, List, Literal, Optional, Tuple, Union, overload

import httpx

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.types.utils import ModelResponse

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any
    ModelResponse = Any


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
        Transform the response from Chutes API.
        
        Routes to MiniMaxM2Config for MiniMax M2.1 models.
        """
        # Lazy import to avoid circular dependency
        from .minimax_m2_transformation import MiniMaxM2Config
        
        # Route to MiniMax M2.1 config if it's a MiniMax model
        if MiniMaxM2Config.is_minimax_m2_model(model):
            return MiniMaxM2Config().transform_response(
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
        
        # Otherwise, use standard OpenAI transformation
        return super().transform_response(
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