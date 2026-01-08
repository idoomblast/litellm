"""
Translates from OpenAI's `/v1/chat/completions` to Chutes's `/v1/chat/completions`
"""

from typing import Any, Coroutine, Dict, List, Literal, Optional, Tuple, Union, overload

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


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
        Chutes supports most OpenAI parameters including thinking.

        Reference: https://chutes.ai/docs
        """
        params = super().get_supported_openai_params(model)
        params.extend(["thinking"])
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

        Chutes is OpenAI-compatible, so most params can be passed through
        with minimal transformation. The thinking parameter is passed through
        as-is for reasoning models.

        Reference: https://chutes.ai/docs
        """
        # Handle thinking parameter before parent processing
        thinking_value = non_default_params.get("thinking")
        if thinking_value is not None:
            # Pass through thinking parameter as-is
            optional_params["thinking"] = thinking_value

        # Let parent handle standard params
        optional_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )
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