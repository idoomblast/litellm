"""
Translates from OpenAI's `/v1/chat/completions` to Xiaomi MiMo's `/v1/chat/completions`
"""

from typing import Any, Coroutine, Dict, List, Literal, Optional, Tuple, Union, overload

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class XiaomiMiMoChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        Xiaomi MiMo supports the thinking parameter.
        Reference: https://platform.xiaomimimo.com/#/docs/api/text-generation/openai-api
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
        Map OpenAI params to Xiaomi MiMo params.

        Handles `thinking` and `reasoning_effort` parameters for Xiaomi MiMo models.
        Xiaomi MiMo only supports `thinking` parameter with `{"type": "enabled"}` or `{"type": "disabled"}` format.
        
        Reference: https://platform.xiaomimimo.com/#/docs/api/text-generation/openai-api
        """
        # Let parent handle standard params first
        optional_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        # Pop thinking and reasoning_effort from optional_params
        thinking_value = optional_params.pop("thinking", None)
        reasoning_effort = optional_params.pop("reasoning_effort", None)

        # Handle thinking parameter - pass it through as-is if valid
        if thinking_value is not None:
            if isinstance(thinking_value, dict) and "type" in thinking_value:
                # Xiaomi MiMo expects thinking parameter with type: "enabled" or "disabled"
                optional_params["thinking"] = thinking_value

        # Handle reasoning_effort - map to thinking parameter
        elif reasoning_effort is not None:
            # Map reasoning_effort to thinking parameter
            # Xiaomi MiMo only supports {"type": "enabled"} or {"type": "disabled"}
            if reasoning_effort in ["none", "minimal"]:
                optional_params["thinking"] = {"type": "disabled"}
            elif reasoning_effort in ["low", "medium", "high"]:
                optional_params["thinking"] = {"type": "enabled"}

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
        Xiaomi MiMo uses standard OpenAI message format, no transformation needed.
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
        api_base = api_base or get_secret_str("XIAOMI_MIMO_API_BASE") or "https://api.xiaomimimo.com/v1"
        dynamic_api_key = api_key or get_secret_str("XIAOMI_MIMO_API_KEY") or ""
        return api_base, dynamic_api_key