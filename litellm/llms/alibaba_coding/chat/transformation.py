from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple, Union

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from .streaming_handler import AlibabaCodingStreamingHandler
from .tool_call_parser import (
    TOOL_CALL_FIELDS,
    deduplicate_reasoning_parts,
    extract_think_content_complete,
    has_think_start_tag,
    is_native_tool_call_model,
    parse_tool_calls_from_content,
    strip_native_tool_tokens,
)

DEFAULT_USER_AGENT = "opencode/1.2.6 ai-sdk/provider-utils/3.0.21 runtime/bun/1.3.9"


class AlibabaCodingChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        params = super().get_supported_openai_params(model)
        params.extend(["thinking", "reasoning_effort"])
        return params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        headers = super().validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )

        # Set User-Agent header with priority:
        # 1. extra_headers["User-Agent"] (already in headers if set by user)
        # 2. litellm_params.user_agent (explicit param)
        # 3. optional_params.user_agent (from model config)
        # 4. DEFAULT_USER_AGENT (fallback)
        if "User-Agent" not in headers:
            custom_user_agent = (
                litellm_params.get("user_agent")
                or optional_params.pop("user_agent", None)
                or DEFAULT_USER_AGENT
            )
            headers["User-Agent"] = custom_user_agent

        return headers

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        optional_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        thinking_value = optional_params.pop("thinking", None)
        reasoning_effort = optional_params.pop("reasoning_effort", None)

        chat_template_kwargs = optional_params.get("chat_template_kwargs", {})

        enable_thinking = None

        if thinking_value is not None:
            if isinstance(thinking_value, bool):
                enable_thinking = thinking_value
            elif isinstance(thinking_value, dict):
                thinking_type = thinking_value.get("type", "").lower()
                enable_thinking = thinking_type == "enabled"
        elif reasoning_effort is not None:
            if reasoning_effort in ["low", "medium", "high"]:
                enable_thinking = True
            elif reasoning_effort in ["none", "minimal"]:
                enable_thinking = False

        if enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = enable_thinking
            optional_params["chat_template_kwargs"] = chat_template_kwargs

        return optional_params

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("ALIBABA_CODING_API_BASE")
            or "https://coding-intl.dashscope.aliyuncs.com/v1"
        )
        dynamic_api_key = api_key or get_secret_str("ALIBABA_CODING_API_KEY") or ""
        return api_base, dynamic_api_key

    def _is_native_tool_call_model(self, model: str) -> bool:
        return is_native_tool_call_model(model)

    def _parse_tool_calls_from_response(self, response: ModelResponse) -> ModelResponse:
        if not response.choices:
            return response

        for choice in response.choices:
            if not hasattr(choice, "message"):
                continue
            message = choice.message
            if message is None:
                continue

            think_reasoning = None
            if message.content and has_think_start_tag(message.content):
                think_reasoning, cleaned = extract_think_content_complete(message.content)
                message.content = cleaned

            reasoning_parts = []
            existing_rc = getattr(message, "reasoning_content", None)
            if existing_rc and isinstance(existing_rc, str) and existing_rc.strip():
                reasoning_parts.append(existing_rc.strip())
            if think_reasoning:
                reasoning_parts.append(think_reasoning)
            for norm_field in ("reasoning", "thinking"):
                norm_value = getattr(message, norm_field, None)
                if norm_value and isinstance(norm_value, str) and norm_value.strip():
                    reasoning_parts.append(norm_value.strip())
                try:
                    setattr(message, norm_field, None)
                except Exception:
                    pass
            if reasoning_parts:
                message.reasoning_content = deduplicate_reasoning_parts(reasoning_parts)

            if message.tool_calls:
                for tc in message.tool_calls:
                    if tc.function and tc.function.name:
                        tc.function.name = tc.function.name.strip()
                    elif tc.function and tc.function.name is None:
                        tc.function.name = ""

                for field in TOOL_CALL_FIELDS:
                    field_value = getattr(message, field, None)
                    if field_value and isinstance(field_value, str):
                        cleaned = strip_native_tool_tokens(field_value)
                        if cleaned:
                            setattr(message, field, cleaned)
                        else:
                            setattr(message, field, None)

                if (
                    hasattr(choice, "finish_reason")
                    and choice.finish_reason != "tool_calls"
                ):
                    choice.finish_reason = "tool_calls"

                continue

            all_tool_calls = []
            for field in TOOL_CALL_FIELDS:
                field_value = getattr(message, field, None)
                if field_value and isinstance(field_value, str):
                    tool_calls, cleaned_content = parse_tool_calls_from_content(field_value)
                    if tool_calls:
                        all_tool_calls.extend(tool_calls)
                    if cleaned_content:
                        setattr(message, field, cleaned_content)
                    else:
                        setattr(message, field, None)

            if all_tool_calls:
                message.tool_calls = all_tool_calls
                if (
                    hasattr(choice, "finish_reason")
                    and choice.finish_reason != "tool_calls"
                ):
                    choice.finish_reason = "tool_calls"

        return response

    def transform_response(
        self,
        model: str,
        raw_response: Any,
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

        # Always parse — content-driven, not model-name-driven.
        # If no native tokens / <think> tags are present, this is a no-op.
        response = self._parse_tool_calls_from_response(response)

        return response

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        return AlibabaCodingStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
