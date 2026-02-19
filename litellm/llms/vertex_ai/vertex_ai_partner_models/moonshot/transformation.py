"""
Transformation for Vertex AI Moonshot models (Kimi K2/K2.5)

Model: vertex_ai/moonshotai/kimi-k2-thinking-maas
Provider: vertex_ai-moonshot_models

Handles:
- Kimi K2 native tool call tokens (reuses parser from chutes provider)
- `finish_reason` fix: "stop" → "tool_calls" when tool calls are present
- `<think>` tag handling for reasoning content
- `thinking` and `reasoning_effort` parameters
"""

import json
import re
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
    overload,
)

from litellm.llms.chutes.chat.kimi_k2_tool_call_parser import (
    extract_think_content_complete,
    has_tool_call_tokens,
    is_kimi_k2_model,
    parse_tool_calls_from_content,
    parse_tool_calls_from_message,
    strip_native_tool_tokens,
    strip_think_tags,
    TOOL_CALL_FIELDS,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Delta,
    GenericStreamingChunk,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

from ....openai.chat.gpt_transformation import OpenAIGPTConfig
from ....base_llm.base_model_iterator import BaseModelResponseIterator


class VertexAIMoonshotConfig(OpenAIGPTConfig):
    """
    Config class for Vertex AI Moonshot (Kimi K2) models.

    Kimi K2 uses native tool call tokens that need to be parsed into
    standard OpenAI format. This config reuses the parser from the
    Chutes provider.

    Reference: https://cloud.google.com/vertex-ai/generative-ai/pricing#partner-models
    """

    def get_supported_openai_params(self, model: str) -> list:
        """
        Kimi K2 supports standard OpenAI parameters plus thinking and reasoning_effort.
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
        Map OpenAI params to Moonshot/Kimi K2 params.

        Handles `thinking` and `reasoning_effort` parameters.
        Maps to `chat_template_kwargs.enable_thinking` format.
        """
        optional_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        thinking_value = optional_params.pop("thinking", None)
        reasoning_effort = optional_params.pop("reasoning_effort", None)

        chat_template_kwargs = optional_params.get("chat_template_kwargs", {})
        enable_thinking = None

        # Process thinking parameter
        if thinking_value is not None:
            if isinstance(thinking_value, bool):
                enable_thinking = thinking_value
            elif isinstance(thinking_value, dict):
                thinking_type = thinking_value.get("type", "").lower()
                enable_thinking = thinking_type == "enabled"
            elif isinstance(thinking_value, str):
                enable_thinking = (
                    thinking_value.lower() == "enabled"
                    or thinking_value.lower() == "true"
                )
        elif reasoning_effort is not None:
            if reasoning_effort in ["low", "medium", "high"]:
                enable_thinking = True
            elif reasoning_effort in ["none", "minimal"]:
                enable_thinking = False
            elif isinstance(reasoning_effort, int) and reasoning_effort > 0:
                enable_thinking = True
            elif isinstance(reasoning_effort, bool):
                enable_thinking = reasoning_effort

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
        """Kimi K2 uses standard OpenAI message format."""
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
        Moonshot model is accessed via Vertex AI endpoint.
        The actual endpoint is constructed by the Vertex AI partner models handler.
        """
        return api_base, api_key

    def _parse_kimi_k2_tool_calls_from_response(
        self, response: ModelResponse
    ) -> ModelResponse:
        """
        Post-process non-streaming response to parse Kimi K2 native tool call tokens.

        Reuses the parser from litellm.llms.chutes.chat.kimi_k2_tool_call_parser.

        This method:
        1. Checks if the model is a Kimi K2 model
        2. Scans content, reasoning, reasoning_content, thinking fields for native tokens
        3. Extracts tool calls and cleans up content
        4. Fixes finish_reason from "stop" to "tool_calls" when tool calls are present
        5. Extracts <think> tags into reasoning_content
        """
        if not response.choices:
            return response

        for choice in response.choices:
            message = choice.message
            if not message:
                continue

            # Handle <think> tags → reasoning_content
            if message.content and "<think>" in message.content:
                thinking, remaining = extract_think_content_complete(message.content)
                if thinking:
                    message.reasoning_content = thinking
                if remaining is not None:
                    message.content = remaining
                else:
                    message.content = None

            # If tool_calls already exist (from standard format), just clean content
            if message.tool_calls:
                # Strip native tool tokens from content fields
                for field in TOOL_CALL_FIELDS:
                    field_value = getattr(message, field, None)
                    if field_value and isinstance(field_value, str) and has_tool_call_tokens(field_value):
                        cleaned = strip_native_tool_tokens(field_value)
                        setattr(message, field, cleaned if cleaned else None)

                # Strip whitespace from function names
                for tc in message.tool_calls:
                    if tc.function and tc.function.name:
                        tc.function.name = tc.function.name.strip()

                # Fix finish_reason
                if hasattr(choice, "finish_reason") and choice.finish_reason != "tool_calls":
                    choice.finish_reason = "tool_calls"
                continue

            # No existing tool_calls — parse from native tokens
            all_tool_calls = []
            for field in TOOL_CALL_FIELDS:
                field_value = getattr(message, field, None)
                if field_value and isinstance(field_value, str):
                    tool_calls, cleaned_content = parse_tool_calls_from_content(
                        field_value
                    )
                    if tool_calls:
                        all_tool_calls.extend(tool_calls)
                    setattr(
                        message, field, cleaned_content if cleaned_content else None
                    )

            # Set tool_calls on message if any were found
            if all_tool_calls:
                message.tool_calls = all_tool_calls
                # Fix finish_reason
                if hasattr(choice, "finish_reason") and choice.finish_reason != "tool_calls":
                    choice.finish_reason = "tool_calls"

        return response

    def transform_response(
        self,
        model: str,
        raw_response: Any,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: dict,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """Transform response and parse Kimi K2 tool calls if needed."""
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

        # Parse Kimi K2 native tool calls
        if is_kimi_k2_model(model):
            response = self._parse_kimi_k2_tool_calls_from_response(response)

        return response

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], Any],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        """
        Returns a custom iterator for Moonshot/Kimi K2 models that handles
        native tool call tokens and <think> tags in streaming responses.
        """
        return VertexAIMoonshotStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class VertexAIMoonshotStreamingHandler(BaseModelResponseIterator):
    """
    Custom streaming handler for Vertex AI Moonshot (Kimi K2) models.

    Handles:
    - Native Kimi K2 tool call tokens in streaming chunks
    - <think>/<\/think> tag processing for reasoning content
    - finish_reason fix when tool calls are present
    - Duplicate tool call prevention (standard format takes priority)
    """

    # Pattern to clean <tool_call> XML tags (GLM-style, if present)
    TOOL_CALL_XML_PATTERN = re.compile(r"</?tool_call>")

    def __init__(self, streaming_response, sync_stream, json_mode=False):
        super().__init__(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
        # Track if we've seen standard OpenAI tool_calls format
        self._saw_standard_tool_calls: bool = False
        # Track if any tool calls were emitted (for finish_reason fix)
        self._emitted_any_tool_calls: bool = False
        # Buffer for accumulating content that might contain native tool tokens
        self._content_buffer: str = ""
        # Track think tag state
        self._in_think_block: bool = False
        self._think_buffer: str = ""

    def _clean_chunk_dict(self, chunk_dict: dict) -> dict:
        """
        Clean chunk dictionary:
        - Parse native Kimi K2 tool call tokens from content
        - Handle <think> tags for reasoning content
        - Fix finish_reason when tool calls are present
        - Clean <tool_call> XML tags from function names
        """
        if not isinstance(chunk_dict, dict) or "choices" not in chunk_dict:
            return chunk_dict

        try:
            for choice in chunk_dict.get("choices", []):
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                # Check for standard tool_calls (OpenAI format) — takes priority
                standard_tool_calls = delta.get("tool_calls")
                if standard_tool_calls:
                    self._saw_standard_tool_calls = True
                    self._emitted_any_tool_calls = True

                    # Clean function names
                    for tc in standard_tool_calls:
                        if "function" in tc and "name" in tc["function"]:
                            name = tc["function"]["name"]
                            if "<tool_call>" in name:
                                tc["function"]["name"] = self.TOOL_CALL_XML_PATTERN.sub(
                                    "", name
                                ).strip()

                # Process content field for native tool tokens
                content = delta.get("content")
                if content and isinstance(content, str):
                    # If we already saw standard tool calls, just strip native tokens
                    if self._saw_standard_tool_calls:
                        if has_tool_call_tokens(content):
                            delta["content"] = strip_native_tool_tokens(content)
                    else:
                        # Check for native tool call tokens
                        if has_tool_call_tokens(content):
                            tool_calls, cleaned = parse_tool_calls_from_content(content)
                            if tool_calls:
                                self._emitted_any_tool_calls = True
                                # Convert to delta tool_calls format
                                delta_tool_calls = []
                                for i, tc in enumerate(tool_calls):
                                    delta_tc = {
                                        "index": i,
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        },
                                    }
                                    delta_tool_calls.append(delta_tc)
                                delta["tool_calls"] = delta_tool_calls
                            delta["content"] = cleaned if cleaned else ""

                # Handle <think> tags
                if content and isinstance(content, str) and "<think>" in content:
                    from litellm.llms.chutes.chat.kimi_k2_tool_call_parser import (
                        has_think_start_tag,
                        has_think_end_tag,
                    )

                    thinking, remaining = extract_think_content_complete(content)
                    if thinking:
                        delta["reasoning_content"] = thinking
                    if remaining is not None:
                        delta["content"] = remaining
                    else:
                        delta["content"] = ""

                # Fix finish_reason if we emitted tool calls
                if finish_reason and self._emitted_any_tool_calls and finish_reason != "tool_calls":
                    choice["finish_reason"] = "tool_calls"

        except Exception:
            pass

        return chunk_dict

    def chunk_parser(
        self, chunk: dict
    ) -> Union[GenericStreamingChunk, ModelResponseStream]:
        """Parse a cleaned chunk dict into a ModelResponseStream."""
        try:
            id_val = chunk.get("id", "")
            created = chunk.get("created", 0)
            model = chunk.get("model", "")
            object_val = chunk.get("object", "chat.completion.chunk")
            system_fingerprint = chunk.get("system_fingerprint", None)

            streaming_choices: List[StreamingChoices] = []
            choices = chunk.get("choices", [])

            for choice in choices:
                delta_dict = choice.get("delta", {})
                delta = Delta(**delta_dict) if delta_dict else Delta()

                streaming_choice = StreamingChoices(
                    index=choice.get("index", 0),
                    delta=delta,
                    finish_reason=choice.get("finish_reason", None),
                    logprobs=choice.get("logprobs", None),
                )
                streaming_choices.append(streaming_choice)

            return ModelResponseStream(
                id=id_val,
                object=object_val,
                created=created,
                model=model,
                system_fingerprint=system_fingerprint,
                choices=streaming_choices,
            )

        except Exception:
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
                tool_use=None,
            )

    def _handle_string_chunk(self, str_line: str) -> Any:
        """Override to clean chunks before parsing."""
        return super()._handle_string_chunk(str_line)

    async def __anext__(self):
        """Override async iterator to clean and parse chunks."""
        try:
            chunk = await self.async_response_iterator.__anext__()
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            str_line = chunk
            if isinstance(chunk, bytes):
                str_line = chunk.decode("utf-8")
                index = str_line.find("data:")
                if index != -1:
                    str_line = str_line[index:]

            stripped_json_chunk = self._string_to_dict_parser(str_line)
            if stripped_json_chunk:
                cleaned_chunk_dict = self._clean_chunk_dict(stripped_json_chunk)
                chunk = self.chunk_parser(cleaned_chunk_dict)
            else:
                chunk = self._handle_string_chunk(str_line)

            return chunk
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")

    def __next__(self):
        """Override sync iterator to clean and parse chunks."""
        try:
            chunk = self.response_iterator.__next__()
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            str_line = chunk
            if isinstance(chunk, bytes):
                str_line = chunk.decode("utf-8")
                index = str_line.find("data:")
                if index != -1:
                    str_line = str_line[index:]

            stripped_json_chunk = self._string_to_dict_parser(str_line)
            if stripped_json_chunk:
                cleaned_chunk_dict = self._clean_chunk_dict(stripped_json_chunk)
                chunk = self.chunk_parser(cleaned_chunk_dict)
            else:
                chunk = self._handle_string_chunk(str_line)

            return chunk
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")
