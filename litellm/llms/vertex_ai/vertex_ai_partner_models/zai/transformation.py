"""
Transformation for Vertex AI ZAI model (vertex_ai/zai-org/glm-4.7-maas)

Handles `thinking` and `reasoning_effort` parameters.
Maps implementing `chat_template_kwargs.enable_thinking` format.
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

from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    Delta,
    GenericStreamingChunk,
    ModelResponseStream,
    StreamingChoices,
)

from ....openai.chat.gpt_transformation import OpenAIGPTConfig
from ....base_llm.base_model_iterator import BaseModelResponseIterator


class ZAIChatConfig(OpenAIGPTConfig):
    """
    Config class for Vertex AI ZAI model.

    ZAI's GLM-4.7-MAAS model supports thinking and reasoning_effort parameters
    that control the model's reasoning behavior.

    Reference: https://cloud.google.com/vertex-ai/generative-ai/pricing#partner-models
    """

    def get_supported_openai_params(self, model: str) -> list:
        """
        ZAI supports standard OpenAI parameters plus thinking and reasoning_effort.
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
        Map OpenAI params to ZAI params.

        Handles `thinking` and `reasoning_effort` parameters for ZAI models.
        Maps implementing `chat_template_kwargs.enable_thinking` format.
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
            elif isinstance(thinking_value, str):
                # Handle string format
                enable_thinking = (
                    thinking_value.lower() == "enabled"
                    or thinking_value.lower() == "true"
                )

        # Process reasoning_effort if thinking not provided
        elif reasoning_effort is not None:
            if reasoning_effort in ["low", "medium", "high"]:
                enable_thinking = True
            elif reasoning_effort in ["none", "minimal"]:
                enable_thinking = False
            elif isinstance(reasoning_effort, int) and reasoning_effort > 0:
                enable_thinking = True
            elif isinstance(reasoning_effort, bool):
                enable_thinking = reasoning_effort

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
        ZAI uses standard OpenAI message format, no transformation needed.
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
        ZAI model is accessed via Vertex AI, so this returns the Vertex AI endpoint.
        The actual endpoint is constructed by the Vertex AI partner models handler.
        """
        return api_base, api_key

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], Any],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        """
        Returns a custom iterator for ZAI models that cleans up <tool_call> tags
        from streaming chunks before parsing.
        """
        return ZAIChatCompletionStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class ZAIChatCompletionStreamingHandler(BaseModelResponseIterator):
    """
    Custom streaming handler for ZAI models that removes unwanted <tool_call> XML tags
    from GLM-4.7 streaming responses.
    """

    # Pattern to remove entire content containing <tool_call> tags (aggressive cleanup for content)
    # This matches patterns like: <tool_call>file, <tool_call><tool_call>, etc.
    CONTENT_CLEANUP_PATTERN = re.compile(r"<tool_call>.*", re.DOTALL)

    # Pattern to remove just the <tool_call> tags from function names
    FUNCTION_NAME_CLEANUP_PATTERN = re.compile(r"</?tool_call>")

    def _clean_chunk_data(self, chunk_str: str) -> str:
        """
        Clean up <tool_call> XML tags from chunk data.

        Args:
            chunk_str: Raw chunk string from streaming response

        Returns:
            Cleaned chunk string with <tool_call> tags removed
        """
        if not chunk_str or "<tool_call>" not in chunk_str:
            return chunk_str

        try:
            # Try to parse as JSON first
            if "data: {" in chunk_str:
                # Extract JSON part from SSE format
                json_start = chunk_str.find("{")
                json_end = chunk_str.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = chunk_str[json_start:json_end]
                    chunk_data = json.loads(json_str)

                    # Clean up content field if present
                    if "choices" in chunk_data:
                        for choice in chunk_data["choices"]:
                            if "delta" in choice and "content" in choice["delta"]:
                                content = choice["delta"]["content"]
                                if "<tool_call>" in content:
                                    # Aggressive cleanup: remove entire content containing tool_call tags
                                    choice["delta"][
                                        "content"
                                    ] = self.CONTENT_CLEANUP_PATTERN.sub("", content)

                            # Clean up function name in tool_calls if present
                            if "delta" in choice and "tool_calls" in choice["delta"]:
                                for tool_call in choice["delta"]["tool_calls"]:
                                    if (
                                        "function" in tool_call
                                        and "name" in tool_call["function"]
                                    ):
                                        function_name = tool_call["function"]["name"]
                                        if "<tool_call>" in function_name:
                                            # Clean up just the tags, keep the function name
                                            tool_call["function"][
                                                "name"
                                            ] = self.FUNCTION_NAME_CLEANUP_PATTERN.sub(
                                                "", function_name
                                            )

                    # Reconstruct the SSE format
                    cleaned_json = json.dumps(chunk_data)
                    if json_start > 0:
                        prefix = chunk_str[:json_start]
                        return f"{prefix}{cleaned_json}"
                    else:
                        return f"data: {cleaned_json}"

        except (json.JSONDecodeError, KeyError, TypeError):
            # If JSON parsing fails, try regex cleanup on the raw string
            pass

        # Fallback: clean up the raw string
        # First, aggressively remove content with tool_call tags
        cleaned = self.CONTENT_CLEANUP_PATTERN.sub("", chunk_str)
        # Then clean up any remaining tool_call tags (for function names)
        cleaned = self.FUNCTION_NAME_CLEANUP_PATTERN.sub("", cleaned)

        return cleaned

    def _clean_chunk_dict(self, chunk_dict: dict) -> dict:
        """
        Clean chunk dictionary by removing unwanted <tool_call> XML tags.

        Args:
            chunk_dict: The parsed chunk dictionary

        Returns:
            Cleaned chunk dictionary with <tool_call> tags removed
        """
        try:
            # Check if this is a chunk with choices and delta
            if isinstance(chunk_dict, dict) and "choices" in chunk_dict:
                for choice in chunk_dict.get("choices", []):
                    delta = choice.get("delta", {})

                    # Clean content field - remove entire content if it contains <tool_call>
                    if "content" in delta and delta["content"]:
                        original_content = delta["content"]
                        if "<tool_call>" in original_content:
                            delta["content"] = ""

                    # Clean function name in tool_calls
                    if "tool_calls" in delta and delta["tool_calls"]:
                        for tool_call in delta["tool_calls"]:
                            if (
                                "function" in tool_call
                                and "name" in tool_call["function"]
                            ):
                                original_name = tool_call["function"]["name"]
                                cleaned_name = self.FUNCTION_NAME_CLEANUP_PATTERN.sub(
                                    "", original_name
                                )
                                tool_call["function"]["name"] = cleaned_name

            return chunk_dict

        except Exception:
            # If anything goes wrong, return the original chunk
            return chunk_dict

    def chunk_parser(
        self, chunk: dict
    ) -> Union[GenericStreamingChunk, ModelResponseStream]:
        """
        Parse a cleaned chunk dict into a ModelResponseStream.

        Args:
            chunk: The cleaned chunk dictionary

        Returns:
            ModelResponseStream
        """
        try:
            # Extract basic fields
            id_val = chunk.get("id", "")
            created = chunk.get("created", 0)
            model = chunk.get("model", "")
            object_val = chunk.get("object", "chat.completion.chunk")
            system_fingerprint = chunk.get("system_fingerprint", None)

            # Process choices
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

            # Create ModelResponseStream
            return ModelResponseStream(
                id=id_val,
                object=object_val,
                created=created,
                model=model,
                system_fingerprint=system_fingerprint,
                choices=streaming_choices,
            )

        except Exception:
            # Fallback to GenericStreamingChunk if parsing fails
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
                tool_use=None,
            )

    def _handle_string_chunk(self, str_line: str) -> Any:
        """
        Override to clean chunks before parsing.
        """
        # Clean the chunk data first
        cleaned_line = self._clean_chunk_data(str_line)

        # Then use the parent class logic to parse the cleaned chunk
        return super()._handle_string_chunk(cleaned_line)

    async def __anext__(self):
        """
        Override async iterator to clean chunks before parsing.
        """
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

            # Parse and clean the chunk
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
        """
        Override sync iterator to clean chunks before parsing.
        """
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

            # Parse and clean the chunk
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
