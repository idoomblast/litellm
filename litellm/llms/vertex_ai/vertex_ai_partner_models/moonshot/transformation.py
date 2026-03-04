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
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

from litellm.llms.chutes.chat.kimi_k2_tool_call_parser import (
    MAX_SPECIAL_TOKEN_LENGTH,
    THINK_END_TAG,
    THINK_START_TAG,
    TOOL_CALL_ARGUMENT_BEGIN,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    TOOL_CALL_FIELDS,
    TOOL_CALLS_SECTION_BEGIN,
    TOOL_CALLS_SECTION_END,
    deduplicate_reasoning_parts,
    extract_think_content_complete,
    extract_tool_call_id_parts,
    has_tool_call_tokens,
    is_kimi_k2_model,
    parse_tool_calls_from_content,
    parse_tool_calls_from_message,
    strip_native_tool_tokens,
    strip_think_tags,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    ChatCompletionMessageToolCall,
    Delta,
    Function,
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

            # First, extract <think> content from message.content
            think_reasoning = None
            if message.content and "<think>" in message.content:
                think_reasoning, cleaned = extract_think_content_complete(message.content)
                # Update content with cleaned version (think tags removed)
                message.content = cleaned

            # Normalize all reasoning fields to reasoning_content (with dedup)
            # API may send same reasoning in reasoning, reasoning_content, and thinking
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
                # Clear the original field after collecting
                try:
                    setattr(message, norm_field, None)
                except Exception:
                    pass
            if reasoning_parts:
                message.reasoning_content = deduplicate_reasoning_parts(reasoning_parts)

            # If tool_calls already exist (from standard format), just clean content
            if message.tool_calls:
                # Strip native tool tokens from content fields
                for field in TOOL_CALL_FIELDS:
                    field_value = getattr(message, field, None)
                    if field_value and isinstance(field_value, str) and has_tool_call_tokens(field_value):
                        cleaned = strip_native_tool_tokens(field_value)
                        setattr(message, field, cleaned if cleaned else None)

                # Strip whitespace from function names; fix null names
                for tc in message.tool_calls:
                    if tc.function and tc.function.name:
                        tc.function.name = tc.function.name.strip()
                    elif tc.function and tc.function.name is None:
                        tc.function.name = ""

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
        """Transform response and parse native tool call tokens if present."""
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

    Ported from ChutesChatCompletionStreamingHandler with full feature parity:

    1. Cross-chunk buffering: Buffers content to detect special tokens that may
       span across chunk boundaries (keeps last MAX_SPECIAL_TOKEN_LENGTH chars)
    2. Stateful think tag processing: State machine tracks <think>/</think> blocks
       across multiple chunks for proper reasoning_content routing
    3. Incremental reasoning emission: Emits reasoning from dedicated fields
       (reasoning_content, reasoning, thinking) incrementally with deduplication
    4. End-of-stream flush: Properly flushes all buffers when finish_reason is set,
       handling unclosed think tags and remaining content
    5. Per-field tool section tracking: Tracks tool call sections independently
       for each field (content, reasoning, reasoning_content, thinking)
    6. Duplicate tool call prevention: Standard OpenAI format takes priority
       over native Kimi tokens

    Streaming State Machine:
    NORMAL -> detecting <|tool_calls_section_begin|> or <|tool_call_begin|>
      |
      v (found section begin)
    IN_TOOL_SECTION -> buffering until <|tool_calls_section_end|>
      |
      v (found section end)
    PARSE_TOOL_CALLS -> extract individual tool calls
      |
      v
    EMIT_DELTAS -> yield tool call delta chunks
      |
      v
    NORMAL (resume)

    For Vertex AI format (may or may not have section wrappers), both formats
    are supported.
    """

    # Pattern to clean <tool_call> XML tags (GLM-style, if present)
    TOOL_CALL_XML_PATTERN = re.compile(r"</?tool_call>")

    def __init__(self, streaming_response: Any, sync_stream: bool, json_mode: Optional[bool] = False):
        super().__init__(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

        # Separate buffer for each field that might contain tool calls
        self._field_buffers: Dict[str, str] = {field: "" for field in TOOL_CALL_FIELDS}

        # Track if we're inside a tool section for each field
        self._in_tool_section: Dict[str, bool] = {
            field: False for field in TOOL_CALL_FIELDS
        }

        # Collected tool calls from streaming
        self._collected_tool_calls: List[ChatCompletionDeltaToolCall] = []
        self._tool_call_index: int = 0

        # Track chunk metadata
        self._chunk_id: Optional[str] = None
        self._chunk_model: Optional[str] = None
        self._chunk_created: Optional[int] = None

        # CRITICAL: Persistent flag to prevent duplicate tool call emission
        self._saw_standard_tool_calls: bool = False

        # Track if any tool calls were emitted (for finish_reason fix)
        self._emitted_any_tool_calls: bool = False

        # State machine for <think> tag processing (persists across chunks)
        self._in_think_block: bool = False

        # Track if we've emitted any reasoning incrementally (affects flush stripping)
        self._emitted_any_reasoning: bool = False

    # ── Cross-chunk buffering helpers ──

    def _check_for_section_begin(self, field: str) -> bool:
        """Check if buffer contains tool section begin token."""
        return TOOL_CALLS_SECTION_BEGIN in self._field_buffers[field]

    def _check_for_section_end(self, field: str) -> bool:
        """Check if buffer contains tool section end token."""
        return TOOL_CALLS_SECTION_END in self._field_buffers[field]

    def _extract_tool_calls_from_buffer(
        self, field: str
    ) -> List[ChatCompletionDeltaToolCall]:
        """
        Extract complete tool calls from the buffer for a specific field.
        Handles official format with section wrappers.

        Returns:
            List of ChatCompletionDeltaToolCall objects
        """
        tool_calls: List[ChatCompletionDeltaToolCall] = []
        buffer = self._field_buffers[field]

        begin_idx = buffer.find(TOOL_CALLS_SECTION_BEGIN)
        end_idx = buffer.find(TOOL_CALLS_SECTION_END)

        if begin_idx == -1 or end_idx == -1:
            return tool_calls

        section_start = begin_idx + len(TOOL_CALLS_SECTION_BEGIN)
        section_content = buffer[section_start:end_idx]

        current_pos = 0
        while True:
            tc_begin_idx = section_content.find(TOOL_CALL_BEGIN, current_pos)
            if tc_begin_idx == -1:
                break

            tc_end_idx = section_content.find(TOOL_CALL_END, tc_begin_idx)
            if tc_end_idx == -1:
                break

            tc_start = tc_begin_idx + len(TOOL_CALL_BEGIN)
            tc_content = section_content[tc_start:tc_end_idx].strip()

            arg_begin_idx = tc_content.find(TOOL_CALL_ARGUMENT_BEGIN)
            if arg_begin_idx != -1:
                tool_call_id = tc_content[:arg_begin_idx].strip()
                arguments = tc_content[arg_begin_idx + len(TOOL_CALL_ARGUMENT_BEGIN):].strip()

                try:
                    func_name, _ = extract_tool_call_id_parts(tool_call_id)

                    tool_call = ChatCompletionDeltaToolCall(
                        id=tool_call_id,
                        type="function",
                        function=Function(
                            name=func_name,
                            arguments=arguments,
                        ),
                        index=self._tool_call_index,
                    )
                    tool_calls.append(tool_call)
                    self._tool_call_index += 1
                except ValueError:
                    pass

            current_pos = tc_end_idx + len(TOOL_CALL_END)

        # Remove the processed tool section from buffer
        before_section = buffer[:begin_idx]
        after_section = buffer[end_idx + len(TOOL_CALLS_SECTION_END):]
        self._field_buffers[field] = before_section + after_section

        return tool_calls

    def _extract_tool_calls_no_section_wrapper(
        self, field: str
    ) -> List[ChatCompletionDeltaToolCall]:
        """
        Extract complete tool calls from buffer without section wrappers.

        Format:
        <|tool_call_begin|> functions.name:idx <|tool_call_argument_begin|> {...} <|tool_call_end|>

        Returns:
            List of ChatCompletionDeltaToolCall objects
        """
        buffer = self._field_buffers[field]

        tool_calls_parsed, cleaned_content = parse_tool_calls_from_content(buffer)

        if tool_calls_parsed:
            self._field_buffers[field] = cleaned_content

            delta_tool_calls = []
            for tc in tool_calls_parsed:
                delta_tc = ChatCompletionDeltaToolCall(
                    id=tc.id,
                    type="function",
                    function=Function(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                    index=self._tool_call_index,
                )
                delta_tool_calls.append(delta_tc)
                self._tool_call_index += 1

            return delta_tool_calls

        return []

    def _get_safe_content_to_emit(self, field: str) -> str:
        """
        Get content that's safe to emit (not potentially part of a special token).

        Keeps last MAX_SPECIAL_TOKEN_LENGTH chars in buffer to detect tokens
        that span chunk boundaries.

        Returns:
            Content safe to emit
        """
        buffer = self._field_buffers[field]

        # If in tool section, don't emit anything until section is complete
        if self._in_tool_section[field]:
            return ""

        # Keep last MAX_SPECIAL_TOKEN_LENGTH chars in buffer to detect split tokens
        if len(buffer) > MAX_SPECIAL_TOKEN_LENGTH:
            safe_content = buffer[:-MAX_SPECIAL_TOKEN_LENGTH]
            self._field_buffers[field] = buffer[-MAX_SPECIAL_TOKEN_LENGTH:]
            return safe_content

        return ""

    def _flush_remaining_content(self, field: str) -> str:
        """
        Flush any remaining content from buffer at end of stream.

        Returns:
            Remaining content
        """
        content = self._field_buffers[field]
        self._field_buffers[field] = ""
        return content

    # ── Think tag state machine ──

    def _process_content_for_think_tags(
        self, content: str
    ) -> tuple:
        """
        Process content and route to appropriate output based on think tag state.

        Uses a persistent state machine (_in_think_block) to track whether we're
        inside a <think> block across multiple chunks.

        Handles unclosed <think> tags (e.g., model calls tools mid-thinking).

        Args:
            content: The raw content string to process

        Returns:
            Tuple of (content_to_emit, reasoning_content_to_emit)
            Either may be None if no content for that category.
        """
        if not content:
            return None, None

        content_out = ""
        reasoning_out = ""

        remaining = content
        while remaining:
            if not self._in_think_block:
                if THINK_START_TAG in remaining:
                    idx = remaining.index(THINK_START_TAG)
                    content_out += remaining[:idx]
                    remaining = remaining[idx + len(THINK_START_TAG):]
                    self._in_think_block = True
                else:
                    content_out += remaining
                    remaining = ""
            else:
                if THINK_END_TAG in remaining:
                    idx = remaining.index(THINK_END_TAG)
                    reasoning_out += remaining[:idx]
                    remaining = remaining[idx + len(THINK_END_TAG):]
                    self._in_think_block = False
                else:
                    reasoning_out += remaining
                    remaining = ""

        return (
            content_out if content_out else None,
            reasoning_out if reasoning_out else None,
        )

    # ── Per-field tool call processing ──

    def _process_field_for_tool_calls(
        self, field: str, delta: dict
    ) -> List[ChatCompletionDeltaToolCall]:
        """
        Process a single field for tool call tokens with cross-chunk buffering.

        Returns:
            List of extracted tool calls (may be empty)
        """
        tool_calls: List[ChatCompletionDeltaToolCall] = []
        field_content = delta.get(field)

        if not field_content:
            return tool_calls

        # Add to buffer
        self._field_buffers[field] += field_content

        # CRITICAL: Only parse native tokens if NO standard format was seen
        if self._saw_standard_tool_calls:
            return tool_calls

        # Check if entering tool section (official format with section wrappers)
        if not self._in_tool_section[field] and self._check_for_section_begin(field):
            self._in_tool_section[field] = True

        # Check if tool section is complete (official format)
        if self._in_tool_section[field] and self._check_for_section_end(field):
            tool_calls = self._extract_tool_calls_from_buffer(field)
            self._in_tool_section[field] = False
        # Also check for format without section wrappers
        elif not self._in_tool_section[field]:
            buffer = self._field_buffers[field]
            if has_tool_call_tokens(buffer) and TOOL_CALL_END in buffer:
                tool_calls = self._extract_tool_calls_no_section_wrapper(field)

        return tool_calls

    # ── Incremental reasoning emission ──

    def _deduplicate_chunk_reasoning(self, parts: List[str]) -> Optional[str]:
        """
        Deduplicate reasoning parts from a single chunk without stripping whitespace.

        Preserves leading/trailing whitespace on each part, which is critical
        for incremental streaming where tokens like " is", " reasoning" need
        their leading spaces preserved for text continuity.

        Args:
            parts: List of reasoning content strings from the same chunk

        Returns:
            Deduplicated content, or None if no content
        """
        if not parts:
            return None
        seen = set()
        unique = []
        for part in parts:
            if part and part not in seen:
                seen.add(part)
                unique.append(part)
        if not unique:
            return None
        return "".join(unique)

    def _get_incremental_reasoning(self) -> Optional[str]:
        """
        Extract reasoning from dedicated reasoning fields for incremental emission.

        Reasoning fields (reasoning_content, reasoning, thinking) don't need
        buffering for think tags or tool call tokens (those are content-field concerns).
        So we can emit their content immediately with per-chunk deduplication.

        Returns:
            Deduplicated reasoning content to emit, or None
        """
        reasoning_parts: List[str] = []
        for field in ("reasoning_content", "reasoning", "thinking"):
            buf = self._field_buffers[field]
            if buf and not self._in_tool_section[field]:
                reasoning_parts.append(buf)
                self._field_buffers[field] = ""

        if not reasoning_parts:
            return None

        return self._deduplicate_chunk_reasoning(reasoning_parts)

    # ── End-of-stream flush ──

    def _flush_buffers_at_end_of_stream(
        self,
    ) -> tuple:
        """
        Flush remaining buffers at end of stream, processing think tags properly.

        Handles:
        - Unclosed <think> tags: all buffered content becomes reasoning_content
        - Any remaining reasoning field buffers
        - Deduplication of reasoning content

        Returns:
            Tuple of (content, reasoning_content) to append to output
        """
        content_out = ""
        reasoning_parts: List[str] = []

        # Flush content buffer through think tag state machine
        remaining_content = self._flush_remaining_content("content")
        if remaining_content:
            content_part, reasoning_part = self._process_content_for_think_tags(remaining_content)
            if content_part:
                content_out += content_part
            if reasoning_part:
                reasoning_parts.append(reasoning_part)

        # Flush any remaining reasoning field buffers
        for field in ("reasoning_content", "reasoning", "thinking"):
            remaining = self._flush_remaining_content(field)
            if remaining:
                reasoning_parts.append(remaining)

        # Deduplicate remaining reasoning parts
        if self._emitted_any_reasoning:
            reasoning_out = self._deduplicate_chunk_reasoning(reasoning_parts) or ""
        else:
            reasoning_out = deduplicate_reasoning_parts(reasoning_parts) or ""

        # Handle unclosed think tag case:
        # If we're still in a think block, all content is reasoning
        if self._in_think_block:
            reasoning_out += content_out
            content_out = ""

        # Strip native tool tokens if standard format was seen
        if self._saw_standard_tool_calls:
            if content_out:
                content_out = strip_native_tool_tokens(content_out)
            if reasoning_out:
                reasoning_out = strip_native_tool_tokens(reasoning_out)

        # Strip any remaining think tags that might have gotten through
        if content_out:
            content_out = strip_think_tags(content_out)
        if reasoning_out and not self._emitted_any_reasoning:
            reasoning_out = strip_think_tags(reasoning_out)

        return (
            content_out.strip() if content_out and content_out.strip() else None,
            reasoning_out if reasoning_out and reasoning_out.strip() else None,
        )

    # ── Main chunk_parser (stateful, buffered) ──

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        """
        Parse chunk and detect Kimi K2 tool call tokens in ALL possible fields.

        This is the main entry point for stream processing. It:
        1. Tracks chunk metadata (id, model, created)
        2. Detects standard tool_calls (priority) and cleans function names
        3. Buffers content per-field for cross-chunk token detection
        4. Emits reasoning incrementally from dedicated fields
        5. Flushes all buffers on finish_reason
        6. Fixes finish_reason when tool calls were emitted

        Returns:
            ModelResponseStream with properly parsed content
        """
        # Store chunk metadata
        self._chunk_id = chunk.get("id", self._chunk_id)
        self._chunk_model = chunk.get("model", self._chunk_model)
        self._chunk_created = chunk.get("created", self._chunk_created)

        choices = chunk.get("choices", [])
        if not choices:
            return ModelResponseStream(
                id=self._chunk_id,
                object="chat.completion.chunk",
                created=self._chunk_created,
                model=self._chunk_model,
                choices=[],
            )

        delta = choices[0].get("delta", {})
        finish_reason = choices[0].get("finish_reason")

        # Check for standard tool_calls (OpenAI format) — takes priority
        standard_tool_calls = delta.get("tool_calls")
        if standard_tool_calls:
            self._saw_standard_tool_calls = True
            self._emitted_any_tool_calls = True
            # Strip whitespace and XML tags from function names; fix null names
            for tc in standard_tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function")
                    if isinstance(func, dict):
                        name = func.get("name")
                        if name is None:
                            func["name"] = ""
                        elif name:
                            if "<tool_call>" in name:
                                name = self.TOOL_CALL_XML_PATTERN.sub("", name)
                            func["name"] = name.strip()

        # Process each field for tool call tokens (adds to _field_buffers)
        tool_calls_to_emit: List[ChatCompletionDeltaToolCall] = []
        for field in TOOL_CALL_FIELDS:
            tool_calls_to_emit.extend(self._process_field_for_tool_calls(field, delta))

        # Content field stays buffered (needs think tag + tool call token detection).
        # Reasoning fields are emitted incrementally.
        content_to_emit: Optional[str] = None
        reasoning_content: Optional[str] = None

        # Emit reasoning fields incrementally (before finish_reason check)
        if not finish_reason:
            incremental_reasoning = self._get_incremental_reasoning()
            if incremental_reasoning:
                reasoning_content = incremental_reasoning
                self._emitted_any_reasoning = True

        # Handle end of stream — flush remaining content buffer with think tag processing
        if finish_reason:
            content_rem, reasoning_rem = self._flush_buffers_at_end_of_stream()
            content_to_emit = content_rem
            if reasoning_rem:
                reasoning_content = reasoning_rem

        # Track if we emitted any tool calls
        if tool_calls_to_emit:
            self._emitted_any_tool_calls = True

        # Determine final tool calls (priority: standard > parsed native)
        final_tool_calls = standard_tool_calls if standard_tool_calls else (
            tool_calls_to_emit if tool_calls_to_emit else None
        )

        # Fix finish_reason if needed
        final_finish_reason = finish_reason
        if finish_reason and self._emitted_any_tool_calls and finish_reason != "tool_calls":
            final_finish_reason = "tool_calls"

        # Build response
        return ModelResponseStream(
            id=self._chunk_id,
            object="chat.completion.chunk",
            created=self._chunk_created,
            model=self._chunk_model,
            choices=[
                StreamingChoices(
                    index=choices[0].get("index", 0),
                    delta=Delta(
                        content=content_to_emit,
                        role=delta.get("role"),
                        reasoning_content=reasoning_content,
                        tool_calls=final_tool_calls,
                    ),
                    finish_reason=final_finish_reason,
                )
            ],
        )
