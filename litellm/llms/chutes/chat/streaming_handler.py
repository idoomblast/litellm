"""
Custom streaming handler for Chutes provider with Kimi K2 tool call support.

This handler buffers chunks to detect special tokens that may span across
chunk boundaries and properly handles tool call tokens in multiple fields:
content, reasoning, reasoning_content, thinking.

IMPORTANT: The Chutes proxy sends tool calls in BOTH formats:
1. OpenAI standard format: delta.tool_calls[] array (streaming chunks)
2. Native Kimi tokens: delta.content field with <|tool_call_begin|>... sequences

To prevent duplicate emissions, we track if standard format was seen and skip native tokens.
The standard format always takes priority when present.

The Chutes proxy stream pattern is:
1. delta.content: " <think>" (start thinking)
2. delta.content: "...thinking text..." (multiple chunks)
3. delta.content: " </think>" (end thinking)
4. delta.tool_calls: [{...}] (OpenAI standard format)
5. delta.content: "<|tool_call_begin|>..." (Native tokens - DUPLICATES! Skip these!)
"""

from typing import Any, Dict, List, Optional

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
)

from .kimi_k2_tool_call_parser import (
    MAX_SPECIAL_TOKEN_LENGTH,
    THINK_END_TAG,
    THINK_START_TAG,
    TOOL_CALL_ARGUMENT_BEGIN,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    TOOL_CALL_FIELDS,
    TOOL_CALLS_SECTION_BEGIN,
    TOOL_CALLS_SECTION_END,
    extract_tool_call_id_parts,
    has_tool_call_tokens,
    parse_tool_calls_from_content,
    strip_native_tool_tokens,
    strip_think_tags,
)


class ChutesChatCompletionStreamingHandler(BaseModelResponseIterator):
    """
    Custom streaming handler for Chutes provider with Kimi K2 tool call support.

    This handler:
    1. Buffers content to detect special tokens that may span chunk boundaries
    2. Detects tool call sections and parses individual tool calls
    3. Emits tool call deltas when complete tool calls are detected
    4. Handles tool call tokens in multiple fields (content, reasoning_content, etc.)
    5. CRITICAL: Prevents duplicate tool call emissions when Chutes sends both formats

    Duplicate Prevention Strategy:
    - Chutes proxy sends tool calls in BOTH OpenAI format (delta.tool_calls[])
      AND native tokens (delta.content with <|tool_call_begin|>...)
    - When delta.tool_calls is seen, we set _saw_any_standard_tool_calls = True
    - When processing native tokens, if the flag is set, we SKIP parsing (they're duplicates)
    - Standard format always takes priority

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

    For Chutes format (no section wrappers), we parse directly from content.
    """

    def __init__(
        self,
        streaming_response: Any,
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        super().__init__(streaming_response, sync_stream, json_mode)

        # Separate buffer for each field that might contain tool calls
        self._field_buffers: Dict[str, str] = {field: "" for field in TOOL_CALL_FIELDS}

        # Track if we're inside a tool section for each field (for official format with section wrappers)
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
        # When Chutes proxy sends BOTH OpenAI standard format (delta.tool_calls[])
        # AND native Kimi tokens (delta.content with <|tool_call_begin|>...),
        # we must only emit one format. This flag tracks if we've seen standard
        # format tool calls anywhere in the stream, and if so, we skip native tokens.
        self._saw_any_standard_tool_calls: bool = False

        # Track if we've emitted any tool calls (for finish_reason fix)
        self._emitted_any_tool_calls: bool = False

        # State machine for <think> tag processing
        # When _in_think_block is True, content goes to reasoning_content
        # When False, content goes to regular content
        self._in_think_block: bool = False

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

        Returns:
            List of ChatCompletionDeltaToolCall objects
        """
        tool_calls: List[ChatCompletionDeltaToolCall] = []
        buffer = self._field_buffers[field]

        # Find the tool section
        begin_idx = buffer.find(TOOL_CALLS_SECTION_BEGIN)
        end_idx = buffer.find(TOOL_CALLS_SECTION_END)

        if begin_idx == -1 or end_idx == -1:
            return tool_calls

        # Extract section content
        section_start = begin_idx + len(TOOL_CALLS_SECTION_BEGIN)
        section_content = buffer[section_start:end_idx]

        # Parse individual tool calls from section
        current_pos = 0
        while True:
            # Find tool call begin
            tc_begin_idx = section_content.find(TOOL_CALL_BEGIN, current_pos)
            if tc_begin_idx == -1:
                break

            # Find tool call end
            tc_end_idx = section_content.find(TOOL_CALL_END, tc_begin_idx)
            if tc_end_idx == -1:
                break

            # Extract tool call content
            tc_start = tc_begin_idx + len(TOOL_CALL_BEGIN)
            tc_content = section_content[tc_start:tc_end_idx].strip()

            # Parse tool call ID and arguments
            arg_begin_idx = tc_content.find(TOOL_CALL_ARGUMENT_BEGIN)
            if arg_begin_idx != -1:
                tool_call_id = tc_content[:arg_begin_idx].strip()
                arguments = tc_content[arg_begin_idx + len(TOOL_CALL_ARGUMENT_BEGIN) :].strip()

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
                    # Skip malformed tool call IDs
                    pass

            current_pos = tc_end_idx + len(TOOL_CALL_END)

        # Remove the processed tool section from buffer
        # Keep content before and after the section
        before_section = buffer[:begin_idx]
        after_section = buffer[end_idx + len(TOOL_CALLS_SECTION_END) :]
        self._field_buffers[field] = before_section + after_section

        return tool_calls

    def _get_safe_content_to_emit(self, field: str) -> str:
        """
        Get content that's safe to emit (not potentially part of a special token).

        When not in a tool section, we need to keep the last N characters
        (where N = max special token length) in the buffer to detect tokens
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

    def _process_content_for_think_tags(
        self, content: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Process content and route to appropriate output based on think tag state.

        Uses a state machine to track whether we're inside a <think> block:
        - When "<think>" is detected, set _in_think_block = True
        - Subsequent content goes to reasoning_content
        - When "</think>" is detected, set _in_think_block = False
        - Subsequent content goes to regular content

        Handles the critical edge case where <think> is never closed
        (e.g., model calls tools mid-thinking). In this case, all content
        after <think> is treated as reasoning_content.

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
                # Looking for <think> start
                if THINK_START_TAG in remaining:
                    idx = remaining.index(THINK_START_TAG)
                    # Add content before the think tag to regular content
                    content_out += remaining[:idx]
                    # Move past the <think> tag
                    remaining = remaining[idx + len(THINK_START_TAG):]
                    self._in_think_block = True
                else:
                    # No think tag, all content goes to regular content
                    content_out += remaining
                    remaining = ""
            else:
                # Inside think block, looking for </think> end
                if THINK_END_TAG in remaining:
                    idx = remaining.index(THINK_END_TAG)
                    # Add content before the end tag to reasoning
                    reasoning_out += remaining[:idx]
                    # Move past the </think> tag
                    remaining = remaining[idx + len(THINK_END_TAG):]
                    self._in_think_block = False
                else:
                    # No end tag yet, all remaining goes to reasoning
                    reasoning_out += remaining
                    remaining = ""

        return (
            content_out if content_out else None,
            reasoning_out if reasoning_out else None,
        )

    def _process_field_for_tool_calls(
        self, field: str, delta: dict
    ) -> List[ChatCompletionDeltaToolCall]:
        """
        Process a single field for tool call tokens.

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
        if self._saw_any_standard_tool_calls:
            return tool_calls

        # Check if entering tool section (official format with section wrappers)
        if not self._in_tool_section[field] and self._check_for_section_begin(field):
            self._in_tool_section[field] = True

        # Check if tool section is complete (official format)
        if self._in_tool_section[field] and self._check_for_section_end(field):
            tool_calls = self._extract_tool_calls_from_buffer(field)
            self._in_tool_section[field] = False
        # Also check for Chutes format (no section wrappers)
        elif not self._in_tool_section[field]:
            buffer = self._field_buffers[field]
            if has_tool_call_tokens(buffer) and TOOL_CALL_END in buffer:
                tool_calls = self._extract_tool_calls_chutes_format(field)

        return tool_calls

    def _flush_buffers_at_end_of_stream(
        self,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Flush remaining buffers at end of stream, processing think tags properly.

        Handles the critical edge case of unclosed <think> tags:
        If we're still in a think block when the stream ends (unclosed tag),
        all buffered content is treated as reasoning_content.

        Returns:
            Tuple of (content, reasoning_content) to append to output
        """
        content_out = ""
        reasoning_out = ""

        # Flush all field buffers and process through think tag state machine
        for field in TOOL_CALL_FIELDS:
            remaining = self._flush_remaining_content(field)
            if remaining and field == "content":
                # Process through think tag state machine
                content_part, reasoning_part = self._process_content_for_think_tags(remaining)
                if content_part:
                    content_out += content_part
                if reasoning_part:
                    reasoning_out += reasoning_part
            elif remaining and field == "reasoning_content":
                # Direct reasoning content bypasses think tag processing
                reasoning_out += remaining

        # Handle unclosed think tag case:
        # If we're still in a think block, all content is reasoning
        if self._in_think_block:
            reasoning_out += content_out
            content_out = ""

        # Strip native tool tokens if standard format was seen
        if self._saw_any_standard_tool_calls:
            if content_out:
                content_out = strip_native_tool_tokens(content_out)
            if reasoning_out:
                reasoning_out = strip_native_tool_tokens(reasoning_out)

        # Strip any remaining think tags that might have gotten through
        if content_out:
            content_out = strip_think_tags(content_out)
        if reasoning_out:
            reasoning_out = strip_think_tags(reasoning_out)

        return (
            content_out.strip() if content_out and content_out.strip() else None,
            reasoning_out.strip() if reasoning_out and reasoning_out.strip() else None,
        )

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        """
        Parse chunk and detect Kimi K2 tool call tokens in ALL possible fields.

        CRITICAL: Handles duplicate prevention when Chutes sends both formats.
        Standard delta.tool_calls format always takes priority over native tokens.

        Also transforms <think>...</think> content to reasoning_content using
        a state machine approach. Handles unclosed think tags by treating all
        subsequent content as reasoning_content.
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

        # Check for standard tool_calls (OpenAI format) - takes priority
        standard_tool_calls = delta.get("tool_calls")
        if standard_tool_calls:
            self._saw_any_standard_tool_calls = True
            self._emitted_any_tool_calls = True

        # Process each field for tool call tokens (this adds to _field_buffers)
        tool_calls_to_emit: List[ChatCompletionDeltaToolCall] = []
        for field in TOOL_CALL_FIELDS:
            tool_calls_to_emit.extend(self._process_field_for_tool_calls(field, delta))

        # Get safe content to emit (keeping some buffer for partial tokens)
        # Note: We DON'T emit content incrementally anymore - we process it all
        # through the think tag state machine at stream end
        content_to_emit: Optional[str] = None
        reasoning_content: Optional[str] = None

        # Handle end of stream - flush remaining buffers with think tag processing
        if finish_reason:
            content_rem, reasoning_rem = self._flush_buffers_at_end_of_stream()
            content_to_emit = content_rem
            reasoning_content = reasoning_rem

        # Track if we emitted any tool calls
        if tool_calls_to_emit:
            self._emitted_any_tool_calls = True

        # Determine final tool calls (priority: standard > parsed native)
        final_tool_calls = standard_tool_calls if standard_tool_calls else (tool_calls_to_emit if tool_calls_to_emit else None)

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

    def _extract_tool_calls_chutes_format(
        self, field: str
    ) -> List[ChatCompletionDeltaToolCall]:
        """
        Extract complete tool calls from buffer using Chutes format (no section wrappers).

        The Chutes format has tool calls directly in content without section wrappers:
        <|tool_call_begin|> functions.name:idx <|tool_call_argument_begin|> {...} <|tool_call_end|>

        Returns:
            List of ChatCompletionDeltaToolCall objects
        """
        buffer = self._field_buffers[field]

        # Use the parser to extract tool calls
        tool_calls, cleaned_content = parse_tool_calls_from_content(buffer)

        if tool_calls:
            # Update buffer with cleaned content
            self._field_buffers[field] = cleaned_content

            # Convert to ChatCompletionDeltaToolCall with index
            delta_tool_calls = []
            for tc in tool_calls:
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
