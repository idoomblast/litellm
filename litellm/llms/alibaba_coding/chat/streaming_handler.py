from typing import Any, Dict, List, Optional

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
)

from .tool_call_parser import (
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
    extract_tool_call_id_parts,
    has_tool_call_tokens,
    parse_tool_calls_from_content,
    strip_native_tool_tokens,
    strip_think_tags,
)


class AlibabaCodingStreamingHandler(BaseModelResponseIterator):
    def __init__(
        self,
        streaming_response: Any,
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        super().__init__(streaming_response, sync_stream, json_mode)

        self._field_buffers: Dict[str, str] = {field: "" for field in TOOL_CALL_FIELDS}
        self._in_tool_section: Dict[str, bool] = {
            field: False for field in TOOL_CALL_FIELDS
        }

        self._collected_tool_calls: List[ChatCompletionDeltaToolCall] = []
        self._tool_call_index: int = 0

        self._chunk_id: Optional[str] = None
        self._chunk_model: Optional[str] = None
        self._chunk_created: Optional[int] = None

        self._saw_any_standard_tool_calls: bool = False
        self._emitted_any_tool_calls: bool = False

        self._in_think_block: bool = False
        self._emitted_any_reasoning: bool = False

    def _check_for_section_begin(self, field: str) -> bool:
        return TOOL_CALLS_SECTION_BEGIN in self._field_buffers[field]

    def _check_for_section_end(self, field: str) -> bool:
        return TOOL_CALLS_SECTION_END in self._field_buffers[field]

    def _extract_tool_calls_from_buffer(
        self, field: str
    ) -> List[ChatCompletionDeltaToolCall]:
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
                    pass

            current_pos = tc_end_idx + len(TOOL_CALL_END)

        before_section = buffer[:begin_idx]
        after_section = buffer[end_idx + len(TOOL_CALLS_SECTION_END) :]
        self._field_buffers[field] = before_section + after_section

        return tool_calls

    def _get_safe_content_to_emit(self, field: str) -> str:
        buffer = self._field_buffers[field]

        if self._in_tool_section[field]:
            return ""

        if len(buffer) > MAX_SPECIAL_TOKEN_LENGTH:
            safe_content = buffer[:-MAX_SPECIAL_TOKEN_LENGTH]
            self._field_buffers[field] = buffer[-MAX_SPECIAL_TOKEN_LENGTH:]
            return safe_content

        return ""

    def _flush_remaining_content(self, field: str) -> str:
        content = self._field_buffers[field]
        self._field_buffers[field] = ""
        return content

    def _process_content_for_think_tags(
        self, content: str
    ) -> tuple[Optional[str], Optional[str]]:
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

    def _process_field_for_tool_calls(
        self, field: str, delta: dict
    ) -> List[ChatCompletionDeltaToolCall]:
        tool_calls: List[ChatCompletionDeltaToolCall] = []
        field_content = delta.get(field)

        if not field_content:
            return tool_calls

        self._field_buffers[field] += field_content

        if self._saw_any_standard_tool_calls:
            return tool_calls

        if not self._in_tool_section[field] and self._check_for_section_begin(field):
            self._in_tool_section[field] = True

        if self._in_tool_section[field] and self._check_for_section_end(field):
            tool_calls = self._extract_tool_calls_from_buffer(field)
            self._in_tool_section[field] = False
        elif not self._in_tool_section[field]:
            buffer = self._field_buffers[field]
            if has_tool_call_tokens(buffer) and TOOL_CALL_END in buffer:
                tool_calls = self._extract_tool_calls_native_format(field)

        return tool_calls

    def _deduplicate_chunk_reasoning(self, parts: List[str]) -> Optional[str]:
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
        reasoning_parts: List[str] = []
        for field in ("reasoning_content", "reasoning", "thinking"):
            buf = self._field_buffers[field]
            if buf and not self._in_tool_section[field]:
                reasoning_parts.append(buf)
                self._field_buffers[field] = ""

        if not reasoning_parts:
            return None

        return self._deduplicate_chunk_reasoning(reasoning_parts)

    def _flush_buffers_at_end_of_stream(
        self,
    ) -> tuple[Optional[str], Optional[str]]:
        content_out = ""
        reasoning_parts: List[str] = []

        remaining_content = self._flush_remaining_content("content")
        if remaining_content:
            content_part, reasoning_part = self._process_content_for_think_tags(remaining_content)
            if content_part:
                content_out += content_part
            if reasoning_part:
                reasoning_parts.append(reasoning_part)

        for field in ("reasoning_content", "reasoning", "thinking"):
            remaining = self._flush_remaining_content(field)
            if remaining:
                reasoning_parts.append(remaining)

        if self._emitted_any_reasoning:
            reasoning_out = self._deduplicate_chunk_reasoning(reasoning_parts) or ""
        else:
            reasoning_out = deduplicate_reasoning_parts(reasoning_parts) or ""

        if self._in_think_block:
            reasoning_out += content_out
            content_out = ""

        if self._saw_any_standard_tool_calls:
            if content_out:
                content_out = strip_native_tool_tokens(content_out)
            if reasoning_out:
                reasoning_out = strip_native_tool_tokens(reasoning_out)

        if content_out:
            content_out = strip_think_tags(content_out)
        if reasoning_out and not self._emitted_any_reasoning:
            reasoning_out = strip_think_tags(reasoning_out)

        return (
            content_out.strip() if content_out and content_out.strip() else None,
            reasoning_out if reasoning_out and reasoning_out.strip() else None,
        )

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
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

        standard_tool_calls = delta.get("tool_calls")
        if standard_tool_calls:
            self._saw_any_standard_tool_calls = True
            self._emitted_any_tool_calls = True
            for tc in standard_tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function")
                    if isinstance(func, dict) and func.get("name"):
                        func["name"] = func["name"].strip()

        tool_calls_to_emit: List[ChatCompletionDeltaToolCall] = []
        for field in TOOL_CALL_FIELDS:
            tool_calls_to_emit.extend(self._process_field_for_tool_calls(field, delta))

        content_to_emit: Optional[str] = None
        reasoning_content: Optional[str] = None

        if not finish_reason:
            incremental_reasoning = self._get_incremental_reasoning()
            if incremental_reasoning:
                reasoning_content = incremental_reasoning
                self._emitted_any_reasoning = True

        if finish_reason:
            content_rem, reasoning_rem = self._flush_buffers_at_end_of_stream()
            content_to_emit = content_rem
            if reasoning_rem:
                reasoning_content = reasoning_rem

        if tool_calls_to_emit:
            self._emitted_any_tool_calls = True

        final_tool_calls = (
            standard_tool_calls
            if standard_tool_calls
            else (tool_calls_to_emit if tool_calls_to_emit else None)
        )

        final_finish_reason = finish_reason
        if (
            finish_reason
            and self._emitted_any_tool_calls
            and finish_reason != "tool_calls"
        ):
            final_finish_reason = "tool_calls"

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

    def _extract_tool_calls_native_format(
        self, field: str
    ) -> List[ChatCompletionDeltaToolCall]:
        buffer = self._field_buffers[field]

        tool_calls, cleaned_content = parse_tool_calls_from_content(buffer)

        if tool_calls:
            self._field_buffers[field] = cleaned_content

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
