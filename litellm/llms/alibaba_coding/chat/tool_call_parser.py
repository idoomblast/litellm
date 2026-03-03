"""
Tool call parser for Alibaba Coding provider.

Re-exports the Kimi K2 tool call parser from the chutes provider.
Alibaba Coding uses the same native tool call format for Kimi K2 models.
"""

from litellm.llms.chutes.chat.kimi_k2_tool_call_parser import (
    MAX_SPECIAL_TOKEN_LENGTH,
    THINK_END_TAG,
    THINK_START_TAG,
    THINK_TAG_PATTERN,
    TOOL_CALL_ARGUMENT_BEGIN,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    TOOL_CALL_FIELDS,
    TOOL_CALL_PATTERN,
    TOOL_CALLS_SECTION_BEGIN,
    TOOL_CALLS_SECTION_END,
    deduplicate_reasoning_parts,
    extract_think_content,
    extract_think_content_complete,
    extract_tool_call_id_parts,
    has_think_end_tag,
    has_think_start_tag,
    has_tool_call_tokens,
    is_kimi_k2_model,
    parse_tool_calls_from_content,
    parse_tool_calls_from_message,
    strip_native_tool_tokens,
    strip_think_tags,
)

# Alias for backwards compatibility
is_native_tool_call_model = is_kimi_k2_model

__all__ = [
    "MAX_SPECIAL_TOKEN_LENGTH",
    "THINK_END_TAG",
    "THINK_START_TAG",
    "THINK_TAG_PATTERN",
    "TOOL_CALL_ARGUMENT_BEGIN",
    "TOOL_CALL_BEGIN",
    "TOOL_CALL_END",
    "TOOL_CALL_FIELDS",
    "TOOL_CALL_PATTERN",
    "TOOL_CALLS_SECTION_BEGIN",
    "TOOL_CALLS_SECTION_END",
    "deduplicate_reasoning_parts",
    "extract_think_content",
    "extract_think_content_complete",
    "extract_tool_call_id_parts",
    "has_think_end_tag",
    "has_think_start_tag",
    "has_tool_call_tokens",
    "is_kimi_k2_model",
    "is_native_tool_call_model",
    "parse_tool_calls_from_content",
    "parse_tool_calls_from_message",
    "strip_native_tool_tokens",
    "strip_think_tags",
]
