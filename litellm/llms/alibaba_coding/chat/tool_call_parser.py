import json
import re
from typing import Any, Dict, List, Optional, Tuple

from litellm.types.utils import ChatCompletionMessageToolCall, Function

TOOL_CALL_BEGIN = "<|tool_call_begin|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"

TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>"
TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>"

MAX_SPECIAL_TOKEN_LENGTH = len(TOOL_CALL_ARGUMENT_BEGIN)

TOOL_CALL_PATTERN = re.compile(
    r"<\|tool_call_begin\|>\s*(?P<tool_call_id>functions\.[\w]+:\d+)\s*"
    r"<\|tool_call_argument_begin\|>\s*(?P<arguments>\{.*?\})\s*<\|tool_call_end\|>",
    re.DOTALL,
)

THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
THINK_START_TAG = "<think>"
THINK_END_TAG = "</think>"

TOOL_CALL_FIELDS = ["content", "reasoning", "reasoning_content", "thinking"]


def has_tool_call_tokens(content: str) -> bool:
    return TOOL_CALL_BEGIN in content or TOOL_CALLS_SECTION_BEGIN in content


def has_think_start_tag(content: str) -> bool:
    return THINK_START_TAG in content


def has_think_end_tag(content: str) -> bool:
    return THINK_END_TAG in content


def extract_think_content_complete(content: str) -> Tuple[Optional[str], Optional[str]]:
    if not content:
        return None, None

    thinking_parts: List[str] = []
    remaining_parts: List[str] = []

    current_pos = 0
    in_think_block = False

    while current_pos < len(content):
        if not in_think_block:
            start_idx = content.find(THINK_START_TAG, current_pos)
            if start_idx == -1:
                remaining_parts.append(content[current_pos:])
                break
            if start_idx > current_pos:
                remaining_parts.append(content[current_pos:start_idx])
            current_pos = start_idx + len(THINK_START_TAG)
            in_think_block = True
        else:
            end_idx = content.find(THINK_END_TAG, current_pos)
            if end_idx == -1:
                thinking_parts.append(content[current_pos:])
                break
            thinking_parts.append(content[current_pos:end_idx])
            current_pos = end_idx + len(THINK_END_TAG)
            in_think_block = False

    thinking = "".join(thinking_parts).strip() if thinking_parts else None
    remaining = "".join(remaining_parts).strip() if remaining_parts else None
    return (thinking if thinking else None, remaining if remaining else None)


def extract_think_content(content: str) -> Tuple[Optional[str], str]:
    if not content:
        return None, ""

    match = THINK_TAG_PATTERN.search(content)
    if match:
        thinking = match.group(1).strip()
        cleaned = THINK_TAG_PATTERN.sub("", content).strip()
        return thinking if thinking else None, cleaned

    return None, content


def strip_think_tags(content: str) -> str:
    if not content:
        return ""
    return THINK_TAG_PATTERN.sub("", content).strip()


def strip_native_tool_tokens(content: str) -> str:
    if not content:
        return ""

    cleaned = TOOL_CALL_PATTERN.sub("", content)
    cleaned = re.sub(r"<\|tool_calls_section_begin\|>", "", cleaned)
    cleaned = re.sub(r"<\|tool_calls_section_end\|>", "", cleaned)
    cleaned = re.sub(r"<\|tool_call[^|]*\|>", "", cleaned)
    return cleaned.strip()


def extract_tool_call_id_parts(tool_call_id: str) -> Tuple[str, int]:
    if not tool_call_id.startswith("functions."):
        raise ValueError(f"Invalid tool call ID format: {tool_call_id}")

    remainder = tool_call_id[len("functions.") :]
    if ":" not in remainder:
        raise ValueError(f"Invalid tool call ID format (missing index): {tool_call_id}")

    parts = remainder.rsplit(":", 1)
    func_name = parts[0].strip()
    try:
        idx = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid tool call ID format (invalid index): {tool_call_id}")

    return func_name, idx


def parse_tool_calls_from_content(
    content: str,
) -> Tuple[Optional[List[ChatCompletionMessageToolCall]], str]:
    if not content:
        return None, ""

    if not has_tool_call_tokens(content):
        return None, content

    tool_calls: List[ChatCompletionMessageToolCall] = []

    matches = list(TOOL_CALL_PATTERN.finditer(content))
    for match in matches:
        tool_call_id = match.group("tool_call_id").strip()
        arguments_str = match.group("arguments").strip()

        try:
            func_name, _ = extract_tool_call_id_parts(tool_call_id)
            try:
                json.loads(arguments_str)
            except json.JSONDecodeError:
                pass

            tool_call = ChatCompletionMessageToolCall(
                id=tool_call_id,
                type="function",
                function=Function(
                    name=func_name,
                    arguments=arguments_str,
                ),
            )
            tool_calls.append(tool_call)
        except ValueError:
            continue

    if matches:
        cleaned_content = TOOL_CALL_PATTERN.sub("", content)
        cleaned_content = re.sub(r"<\|tool_call[^|]*\|>", "", cleaned_content)
        cleaned_content = re.sub(r"<\|tool_calls_section[^|]*\|>", "", cleaned_content)
        cleaned_content = cleaned_content.strip()
        if tool_calls:
            return tool_calls, cleaned_content

    return None, content


def parse_tool_calls_from_message(
    message: Dict[str, Any],
) -> Tuple[Optional[List[ChatCompletionMessageToolCall]], Dict[str, Any]]:
    all_tool_calls: List[ChatCompletionMessageToolCall] = []
    updated_message = message.copy()

    for field in TOOL_CALL_FIELDS:
        field_value = message.get(field)
        if field_value and isinstance(field_value, str):
            tool_calls, cleaned_content = parse_tool_calls_from_content(field_value)
            if tool_calls:
                all_tool_calls.extend(tool_calls)
            if cleaned_content:
                updated_message[field] = cleaned_content
            else:
                updated_message[field] = None

    return all_tool_calls if all_tool_calls else None, updated_message


def deduplicate_reasoning_parts(parts: List[str]) -> Optional[str]:
    if not parts:
        return None
    seen = set()
    unique = []
    for part in parts:
        stripped = part.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            unique.append(stripped)
    if not unique:
        return None
    return "\n".join(unique)


def is_native_tool_call_model(model: str) -> bool:
    model_lower = model.lower()
    return "kimi" in model_lower and "k2" in model_lower


__all__ = [
    "MAX_SPECIAL_TOKEN_LENGTH",
    "THINK_END_TAG",
    "THINK_START_TAG",
    "TOOL_CALL_ARGUMENT_BEGIN",
    "TOOL_CALL_BEGIN",
    "TOOL_CALL_END",
    "TOOL_CALL_FIELDS",
    "TOOL_CALLS_SECTION_BEGIN",
    "TOOL_CALLS_SECTION_END",
    "deduplicate_reasoning_parts",
    "extract_think_content_complete",
    "extract_tool_call_id_parts",
    "has_think_start_tag",
    "has_tool_call_tokens",
    "is_native_tool_call_model",
    "parse_tool_calls_from_content",
    "strip_native_tool_tokens",
    "strip_think_tags",
]
