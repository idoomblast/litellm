"""
Parser for Kimi K2 native tool call format with special tokens.

Kimi K2 uses a specific format for tool calls with special tokens:
- <|tool_call_begin|> / <|tool_call_end|> - Wraps individual tool calls
- <|tool_call_argument_begin|> - Separates tool call ID from JSON arguments

Tool call ID format: functions.{func_name}:{idx} where idx is a global counter starting at 0.

IMPORTANT: The Chutes proxy sends tool calls in BOTH formats:
1. OpenAI standard format: delta.tool_calls[] array (streaming chunks)
2. Native Kimi tokens: delta.content field with <|tool_call_begin|>... sequences

To prevent duplicate emissions, we track if standard format was seen and skip native tokens.

The Chutes proxy also uses <think>...</think> tags for thinking content.

Example raw output from Chutes (no section wrappers, spaces around tokens):
<|tool_call_begin|> functions.get_weather:0 <|tool_call_argument_begin|> {"city": "Beijing"} <|tool_call_end|>

Example official Kimi K2 format (with section wrappers):
<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Beijing"}<|tool_call_end|>
<|tool_calls_section_end|>
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from litellm.types.utils import ChatCompletionMessageToolCall, Function

# Special tokens for Kimi K2 tool call format
# Note: Chutes proxy does NOT use section wrappers, only individual tool call markers
TOOL_CALL_BEGIN = "<|tool_call_begin|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"

# Legacy section markers (some implementations may still use these)
TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>"
TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>"

# Maximum special token length (for buffering in streaming)
MAX_SPECIAL_TOKEN_LENGTH = len(TOOL_CALL_ARGUMENT_BEGIN)  # 27 characters

# Regex pattern for individual tool calls (handles spaces around tokens)
# Matches: <|tool_call_begin|> functions.name:idx <|tool_call_argument_begin|> {...} <|tool_call_end|>
TOOL_CALL_PATTERN = re.compile(
    r"<\|tool_call_begin\|>\s*(?P<tool_call_id>functions\.[\w]+:\d+)\s*"
    r"<\|tool_call_argument_begin\|>\s*(?P<arguments>\{.*?\})\s*<\|tool_call_end\|>",
    re.DOTALL,
)

# Think tag pattern for Chutes proxy (may have spaces around tags)
# Matches: <think>...</think> or <think> ... </think>
THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Think tag constants for streaming state machine
THINK_START_TAG = "<think>"
THINK_END_TAG = "</think>"

# Fields to check for tool call tokens
TOOL_CALL_FIELDS = ["content", "reasoning", "reasoning_content", "thinking"]


def has_tool_call_tokens(content: str) -> bool:
    """
    Check if content contains Kimi K2 tool call tokens.

    Checks for both individual tool call markers and (legacy) section markers.

    Args:
        content: String to check for tool call tokens

    Returns:
        True if content contains tool call markers
    """
    return TOOL_CALL_BEGIN in content or TOOL_CALLS_SECTION_BEGIN in content


def has_think_start_tag(content: str) -> bool:
    """
    Check if content contains the start of a think tag.

    Args:
        content: String to check for think start tag

    Returns:
        True if content contains <think>
    """
    return THINK_START_TAG in content


def has_think_end_tag(content: str) -> bool:
    """
    Check if content contains the end of a think tag.

    Args:
        content: String to check for think end tag

    Returns:
        True if content contains </think>
    """
    return THINK_END_TAG in content


def extract_think_content_complete(content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract all thinking content from <think>...</think> tags, handling unclosed tags.

    This function handles:
    - Complete think blocks: <think>content</think>
    - Unclosed think tags: <think>content... (treats all content after as thinking)
    - Multiple think blocks: concatenates all thinking content
    - Mixed content: text before/after think tags

    Args:
        content: String that may contain <think>...</think> tags

    Returns:
        Tuple of:
        - All thinking content concatenated (or None if no think tags)
        - Remaining content with think tags removed (or None if empty)
    """
    if not content:
        return None, None

    thinking_parts: List[str] = []
    remaining_parts: List[str] = []

    current_pos = 0
    in_think_block = False

    while current_pos < len(content):
        if not in_think_block:
            # Looking for <think> start
            start_idx = content.find(THINK_START_TAG, current_pos)
            if start_idx == -1:
                # No more think tags, add remaining content
                remaining_parts.append(content[current_pos:])
                break
            else:
                # Add content before the think tag
                if start_idx > current_pos:
                    remaining_parts.append(content[current_pos:start_idx])
                # Move past the <think> tag
                current_pos = start_idx + len(THINK_START_TAG)
                in_think_block = True
        else:
            # Inside think block, looking for </think> end
            end_idx = content.find(THINK_END_TAG, current_pos)
            if end_idx == -1:
                # Unclosed tag - all remaining content is thinking
                thinking_parts.append(content[current_pos:])
                break
            else:
                # Add thinking content
                thinking_parts.append(content[current_pos:end_idx])
                # Move past the </think> tag
                current_pos = end_idx + len(THINK_END_TAG)
                in_think_block = False

    # Combine thinking and remaining parts
    thinking = "".join(thinking_parts).strip() if thinking_parts else None
    remaining = "".join(remaining_parts).strip() if remaining_parts else None

    # Return None instead of empty strings
    return (thinking if thinking else None, remaining if remaining else None)


def extract_think_content(content: str) -> Tuple[Optional[str], str]:
    """
    Extract thinking content from <think>...</think> tags.

    The Chutes proxy uses these tags to wrap thinking/reasoning content
    in the delta.content field.

    Args:
        content: String that may contain <think>...</think> tags

    Returns:
        Tuple of:
        - Thinking content (or None if no think tags found)
        - Remaining content with think tags removed
    """
    if not content:
        return None, ""

    match = THINK_TAG_PATTERN.search(content)
    if match:
        thinking = match.group(1).strip()
        cleaned = THINK_TAG_PATTERN.sub("", content).strip()
        return thinking if thinking else None, cleaned

    return None, content


def strip_think_tags(content: str) -> str:
    """
    Strip <think>...</think> tags from content, keeping any remaining text.

    Args:
        content: String that may contain <think>...</think> tags

    Returns:
        Content with think tags and their contents removed
    """
    if not content:
        return ""
    return THINK_TAG_PATTERN.sub("", content).strip()


def strip_native_tool_tokens(content: str) -> str:
    """
    Strip all native Kimi K2 tool call tokens from content.

    This removes:
    - Complete tool call patterns: <|tool_call_begin|>...<|tool_call_end|>
    - Section markers: <|tool_calls_section_begin|>, <|tool_calls_section_end|>
    - Any remaining partial tokens

    Args:
        content: String that may contain tool call tokens

    Returns:
        Content with all tool call tokens removed
    """
    if not content:
        return ""

    # Remove complete tool call patterns first
    cleaned = TOOL_CALL_PATTERN.sub("", content)

    # Remove section markers
    cleaned = re.sub(r"<\|tool_calls_section_begin\|>", "", cleaned)
    cleaned = re.sub(r"<\|tool_calls_section_end\|>", "", cleaned)

    # Remove any remaining partial tool call tokens
    cleaned = re.sub(r"<\|tool_call[^|]*\|>", "", cleaned)

    return cleaned.strip()


def extract_tool_call_id_parts(tool_call_id: str) -> Tuple[str, int]:
    """
    Extract function name and index from a Kimi K2 tool call ID.

    Args:
        tool_call_id: Tool call ID in format "functions.{func_name}:{idx}"

    Returns:
        Tuple of (function_name, index)

    Raises:
        ValueError: If tool_call_id format is invalid
    """
    # Expected format: functions.get_weather:0
    if not tool_call_id.startswith("functions."):
        raise ValueError(f"Invalid tool call ID format: {tool_call_id}")

    # Remove "functions." prefix
    remainder = tool_call_id[len("functions.") :]

    # Split by ":" to get function name and index
    if ":" not in remainder:
        raise ValueError(f"Invalid tool call ID format (missing index): {tool_call_id}")

    parts = remainder.rsplit(":", 1)
    func_name = parts[0]
    try:
        idx = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid tool call ID format (invalid index): {tool_call_id}")

    return func_name, idx


def parse_tool_calls_from_content(
    content: str,
) -> Tuple[Optional[List[ChatCompletionMessageToolCall]], str]:
    """
    Parse Kimi K2 native tool call format from a single content string.

    This function:
    1. Finds all tool calls marked by special tokens (with or without section wrappers)
    2. Extracts individual tool calls
    3. Removes the tool call tokens from the content

    Args:
        content: String that may contain Kimi K2 tool call tokens

    Returns:
        Tuple of:
        - List of ChatCompletionMessageToolCall objects (or None if no tool calls found)
        - Remaining content with tool call tokens removed
    """
    if not content:
        return None, ""

    # Check for tool call tokens first
    if not has_tool_call_tokens(content):
        return None, content

    tool_calls: List[ChatCompletionMessageToolCall] = []
    cleaned_content = content

    # Find all individual tool calls using regex
    matches = list(TOOL_CALL_PATTERN.finditer(content))

    for match in matches:
        tool_call_id = match.group("tool_call_id").strip()
        arguments_str = match.group("arguments").strip()

        try:
            # Extract function name from tool call ID
            func_name, _ = extract_tool_call_id_parts(tool_call_id)

            # Validate JSON arguments
            try:
                # Parse to validate, but keep as string
                json.loads(arguments_str)
            except json.JSONDecodeError:
                # If JSON is invalid, try to use it as-is
                # This allows for graceful handling of malformed responses
                pass

            # Create the tool call object
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
            # Skip malformed tool call IDs
            continue

    # Remove all matched tool call patterns from content
    if matches:
        cleaned_content = TOOL_CALL_PATTERN.sub("", content)
        # Also clean up any remaining partial tokens or section markers
        cleaned_content = re.sub(r"<\|tool_call[^|]*\|>", "", cleaned_content)
        cleaned_content = re.sub(r"<\|tool_calls_section[^|]*\|>", "", cleaned_content)
        cleaned_content = cleaned_content.strip()

    if tool_calls:
        return tool_calls, cleaned_content
    else:
        return None, content


def parse_tool_calls_from_message(
    message: Dict[str, Any],
) -> Tuple[Optional[List[ChatCompletionMessageToolCall]], Dict[str, Any]]:
    """
    Parse Kimi K2 tool calls from ALL possible fields in a message.

    Tool call tokens can appear in any of these fields:
    - content: Main response content
    - reasoning: Reasoning field
    - reasoning_content: Reasoning content field
    - thinking: Thinking blocks content

    This function checks all fields and combines any tool calls found.

    Args:
        message: Message dictionary that may contain tool call tokens in various fields

    Returns:
        Tuple of:
        - Combined list of tool calls from all fields (or None if no tool calls found)
        - Updated message dict with tool call tokens removed from all fields
    """
    all_tool_calls: List[ChatCompletionMessageToolCall] = []
    updated_message = message.copy()

    for field in TOOL_CALL_FIELDS:
        field_value = message.get(field)
        if field_value and isinstance(field_value, str):
            tool_calls, cleaned_content = parse_tool_calls_from_content(field_value)
            if tool_calls:
                all_tool_calls.extend(tool_calls)
            # Update the field with cleaned content (or None if empty)
            if cleaned_content:
                updated_message[field] = cleaned_content
            else:
                # Set to None if content is empty after removing tool calls
                updated_message[field] = None

    return all_tool_calls if all_tool_calls else None, updated_message


def is_kimi_k2_model(model: str) -> bool:
    """
    Check if the model is a Kimi K2 model that uses native tool call format.

    Args:
        model: Model name/identifier

    Returns:
        True if the model is a Kimi K2 model
    """
    model_lower = model.lower()
    return "kimi" in model_lower and "k2" in model_lower
