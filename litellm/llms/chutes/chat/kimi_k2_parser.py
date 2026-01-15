"""
Transformation for Kimi-K2 models.

Kimi-K2 uses native special tokens for tool calls that need to be parsed
and converted to standard OpenAI format.

Native Format:
<|tool_calls_section_begin|>
<|tool_call_begin|> func_name
<|tool_call_argument_begin|> {...}
<|tool_call_end|>
<|tool_calls_section_end|>

Reference: https://huggingface.co/unsloth/Kimi-K2-Thinking-GGUF
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from litellm import verbose_logger


class KimiK2ToolParser:
    """
    Parser for Kimi-K2 native tool call format.

    Handles parsing of tool calls from special token format in
    reasoning_content or content fields.
    """

    # Special tokens used by Kimi-K2 for tool calls
    TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>"
    TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>"
    TOOL_CALL_BEGIN = "<|tool_call_begin|>"
    TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"
    TOOL_CALL_END = "<|tool_call_end|>"

    @staticmethod
    def _normalize_json_string(json_str: str) -> str:
        """
        Normalize JSON string to handle newlines and other formatting issues.

        Kimi-K2 may return JSON with literal newlines that need to be escaped.

        Args:
            json_str: The JSON string to normalize

        Returns:
            Normalized JSON string
        """
        # Replace literal newlines in JSON strings with escaped newlines
        # Only newlines inside string values (between quotes) should be escaped
        result = []
        in_string = False
        i = 0

        while i < len(json_str):
            char = json_str[i]

            if char == '"' and (i == 0 or json_str[i - 1] != '\\'):
                in_string = not in_string
                result.append(char)
            elif char == '\n' and in_string:
                # Escape newline inside string
                result.append('\\n')
            elif char == '\r' and in_string:
                # Escape carriage return inside string
                result.append('\\r')
            elif char == '\t' and in_string:
                # Escape tab inside string
                result.append('\\t')
            else:
                result.append(char)

            i += 1

        return ''.join(result)

    def extract_tool_calls(self, content: str) -> List[Dict]:
        """
        Extract tool calls from Kimi-K2 native format.

        Args:
            content: Text content that may contain tool calls

        Returns:
            List of tool call dictionaries in OpenAI format
        """
        if not content:
            return []

        tool_calls = []

        # Check if content contains tool call markers
        if self.TOOL_CALLS_SECTION_BEGIN not in content:
            verbose_logger.debug("No tool_calls_section_begin marker found in content")
            return []

        try:
            # Extract all tool calls from the section
            tool_calls_start = content.find(self.TOOL_CALLS_SECTION_BEGIN)
            if tool_calls_start == -1:
                return []

            tool_calls_start += len(self.TOOL_CALLS_SECTION_BEGIN)

            # Find the end of the tool calls section
            tool_calls_end = content.find(self.TOOL_CALLS_SECTION_END, tool_calls_start)
            if tool_calls_end == -1:
                # No end marker, use rest of content
                tool_calls_section = content[tool_calls_start:]
            else:
                tool_calls_section = content[tool_calls_start:tool_calls_end]

            # Parse individual tool calls
            tool_calls = self._parse_tool_calls_section(tool_calls_section)

            verbose_logger.debug(f"Parsed {len(tool_calls)} tool calls from Kimi-K2 format")
            return tool_calls

        except Exception as e:
            verbose_logger.error(f"Error parsing Kimi-K2 tool calls: {e}")
            return []

    def _parse_tool_calls_section(self, section: str) -> List[Dict]:
        """
        Parse individual tool calls from the tool calls section.

        Format:
        <|tool_call_begin|> function_name
        <|tool_call_argument_begin|> {...}
        <|tool_call_end|>
        """
        tool_calls = []
        tool_call_index = 0

        while True:
            # Find the start of the next tool call
            call_begin_pos = section.find(self.TOOL_CALL_BEGIN, tool_call_index)

            if call_begin_pos == -1:
                break

            # Find the argument section
            argument_begin_pos = section.find(self.TOOL_CALL_ARGUMENT_BEGIN, call_begin_pos)

            if argument_begin_pos == -1:
                verbose_logger.debug(f"No argument_begin found for tool call at {call_begin_pos}")
                break

            argument_begin_pos += len(self.TOOL_CALL_ARGUMENT_BEGIN)

            # Find the end of this tool call
            call_end_pos = section.find(self.TOOL_CALL_END, argument_begin_pos)

            if call_end_pos == -1:
                verbose_logger.debug(f"No tool_call_end found after argument_begin at {argument_begin_pos}")
                break

            # Extract function name
            name_section = section[call_begin_pos + len(self.TOOL_CALL_BEGIN):argument_begin_pos - len(self.TOOL_CALL_ARGUMENT_BEGIN)].strip()

            # Extract arguments
            arguments_str = section[argument_begin_pos:call_end_pos].strip()
            arguments_start = arguments_str.find("{")
            arguments_end = arguments_str.rfind("}")

            if arguments_start != -1 and arguments_end != -1:
                arguments_json = arguments_str[arguments_start:arguments_end + 1]

                try:
                    # Parse JSON arguments - handle escaped newlines and other issues
                    # The JSON may contain literal newlines that need to be escaped
                    arguments_json = self._normalize_json_string(arguments_json)
                    arguments = json.loads(arguments_json)
                except json.JSONDecodeError as e:
                    verbose_logger.error(f"Failed to parse tool call arguments: {arguments_json}, error: {e}")
                    arguments = {}

                # Create tool call in OpenAI format
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": name_section,
                        "arguments": json.dumps(arguments) if arguments else "{}"
                    }
                })
            else:
                # No valid JSON arguments found, create empty arguments
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": name_section,
                        "arguments": "{}"
                    }
                })

            # Move past this tool call
            tool_call_index = call_end_pos + len(self.TOOL_CALL_END)

        return tool_calls

    def extract_reasoning_content(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content and main content from Kimi-K2 response.

        Kimi-K2 may return tool calls in special tokens that should be
        removed from the main content.

        Args:
            content: Full content string

        Returns:
            Tuple of (reasoning_content, main_content)
        """
        if not content:
            return None, None

        # Check if there are tool calls in the content
        if self.TOOL_CALLS_SECTION_BEGIN in content:
            # The content contains tool calls - extract them and the reasoning
            tool_calls_start = content.find(self.TOOL_CALLS_SECTION_BEGIN)

            # Everything before tool calls is reasoning content
            reasoning = content[:tool_calls_start].strip()

            # Everything after tool calls section is main content
            tool_calls_end = content.find(self.TOOL_CALLS_SECTION_END, tool_calls_start)
            if tool_calls_end != -1:
                main_content = content[tool_calls_end + len(self.TOOL_CALLS_SECTION_END):].strip()
            else:
                main_content = ""

            # Remove reasoning content if it's just whitespace
            if not reasoning or reasoning.isspace():
                reasoning = None

            # Remove main content if it's just whitespace
            if not main_content or main_content.isspace():
                main_content = None

            return reasoning, main_content
        else:
            # No tool calls, content is main content
            return None, content if content.strip() else None

    def clean_tool_calls_from_content(self, content: str) -> Optional[str]:
        """
        Remove tool call markers from content.

        Args:
            content: Content that may contain tool calls

        Returns:
            Content with tool calls removed
        """
        if not content or self.TOOL_CALLS_SECTION_BEGIN not in content:
            return None if not content else content

        # Extract tool calls section
        tool_calls_start = content.find(self.TOOL_CALLS_SECTION_BEGIN)
        tool_calls_end = content.find(self.TOOL_CALLS_SECTION_END, tool_calls_start)

        if tool_calls_end == -1:
            # No end marker, remove everything from start
            before = content[:tool_calls_start].strip()
            return before if before else None
        else:
            # Remove the tool calls section
            before = content[:tool_calls_start].strip()
            after = content[tool_calls_end + len(self.TOOL_CALLS_SECTION_END):].strip()

            # Combine before and after content
            parts = []
            if before:
                parts.append(before)
            if after:
                parts.append(after)

            return " ".join(parts) if parts else None

    @classmethod
    def normalize_function_name(cls, function_name: str) -> str:
        """
        Normalize function names from Kimi-K2 provider responses.

        Kimi-K2 provider (especially through Vertex AI/Moonshot MaaS) may return
        function names with prefixes like "functions." and/or suffixes like ":N"
        that need to be cleaned up.

        Examples:
        - "functions.Read:3" -> "Read"
        - "functions.Bash" -> "Bash"
        - "Grep:1" -> "Grep"
        - "Bash" -> "Bash" (already normalized)

        Args:
            function_name: The raw function name from the provider

        Returns:
            Normalized function name
        """
        if not function_name:
            return function_name

        normalized = function_name

        # Remove "functions." prefix
        if normalized.startswith("functions."):
            normalized = normalized[len("functions."):]

        # Remove ":N" suffix (where N is a number)
        # Match pattern like ":3", ":10", ":123", etc.
        normalized = re.sub(r":\d+$", "", normalized)

        return normalized

    @classmethod
    def normalize_tool_calls(cls, tool_calls: List[Dict]) -> List[Dict]:
        """
        Normalize a list of tool calls from Kimi-K2 responses.

        This cleans up function names in tool_calls that may have been
        formatted incorrectly by the provider.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            List of tool calls with normalized function names
        """
        normalized_calls = []

        for tool_call in tool_calls:
            normalized_call = tool_call.copy()

            if "function" in normalized_call and "name" in normalized_call["function"]:
                original_name = normalized_call["function"]["name"]
                normalized_name = cls.normalize_function_name(original_name)

                if normalized_name != original_name:
                    verbose_logger.debug(
                        f"Normalized Kimi-K2 function name: '{original_name}' -> '{normalized_name}'"
                    )

                normalized_call["function"]["name"] = normalized_name

            normalized_calls.append(normalized_call)

        return normalized_calls

    @classmethod
    def is_kimi_k2_model(cls, model: str) -> bool:
        """
        Detect if the model is a Kimi K2 model.

        Args:
            model: Model name

        Returns:
            True if this is a Kimi K2 model
        """
        model_lower = model.lower()
        return (
            "kimi-k2" in model_lower
            or "kimi_k2" in model_lower
            or "kimik2" in model_lower
        )

    @classmethod
    def has_native_tool_calls(cls, response: Dict) -> bool:
        """
        Check if response contains native Kimi-K2 tool call format.

        Args:
            response: Raw API response

        Returns:
            True if response contains native tool calls
        """
        # Check reasoning_content
        choices = response.get("choices", [])
        if not choices:
            return False

        message = choices[0].get("message", {})

        # Check reasoning_content
        reasoning_content = message.get("reasoning_content", "")
        if reasoning_content and cls.TOOL_CALLS_SECTION_BEGIN in reasoning_content:
            return True

        # Check content
        content = message.get("content", "")
        if content and cls.TOOL_CALLS_SECTION_BEGIN in content:
            return True

        return False


# Convenience function for parsing
def parse_kimi_k2_tool_calls(content: str) -> List[Dict]:
    """
    Parse Kimi-K2 tool calls from content.

    Args:
        content: Text content containing tool calls

    Returns:
        List of tool call dictionaries
    """
    parser = KimiK2ToolParser()
    return parser.extract_tool_calls(content)


def extract_kimi_k2_reasoning_and_content(content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract reasoning and main content from Kimi-K2 response.

    Args:
        content: Full content string

    Returns:
        Tuple of (reasoning_content, main_content)
    """
    parser = KimiK2ToolParser()
    return parser.extract_reasoning_content(content)
