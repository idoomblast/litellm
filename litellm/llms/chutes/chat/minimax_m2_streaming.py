"""
Streaming handler for MiniMax M2.1 models.

Handles streaming responses with MiniMax M2 native XML format.
The model is detected from the first chunk's "model" field.
"""

import json
import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

from litellm import verbose_logger
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator

if TYPE_CHECKING:
    from litellm.types.utils import GenericStreamingChunk, ModelResponse, ModelResponseStream
else:
    from litellm.types.utils import GenericStreamingChunk as _GenericStreamingChunk
    from litellm.types.utils import ModelResponse, ModelResponseStream

    GenericStreamingChunk = _GenericStreamingChunk


class MiniMaxM2StreamingHandler(BaseModelResponseIterator):
    """
    Streaming response handler for MiniMax M2.1 models.

    Handles parsing of MiniMax M2 native XML format during streaming.
    Model is detected from the first chunk's "model" field.

    XML Format:
    <minimax:tool_call>
    <invoke name="function_name">
    <parameter name="param_name">value</parameter>
    </invoke>
    </minimax:tool_call>
    """

    # XML markers used by MiniMax M2 for tool calls
    TOOL_CALL_BEGIN = "<minimax:tool_call>"
    TOOL_CALL_END = "</minimax:tool_call>"
    INVOKE_BEGIN = "<invoke"
    INVOKE_END = "</invoke>"
    PARAMETER_BEGIN = "<parameter"
    PARAMETER_END = "</parameter>"

    def __init__(
        self,
        streaming_response: Union[Any, List[ModelResponse]],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> None:
        super().__init__(streaming_response, sync_stream, json_mode)
        self._model: Optional[str] = None
        self._buffer = ""
        self._should_parse_minimax_tool_calls = False

    def chunk_parser(self, chunk: dict) -> Union[GenericStreamingChunk, ModelResponseStream]:
        """
        Parse a single streaming chunk.

        Args:
            chunk: The parsed JSON chunk dictionary

        Returns:
            Parsed chunk with MiniMax native format converted to OpenAI format
        """
        # Extract model from first chunk if not set
        if self._model is None:
            self._model = chunk.get("model", "")
            # Check if this is a MiniMax M2 model
            self._should_parse_minimax_tool_calls = self._is_minimax_m2_model(self._model)

        # Check for existing tool_calls in OpenAI format (provider may return these)
        existing_tool_calls = self._extract_tool_calls_from_delta(chunk)

        # If existing tool_calls found and it's a MiniMax M2 model, return them as-is
        if existing_tool_calls and self._should_parse_minimax_tool_calls:
            # Provider already converted to OpenAI format, just pass through
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=chunk.get("index", 0),
                tool_use={
                    "id": existing_tool_calls[0].get("id", "call_0"),
                    "type": "function",
                    "function": {
                        "name": existing_tool_calls[0].get("function", {}).get("name", ""),
                        "arguments": existing_tool_calls[0].get("function", {}).get("arguments", "{}"),
                    }
                }
                if existing_tool_calls
                else None,
            )

        # Extract text/delta from chunk
        text = self._extract_text_from_chunk(chunk)

        # If not MiniMax M2 model or no text, return plain text chunk
        if not self._should_parse_minimax_tool_calls:
            return GenericStreamingChunk(
                text=text if text else "",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=chunk.get("index", 0),
                tool_use=None,
            )

        # If no text, return empty chunk
        if not text:
            return self._create_empty_chunk(chunk)

        # Process text through MiniMax XML parser
        return self._process_text_chunk(text, chunk)

    def _extract_tool_calls_from_delta(self, chunk: dict) -> Optional[List[Dict]]:
        """Extract existing tool_calls from the chunk delta (OpenAI format)."""
        if "choices" in chunk and chunk["choices"]:
            for choice in chunk["choices"]:
                if "delta" in choice:
                    delta = choice["delta"]
                    # Check for tool_calls in delta (OpenAI streaming format)
                    if "tool_calls" in delta and delta["tool_calls"]:
                        return delta["tool_calls"]
        return None

    def _extract_text_from_chunk(self, chunk: dict) -> str:
        """Extract text content from a streaming chunk."""
        # Convert chunk to JSON string and extract content
        if "choices" in chunk and chunk["choices"]:
            for choice in chunk["choices"]:
                if "delta" in choice:
                    delta = choice["delta"]
                    if isinstance(delta, dict):
                        if "content" in delta and delta["content"]:
                            return delta["content"]
                        elif "reasoning_content" in delta and delta["reasoning_content"]:
                            return delta["reasoning_content"]
        return ""

    def _create_empty_chunk(self, chunk: dict) -> GenericStreamingChunk:
        """Create an empty chunk when there's no text."""
        return GenericStreamingChunk(
            text="",
            is_finished=False,
            finish_reason="",
            usage=None,
            index=chunk.get("index", 0),
            tool_use=None,
        )

    def _is_minimax_m2_model(self, model: str) -> bool:
        """
        Detect if the model is a MiniMax M2 model.

        Args:
            model: Model name from chunk

        Returns:
            True if this is a MiniMax M2 model
        """
        model_lower = model.lower()
        return (
            "minimax-m2" in model_lower
            or "minimax_m2" in model_lower
            or "minimax m2" in model_lower
        )

    def _process_text_chunk(self, text: str, original_chunk: dict) -> GenericStreamingChunk:
        """
        Process a text chunk and yield any MiniMax tool call chunks.

        Args:
            text: The text content from the chunk
            original_chunk: The original chunk for context

        Returns:
            Parsed chunk with MiniMax native format converted
        """
        if not text:
            return self._create_empty_chunk(original_chunk)

        # Check if this chunk contains tool call markers
        if self.TOOL_CALL_BEGIN not in text and self.TOOL_CALL_BEGIN not in self._buffer:
            # No tool call markers in this chunk or buffer, return as-is
            return GenericStreamingChunk(
                text=text,
                is_finished=False,
                finish_reason="",
                usage=None,
                index=original_chunk.get("index", 0),
                tool_use=None,
            )

        # Accumulate buffer for tool call markers
        self._buffer += text

        # Check if we have any complete tool_call blocks
        return self._parse_buffered_tool_calls(original_chunk)

    def _parse_buffered_tool_calls(self, original_chunk: dict) -> GenericStreamingChunk:
        """
        Parse complete tool calls from accumulated buffer.

        Args:
            original_chunk: The original chunk for context

        Returns:
            Chunk with tool_use field populated if found, or text chunk if none found
        """
        # Check if we have a complete tool_call block
        if self.TOOL_CALL_END not in self._buffer:
            # Still waiting for end marker, return empty chunk if we're buffering
            if self.TOOL_CALL_BEGIN in self._buffer:
                return self._create_empty_chunk(original_chunk)
            # Return any text before the tool_call marker
            before_tool_call = self._buffer.split(self.TOOL_CALL_BEGIN)[0]
            return GenericStreamingChunk(
                text=before_tool_call,
                is_finished=False,
                finish_reason="",
                usage=None,
                index=original_chunk.get("index", 0),
                tool_use=None,
            )

        # Extract and parse complete tool_call blocks
        tool_call = self._extract_complete_tool_call()

        # Update buffer to remove the processed tool_call block
        if self.TOOL_CALL_END in self._buffer:
            end_pos = self._buffer.find(self.TOOL_CALL_END) + len(self.TOOL_CALL_END)
            self._buffer = self._buffer[end_pos:]

        # Return chunk with tool calls if found
        if tool_call:
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=original_chunk.get("index", 0),
                tool_use=tool_call,
            )
        else:
            return self._create_empty_chunk(original_chunk)

    def _extract_complete_tool_call(self) -> Optional[Dict]:
        """
        Extract a complete tool call from the buffer.

        Returns:
            Tool call in OpenAI format or None if incomplete
        """
        try:
            # Find tool_call block
            start_pos = self._buffer.find(self.TOOL_CALL_BEGIN)
            if start_pos == -1:
                return None

            end_pos = self._buffer.find(self.TOOL_CALL_END)
            if end_pos == -1:
                return None

            # Extract the tool_call block
            tool_call_block = self._buffer[start_pos + len(self.TOOL_CALL_BEGIN):end_pos]

            # Parse the tool call block
            return self._parse_xml_tool_call(tool_call_block)

        except Exception as e:
            verbose_logger.error(f"Error extracting MiniMax tool call from streaming buffer: {e}")
            return None

    def _parse_xml_tool_call(self, xml_content: str) -> Optional[Dict]:
        """
        Parse a single MiniMax XML tool call block.

        Args:
            xml_content: The XML content between <minimax:tool_call> tags

        Returns:
            Tool call in OpenAI format or None if parsing fails
        """
        try:
            # Find <invoke name="function_name">
            invoke_match = re.search(r'<invoke name="([^"]+)"', xml_content)
            if not invoke_match:
                return None

            function_name = invoke_match.group(1)

            # Find all <parameter name="...">value</parameter>
            param_dict = {}
            param_regex = re.compile(r'<parameter name="([^"]+)">(.*?)</parameter>', re.DOTALL)
            for param_match in param_regex.finditer(xml_content):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()

                # Remove leading and trailing newlines
                param_value = param_value.strip()

                # Try to convert to appropriate type
                param_dict[param_name] = self._convert_param_value(param_value)

            # Create tool call in OpenAI format
            return {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(param_dict) if param_dict else "{}"
                }
            }

        except Exception as e:
            verbose_logger.error(f"Error parsing MiniMax XML tool call: {e}")
            return None

    def _convert_param_value(self, value: str) -> Any:
        """
        Convert parameter value based on content.

        Args:
            value: The string value from XML

        Returns:
            Converted value (int, float, bool, string, or JSON object/array)
        """
        value = value.strip()

        # Handle null
        if value.lower() == "null":
            return None

        # Handle boolean
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False

        # Handle numbers
        try:
            # Try integer first
            if "." not in value:
                return int(value)
            # Try float
            val = float(value)
            return int(val) if val == int(val) else val
        except ValueError:
            pass

        # Try JSON parsing for objects and arrays
        if value.startswith("{") or value.startswith("["):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        # Default to string
        return value