"""
Streaming handler for Kimi-K2 models.

Handles streaming responses with Kimi-K2 native special token format.
The model is detected from the first chunk's "model" field.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

from litellm import verbose_logger
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator

if TYPE_CHECKING:
    from litellm.types.utils import GenericStreamingChunk, ModelResponse, ModelResponseStream
else:
    from litellm.types.utils import GenericStreamingChunk as _GenericStreamingChunk
    from litellm.types.utils import ModelResponse, ModelResponseStream

    GenericStreamingChunk = _GenericStreamingChunk


class KimiK2StreamingHandler(BaseModelResponseIterator):
    """
    Streaming response handler for Kimi-K2 models.

    Handles parsing of Kimi-K2 native special tokens during streaming.
    Model is detected from the first chunk's "model" field.

    Special Token Format:
    <|tool_calls_section_begin|>
    <|tool_call_begin|> function_name
    <|tool_call_argument_begin|> {...}
    <|tool_call_end|>
    <|tool_calls_section_end|>
    """

    # Special tokens used by Kimi-K2 for tool calls
    TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>"
    TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>"
    TOOL_CALL_BEGIN = "<|tool_call_begin|>"
    TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"
    TOOL_CALL_END = "<|tool_call_end|>"

    def __init__(
        self,
        streaming_response: Union[Any, List[ModelResponse]],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        super().__init__(streaming_response, sync_stream, json_mode)
        self._model: Optional[str] = None
        self._buffer = ""
        self._should_parse_kimi_tool_calls = False

    def chunk_parser(self, chunk: dict) -> Union[GenericStreamingChunk, ModelResponseStream]:
        """
        Parse a single streaming chunk.

        Args:
            chunk: The parsed JSON chunk dictionary

        Returns:
            Parsed chunk with Kimi-K2 native format converted to OpenAI format
        """
        from .kimi_k2_parser import KimiK2ToolParser

        # Extract model from first chunk if not set
        if self._model is None:
            self._model = chunk.get("model", "")
            # Check if this is a Kimi-K2 model
            self._should_parse_kimi_tool_calls = self._is_kimi_k2_model(self._model)

        # Check for existing tool_calls in OpenAI format (provider may return these)
        existing_tool_calls = self._extract_tool_calls_from_delta(chunk)

        # If existing tool_calls found and it's a Kimi-K2 model, normalize them
        if existing_tool_calls and self._should_parse_kimi_tool_calls:
            normalized_tool_calls = KimiK2ToolParser.normalize_tool_calls(existing_tool_calls)
            # Return chunk with normalized tool calls
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=chunk.get("index", 0),
                tool_use={
                    "id": normalized_tool_calls[0].get("id", "call_0"),
                    "type": "function",
                    "function": {
                        "name": normalized_tool_calls[0]["function"]["name"],
                        "arguments": normalized_tool_calls[0]["function"]["arguments"],
                    }
                }
                if normalized_tool_calls
                else None,
            )

        # Extract text/delta from chunk
        text = self._extract_text_from_chunk(chunk)

        # If not Kimi-K2 model, return plain text chunk
        if not self._should_parse_kimi_tool_calls:
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

        # Process text through Kimi-K2 parser
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

    def _is_kimi_k2_model(self, model: str) -> bool:
        """
        Detect if the model is a Kimi-K2 model.

        Args:
            model: Model name from chunk

        Returns:
            True if this is a Kimi-K2 model
        """
        model_lower = model.lower()
        return (
            "kimi-k2" in model_lower
            or "kimik2" in model_lower
            or "kimi_k2" in model_lower
        )

    def _process_text_chunk(self, text: str, original_chunk: dict) -> GenericStreamingChunk:
        """
        Process a text chunk and yield any Kimi-K2 tool call chunks.

        Args:
            text: The text content from the chunk
            original_chunk: The original chunk for context

        Returns:
            Parsed chunk with Kimi-K2 native format converted
        """
        if not text:
            return self._create_empty_chunk(original_chunk)

        # Check if this chunk contains tool call markers
        if self.TOOL_CALLS_SECTION_BEGIN not in text and self.TOOL_CALLS_SECTION_BEGIN not in self._buffer:
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

        # Check if we complete tool calls section
        if self.TOOL_CALLS_SECTION_END in self._buffer:
            # We have a complete tool calls section
            return self._parse_complete_tool_calls(original_chunk)
        else:
            # Still waiting for end marker, wait for next chunk
            # Don't return text if we're buffering tool call markers
            if self.TOOL_CALLS_SECTION_BEGIN in self._buffer:
                return self._create_empty_chunk(original_chunk)
            else:
                return GenericStreamingChunk(
                    text=text,
                    is_finished=False,
                    finish_reason="",
                    usage=None,
                    index=original_chunk.get("index", 0),
                    tool_use=None,
                )

    def _parse_complete_tool_calls(self, original_chunk: dict) -> GenericStreamingChunk:
        """
        Parse complete tool calls from accumulated buffer.

        Args:
            original_chunk: The original chunk for context

        Returns:
            Chunk with tool_calls field populated if found
        """
        tool_calls = []

        try:
            # Extract tool calls section
            start_pos = self._buffer.find(self.TOOL_CALLS_SECTION_BEGIN)
            if start_pos != -1:
                start_pos += len(self.TOOL_CALLS_SECTION_BEGIN)

            end_pos = self._buffer.find(self.TOOL_CALLS_SECTION_END)
            if end_pos != -1:
                tool_calls_section = self._buffer[start_pos:end_pos]
                tool_calls = self._parse_tool_calls_section(tool_calls_section)

        except Exception as e:
            verbose_logger.error(f"Error parsing Kimi-K2 tool calls from streaming chunk: {e}")

        # Update buffer to only keep content after tool calls
        if self.TOOL_CALLS_SECTION_END in self._buffer:
            end_pos = self._buffer.find(self.TOOL_CALLS_SECTION_END)
            if end_pos != -1:
                remaining_text = self._buffer[end_pos + len(self.TOOL_CALLS_SECTION_END):].strip()
                self._buffer = remaining_text

        # Return chunk with tool calls
        if tool_calls:
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=original_chunk.get("index", 0),
                tool_use={
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": tool_calls[0]["function"]["name"],
                        "arguments": tool_calls[0]["function"]["arguments"],
                    }
                },
            )
        else:
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=original_chunk.get("index", 0),
                tool_use=None,
            )

    def _parse_tool_calls_section(self, section: str) -> List[Dict]:
        """
        Parse individual tool calls from the tool calls section.

        Args:
            section: The tool calls section content

        Returns:
            List of tool calls in OpenAI format
        """
        tool_calls = []
        tool_call_index = 0

        while True:
            # Find the start of the next tool call
            call_begin_pos = section.find(self.TOOL_CALL_BEGIN)
            if call_begin_pos == -1:
                break

            # Find the argument section
            argument_begin_pos = section.find(self.TOOL_CALL_ARGUMENT_BEGIN, call_begin_pos)
            if argument_begin_pos == -1:
                break

            argument_begin_pos += len(self.TOOL_CALL_ARGUMENT_BEGIN)

            # Find the end of this tool call
            call_end_pos = section.find(self.TOOL_CALL_END, argument_begin_pos)
            if call_end_pos == -1:
                break

            # Extract function name
            name_section = section[call_begin_pos + len(self.TOOL_CALL_BEGIN):argument_begin_pos - len(self.TOOL_CALL_ARGUMENT_BEGIN)].strip()

            # Extract arguments
            arguments_str = section[argument_begin_pos:call_end_pos].strip()
            arguments_start = arguments_str.find("{")
            arguments_end = arguments_str.rfind("}")

            tool_call = None
            if arguments_start != -1 and arguments_end != -1:
                arguments_json = arguments_str[arguments_start:arguments_end + 1]

                try:
                    arguments = json.loads(arguments_json)
                    tool_call = {
                        "id": f"call_{tool_call_index}",
                        "type": "function",
                        "function": {
                            "name": name_section,
                            "arguments": json.dumps(arguments) if arguments else "{}"
                        }
                    }
                except json.JSONDecodeError:
                    tool_call = {
                        "id": f"call_{tool_call_index}",
                        "type": "function",
                        "function": {
                            "name": name_section,
                            "arguments": "{}"
                        }
                    }

            if tool_call:
                tool_calls.append(tool_call)

            # Move past this tool call
            tool_call_index = call_end_pos + len(self.TOOL_CALL_END)

            section = section[tool_call_index:]

        return tool_calls
