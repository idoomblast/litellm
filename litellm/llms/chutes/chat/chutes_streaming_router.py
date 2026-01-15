"""
Streaming router for Chutes platform.

Routes streaming requests to appropriate handlers based on model detection.
Model is detected from the first chunk's "model" field.
"""

from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator

if TYPE_CHECKING:
    from litellm.types.utils import GenericStreamingChunk, ModelResponse, ModelResponseStream
else:
    from litellm.types.utils import GenericStreamingChunk as _GenericStreamingChunk
    from litellm.types.utils import ModelResponse, ModelResponseStream

    GenericStreamingChunk = _GenericStreamingChunk


class ChutesStreamingRouter(BaseModelResponseIterator):
    """
    Streaming response router for Chutes platform.

    Routes streaming requests to appropriate specialized handlers:
    - KimiK2StreamingHandler for Kimi-K2 models (native special token format)
    - MiniMaxM2StreamingHandler for MiniMax M2 models (native XML format)
    - Default OpenAI handler for other models

    Model is detected from the first chunk's "model" field.
    """

    def __init__(
        self,
        streaming_response: Union[Any, List[ModelResponseStream]],
        sync_stream: bool,
        json_mode: Optional[bool] = None,
    ) -> None:
        super().__init__(streaming_response, sync_stream, json_mode)
        self._model: Optional[str] = None
        self._handler: Optional[BaseModelResponseIterator] = None
        self._chunks_buffer: List[dict] = []
        self._use_default_handler = False

    def chunk_parser(self, chunk: dict) -> Union[GenericStreamingChunk, ModelResponseStream]:
        """
        Parse a single streaming chunk.

        On first chunk, detects the model and routes to appropriate handler.
        Subsequent chunks are delegated to the detected handler.

        Args:
            chunk: The parsed JSON chunk dictionary

        Returns:
            Parsed chunk with provider-specific conversion applied
        """
        # If using default handler, delegate to parent's chunk_parser
        if self._use_default_handler:
            return super().chunk_parser(chunk)

        # Store first chunk for model detection
        if self._model is None:
            self._model = chunk.get("model", "")
            self._handler = self._get_handler_for_model(self._model)

            # Process buffered chunks (if any)
            if self._chunks_buffer:
                for buffered_chunk in self._chunks_buffer:
                    self._handler.chunk_parser(buffered_chunk)
                self._chunks_buffer.clear()

            return self._handler.chunk_parser(chunk)

        # Buffer chunks until model is detected
        if self._handler is None and self._model is None:
            self._chunks_buffer.append(chunk)
            # Return empty chunk while buffering
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=chunk.get("index", 0),
                tool_use=None,
            )

        # Delegate to handler
        return self._handler.chunk_parser(chunk)

    def _get_handler_for_model(self, model: str) -> BaseModelResponseIterator:
        """
        Get the appropriate streaming handler for the detected model.

        Args:
            model: Model name from chunk

        Returns:
            Appropriate streaming handler instance
        """
        model_lower = model.lower()

        # Check for Kimi-K2 models
        if (
            "kimi-k2" in model_lower
            or "kimik2" in model_lower
            or "kimi_k2" in model_lower
        ):
            from .kimi_k2_streaming import KimiK2StreamingHandler

            return KimiK2StreamingHandler(
                streaming_response=self.streaming_response,
                sync_stream=self.sync_stream,
                json_mode=self.json_mode,
            )

        # Check for MiniMax M2 models
        if (
            "minimax-m2" in model_lower
            or "minimax_m2" in model_lower
            or "minimax m2" in model_lower
        ):
            from .minimax_m2_streaming import MiniMaxM2StreamingHandler

            return MiniMaxM2StreamingHandler(
                streaming_response=self.streaming_response,
                sync_stream=self.sync_stream,
                json_mode=self.json_mode,
            )

        # Default: use the parent's chunk_parser
        self._use_default_handler = True
        return self