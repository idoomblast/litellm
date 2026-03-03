"""
Streaming handler for Alibaba Coding provider.

Reuses the Chutes streaming handler since Alibaba Coding uses the same
Kimi K2 native tool call format for tool calling and <think> tag handling.
"""

from typing import Any, Optional

from litellm.llms.chutes.chat.streaming_handler import (
    ChutesChatCompletionStreamingHandler,
)


class AlibabaCodingStreamingHandler(ChutesChatCompletionStreamingHandler):
    """
    Streaming handler for Alibaba Coding provider.

    Inherits all Kimi K2 tool call parsing, think tag handling, and
    duplicate tool call prevention from ChutesChatCompletionStreamingHandler.
    """

    pass
