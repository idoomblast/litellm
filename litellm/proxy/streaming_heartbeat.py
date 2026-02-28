"""
SSE Heartbeat wrapper for streaming responses.

Prevents reverse proxy timeouts (e.g., Cloudflare 524) when TTFB
from upstream AI providers exceeds the proxy's read timeout.

Usage:
    Set environment variable PROXY_SSE_HEARTBEAT_INTERVAL to the interval
    in seconds between heartbeat SSE comments (e.g., "30" for 30s).
    Default is "0" (disabled).

    Cloudflare free/pro/business plan has a 100s proxy read timeout,
    so 30s is a safe heartbeat interval.

How it works:
    SSE comments (lines starting with ':') are valid per the SSE specification
    and are silently ignored by all standard SSE clients (OpenAI SDK, browser
    EventSource, etc.), but they count as data transfer for reverse proxy
    timeout purposes — resetting Cloudflare/nginx read timeout timers.

Reference:
    - SSE spec: https://html.spec.whatwg.org/multipage/server-sent-events.html
    - Cloudflare 524: https://developers.cloudflare.com/support/troubleshooting/cloudflare-errors/troubleshooting-cloudflare-5xx-errors/#error-524-a-timeout-occurred
"""

import asyncio
import contextlib
import os
from typing import AsyncIterator, Union

from litellm._logging import verbose_proxy_logger

# Interval in seconds between heartbeat SSE comments. 0 = disabled.
# Cloudflare free/pro/business has 100s proxy read timeout,
# so 30s is a safe interval to keep the connection alive.
PROXY_SSE_HEARTBEAT_INTERVAL = float(
    os.getenv("PROXY_SSE_HEARTBEAT_INTERVAL", "0")
)

# SSE comment — valid per spec, ignored by clients, resets proxy timers.
SSE_HEARTBEAT_COMMENT = ": heartbeat\n\n"


def maybe_wrap_with_heartbeat(
    generator: AsyncIterator[str],
) -> AsyncIterator[str]:
    """
    Convenience wrapper: wraps a generator with heartbeat if enabled via env var.

    Returns the original generator unchanged if heartbeat is disabled (interval=0).
    """
    if PROXY_SSE_HEARTBEAT_INTERVAL > 0:
        return sse_heartbeat_generator(
            inner=generator,
            heartbeat_interval=PROXY_SSE_HEARTBEAT_INTERVAL,
        )
    return generator


async def sse_heartbeat_generator(
    inner: AsyncIterator[str],
    heartbeat_interval: float = 0,
) -> AsyncIterator[str]:
    """
    Wraps an SSE async generator to inject periodic heartbeat comments.

    Uses async queue + background reader task to safely inject heartbeats
    without interfering with the inner generator's iteration state.

    Args:
        inner: The original SSE async generator.
        heartbeat_interval: Seconds between heartbeats. 0 = passthrough (no heartbeat).

    Yields:
        Original SSE chunks interleaved with `: heartbeat` comments
        when no data arrives within ``heartbeat_interval`` seconds.
    """
    if heartbeat_interval <= 0:
        async for chunk in inner:
            yield chunk
        return

    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    async def _reader() -> None:
        """Read from inner generator and push to queue."""
        try:
            async for chunk in inner:
                await queue.put(chunk)
        except Exception as exc:
            await queue.put(exc)
        finally:
            await queue.put(_DONE)

    task = asyncio.create_task(_reader())
    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    queue.get(), timeout=heartbeat_interval
                )
            except asyncio.TimeoutError:
                verbose_proxy_logger.debug(
                    "SSE heartbeat: sending keep-alive comment "
                    "(no chunk received in %.1fs)",
                    heartbeat_interval,
                )
                yield SSE_HEARTBEAT_COMMENT
                continue

            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
