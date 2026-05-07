"""
Unit tests for SSE heartbeat wrapper.

Tests the streaming_heartbeat module that prevents Cloudflare 524 timeouts
by sending SSE comments between chunks when TTFB is too long.
"""

import asyncio
import os
import time
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_heartbeat_interval():
    """Reset env var before/after each test."""
    old = os.environ.pop("PROXY_SSE_HEARTBEAT_INTERVAL", None)
    yield
    if old is not None:
        os.environ["PROXY_SSE_HEARTBEAT_INTERVAL"] = old
    else:
        os.environ.pop("PROXY_SSE_HEARTBEAT_INTERVAL", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _slow_generator(chunks, delay_before_first=0, delay_between=0):
    """Simulate an upstream SSE generator with configurable delays."""
    if delay_before_first > 0:
        await asyncio.sleep(delay_before_first)
    for i, chunk in enumerate(chunks):
        yield chunk
        if delay_between > 0 and i < len(chunks) - 1:
            await asyncio.sleep(delay_between)


async def _collect(gen):
    """Collect all items from an async generator into a list."""
    result = []
    async for item in gen:
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSSEHeartbeatGenerator:
    """Tests for sse_heartbeat_generator."""

    @pytest.mark.asyncio
    async def test_should_passthrough_when_interval_is_zero(self):
        """When heartbeat_interval=0, chunks pass through unchanged."""
        from litellm.proxy.streaming_heartbeat import sse_heartbeat_generator

        chunks = ["data: chunk1\n\n", "data: chunk2\n\n", "data: [DONE]\n\n"]
        gen = sse_heartbeat_generator(
            inner=_slow_generator(chunks),
            heartbeat_interval=0,
        )
        result = await _collect(gen)
        assert result == chunks

    @pytest.mark.asyncio
    async def test_should_passthrough_when_interval_is_negative(self):
        """Negative interval should behave same as disabled."""
        from litellm.proxy.streaming_heartbeat import sse_heartbeat_generator

        chunks = ["data: chunk1\n\n"]
        gen = sse_heartbeat_generator(
            inner=_slow_generator(chunks),
            heartbeat_interval=-1,
        )
        result = await _collect(gen)
        assert result == chunks

    @pytest.mark.asyncio
    async def test_should_pass_chunks_without_heartbeat_when_fast(self):
        """When chunks arrive faster than heartbeat interval, no heartbeats injected."""
        from litellm.proxy.streaming_heartbeat import sse_heartbeat_generator

        chunks = ["data: a\n\n", "data: b\n\n", "data: c\n\n"]
        gen = sse_heartbeat_generator(
            inner=_slow_generator(chunks, delay_between=0.01),
            heartbeat_interval=10,  # much longer than delay
        )
        result = await _collect(gen)
        assert result == chunks

    @pytest.mark.asyncio
    async def test_should_inject_heartbeat_when_ttfb_exceeds_interval(self):
        """When first chunk takes longer than interval, heartbeat(s) should be injected."""
        from litellm.proxy.streaming_heartbeat import (
            SSE_HEARTBEAT_COMMENT,
            sse_heartbeat_generator,
        )

        chunks = ["data: chunk1\n\n", "data: [DONE]\n\n"]
        # First chunk delayed 0.35s, heartbeat every 0.1s → expect ~3 heartbeats before first chunk
        gen = sse_heartbeat_generator(
            inner=_slow_generator(chunks, delay_before_first=0.35),
            heartbeat_interval=0.1,
        )
        result = await _collect(gen)

        # Should have heartbeats before first real chunk
        heartbeats = [c for c in result if c == SSE_HEARTBEAT_COMMENT]
        real_chunks = [c for c in result if c != SSE_HEARTBEAT_COMMENT]

        assert len(heartbeats) >= 2, f"Expected >=2 heartbeats, got {len(heartbeats)}"
        assert real_chunks == chunks
        # First items should be heartbeats, then real chunks
        assert result[0] == SSE_HEARTBEAT_COMMENT

    @pytest.mark.asyncio
    async def test_should_inject_heartbeat_between_slow_chunks(self):
        """When gap between chunks exceeds interval, heartbeat injected between them."""
        from litellm.proxy.streaming_heartbeat import (
            SSE_HEARTBEAT_COMMENT,
            sse_heartbeat_generator,
        )

        chunks = ["data: a\n\n", "data: b\n\n"]
        gen = sse_heartbeat_generator(
            inner=_slow_generator(chunks, delay_between=0.25),
            heartbeat_interval=0.1,
        )
        result = await _collect(gen)

        heartbeats = [c for c in result if c == SSE_HEARTBEAT_COMMENT]
        real_chunks = [c for c in result if c != SSE_HEARTBEAT_COMMENT]

        assert len(heartbeats) >= 1
        assert real_chunks == chunks

    @pytest.mark.asyncio
    async def test_should_handle_empty_generator(self):
        """Empty inner generator should produce no output."""
        from litellm.proxy.streaming_heartbeat import sse_heartbeat_generator

        async def empty_gen():
            return
            yield  # make it a generator  # noqa: E501

        gen = sse_heartbeat_generator(inner=empty_gen(), heartbeat_interval=0.1)
        result = await _collect(gen)
        assert result == []

    @pytest.mark.asyncio
    async def test_should_propagate_exception_from_inner(self):
        """Exceptions from inner generator should propagate."""
        from litellm.proxy.streaming_heartbeat import sse_heartbeat_generator

        async def error_gen():
            yield "data: ok\n\n"
            raise ValueError("upstream error")

        gen = sse_heartbeat_generator(inner=error_gen(), heartbeat_interval=1)
        with pytest.raises(ValueError, match="upstream error"):
            await _collect(gen)

    @pytest.mark.asyncio
    async def test_should_complete_in_reasonable_time(self):
        """Heartbeat generator should not add significant overhead."""
        from litellm.proxy.streaming_heartbeat import sse_heartbeat_generator

        chunks = [f"data: chunk{i}\n\n" for i in range(100)]
        gen = sse_heartbeat_generator(
            inner=_slow_generator(chunks),
            heartbeat_interval=10,
        )
        start = time.monotonic()
        result = await _collect(gen)
        elapsed = time.monotonic() - start

        assert len(result) == 100
        assert elapsed < 2.0, f"Took {elapsed:.2f}s — too slow"


class TestMaybeWrapWithHeartbeat:
    """Tests for maybe_wrap_with_heartbeat helper."""

    @pytest.mark.asyncio
    async def test_should_return_original_when_disabled(self):
        """When PROXY_SSE_HEARTBEAT_INTERVAL=0, returns the same generator."""
        os.environ["PROXY_SSE_HEARTBEAT_INTERVAL"] = "0"
        # Re-import to pick up env var
        import importlib
        import litellm.proxy.streaming_heartbeat as mod
        importlib.reload(mod)

        chunks = ["data: a\n\n"]
        original = _slow_generator(chunks)
        wrapped = mod.maybe_wrap_with_heartbeat(original)
        # When disabled, should return the exact same generator object
        # (passthrough — no wrapping overhead)
        assert wrapped is original

    @pytest.mark.asyncio
    async def test_should_wrap_when_enabled(self):
        """When PROXY_SSE_HEARTBEAT_INTERVAL>0, returns a different generator."""
        os.environ["PROXY_SSE_HEARTBEAT_INTERVAL"] = "30"
        import importlib
        import litellm.proxy.streaming_heartbeat as mod
        importlib.reload(mod)

        chunks = ["data: a\n\n"]
        original = _slow_generator(chunks)
        wrapped = mod.maybe_wrap_with_heartbeat(original)
        # Should NOT be the same object — it's been wrapped
        assert wrapped is not original

        # But should still yield the same chunks
        result = await _collect(wrapped)
        assert result == chunks


class TestHeartbeatComment:
    """Tests for SSE comment format validity."""

    def test_should_be_valid_sse_comment(self):
        """Heartbeat should start with ':' per SSE spec."""
        from litellm.proxy.streaming_heartbeat import SSE_HEARTBEAT_COMMENT

        assert SSE_HEARTBEAT_COMMENT.startswith(":")
        assert SSE_HEARTBEAT_COMMENT.endswith("\n\n")

    def test_should_not_contain_data_prefix(self):
        """Heartbeat should NOT be a data event — only a comment."""
        from litellm.proxy.streaming_heartbeat import SSE_HEARTBEAT_COMMENT

        assert not SSE_HEARTBEAT_COMMENT.startswith("data:")
