"""
Test context caching minimum token validation.

This file tests the validation logic that ensures cached content meets minimum token requirements.
"""
import pytest
from unittest.mock import MagicMock, patch
from typing import List

import sys
import os
sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching import (
    ContextCachingEndpoints,
)
from litellm.llms.vertex_ai.context_caching.transformation import (
    get_context_cache_min_tokens,
    estimate_message_tokens,
)
from litellm.types.llms.openai import AllMessageValues


class TestContextCachingMinTokens:
    """Test minimum token validation for context caching"""

    def setup_method(self):
        """Setup for each test method"""
        self.context_caching = ContextCachingEndpoints()
        self.mock_logging = MagicMock(spec=Logging)
        self.mock_client = MagicMock(spec=HTTPHandler)

        # Sample cached messages with short text (should fail validation)
        self.short_cached_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Hi",  # Very short
                        "cache_control": {"type": "ephemeral", "ttl": "3600s"},
                    }
                ],
            },  # type: ignore[assignment]  # Test data, simplified from AllMessageValues
        ]

        # Sample cached messages with long text (should pass validation)
        self.long_cached_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "This is a test message. " * 200,  # ~3600 tokens estimate
                        "cache_control": {"type": "ephemeral", "ttl": "3600s"},
                    }
                ],
            },  # type: ignore[assignment]  # Test data, simplified from AllMessageValues
        ]

        self.non_cached_messages = [
            {"role": "user", "content": "Hello, how are you?"}  # type: ignore[assignment]  # Test data
        ]

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj"
    )
    @patch.object(ContextCachingEndpoints, "check_cache")
    def test_check_and_create_cache_insufficient_tokens_auto_disables(
        self, mock_check_cache, mock_cache_obj, mock_separate, custom_llm_provider
    ):
        """Test that check_and_create_cache auto-disables caching when cached content has too few tokens"""
        # Setup
        mock_separate.return_value = (self.short_cached_messages, self.non_cached_messages)
        mock_cache_obj.get_cache_key.return_value = "test_cache_key"
        mock_check_cache.return_value = None  # No existing cache

        test_project = "test_project"
        test_location = "test_location"

        # Execute - should NOT raise VertexAIError, but auto-disable caching
        result = self.context_caching.check_and_create_cache(
            messages=self.short_cached_messages + self.non_cached_messages,  # type: ignore[arg-type]
            optional_params={},
            api_key="test_key",
            api_base=None,
            model="gemini-2.5-flash",  # Requires 1024 minimum
            client=self.mock_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
            custom_llm_provider=custom_llm_provider,
            vertex_project=test_project,
            vertex_location=test_location,
            vertex_auth_header="vertext_test_token",
        )

        # Verify result contains messages without cache_control
        messages, returned_params, returned_cache = result
        assert returned_cache is None  # No cache created
        
        # Verify cache_control has been removed from all messages
        from litellm.llms.vertex_ai.context_caching.transformation import remove_cache_control_from_messages
        expected_messages = remove_cache_control_from_messages(self.short_cached_messages + self.non_cached_messages)  # type: ignore[arg-type]
        
        # Messages should match those with cache_control removed
        assert messages == expected_messages

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj"
    )
    @patch.object(ContextCachingEndpoints, "check_cache")
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.transform_openai_messages_to_gemini_context_caching"
    )
    def test_check_and_create_cache_sufficient_tokens_proceeds(
        self, mock_transform, mock_check_cache, mock_cache_obj, mock_separate, custom_llm_provider
    ):
        """Test that check_and_create_cache proceeds normally when cached content has enough tokens"""
        # Setup
        mock_separate.return_value = (self.long_cached_messages, self.non_cached_messages)
        mock_cache_obj.get_cache_key.return_value = "test_cache_key"
        mock_check_cache.return_value = None  # No existing cache
        mock_transform.return_value = {
            "contents": self.long_cached_messages,
            "model": "models/gemini-2.5-flash",
            "displayName": "test_cache_key",
        }
        self.mock_client.post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"name": "new_cache_id", "model": "models/gemini-2.5-flash"})
        )

        test_project = "test_project"
        test_location = "test_location"

        # Execute - should proceed without error
        try:
            result = self.context_caching.check_and_create_cache(
                messages=self.long_cached_messages + self.non_cached_messages,  # type: ignore[arg-type]
                optional_params={},
                api_key="test_key",
                api_base=None,
                model="gemini-2.5-flash",
                client=self.mock_client,
                timeout=30.0,
                logging_obj=self.mock_logging,
                custom_llm_provider=custom_llm_provider,
                vertex_project=test_project,
                vertex_location=test_location,
                vertex_auth_header="vertext_test_token",
            )

            # Assert - should return the result
            messages, returned_params, returned_cache = result
            assert messages == self.non_cached_messages
            assert returned_cache == "new_cache_id"

            # Verify transform was called (validation passed)
            mock_transform.assert_called_once()
        except VertexAIError as e:
            pytest.fail(f"Should not raise error with sufficient tokens: {e}")

    @pytest.mark.parametrize(
        "gemini_model, min_tokens",
        [
            ("gemini-2.5-flash", 1024),
            ("gemini-2.5-pro", 2048),
            ("gemini-2.5-flash-lite", 512),
            ("gemini-1.5-pro", 32768),
        ],
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj"
    )
    @patch.object(ContextCachingEndpoints, "check_cache")
    def test_gemini_provider_minimum_tokens_auto_disable(
        self,
        mock_check_cache,
        mock_cache_obj,
        mock_separate,
        gemini_model,
        min_tokens,
    ):
        """
        Test that gemini provider (Google AI Studio) auto-disables caching
        when cached content has too few tokens.
        This ensures the auto-disable feature works for both providers since they share the same
        ContextCachingEndpoints implementation.
        """
        # Setup
        mock_separate.return_value = (self.short_cached_messages, self.non_cached_messages)
        mock_cache_obj.get_cache_key.return_value = "test_cache_key"
        mock_check_cache.return_value = None  # No existing cache

        # Execute - should NOT raise VertexAIError, but auto-disable caching
        result = self.context_caching.check_and_create_cache(
            messages=self.short_cached_messages + self.non_cached_messages,  # type: ignore[arg-type]
            optional_params={},
            api_key="test_key",
            api_base=None,
            model=gemini_model,
            client=self.mock_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
            custom_llm_provider="gemini",  # Explicitly use gemini provider
            vertex_project=None,
            vertex_location=None,
            vertex_auth_header=None,
        )

        # Verify result contains messages without cache_control and no cache created
        messages, returned_params, returned_cache = result
        assert returned_cache is None  # No cache created
        
        # Verify cache_control has been removed from all messages
        from litellm.llms.vertex_ai.context_caching.transformation import remove_cache_control_from_messages
        expected_messages = remove_cache_control_from_messages(self.short_cached_messages + self.non_cached_messages)  # type: ignore[arg-type]
        
        # Messages should match those with cache_control removed
        assert messages == expected_messages


class TestContextCacheTransformHelpers:
    """
    Tests for helper functions in context caching transformation module.
    These tests verify that the minimum token validation infrastructure 
    works correctly.
    """

    @pytest.mark.parametrize(
        "model, expected_min",
        [
            ("gemini-1.5-pro", 32768),
            ("gemini-2.5-flash", 1024),
            ("gemini-2.5-flash-lite", 512),
            ("gemini-2.5-pro", 2048),
            ("vertex_ai/gemini-2.5-flash", 1024),
            ("vertex_ai/gemini-2.5-pro", 2048),
        ],
    )
    def test_get_context_cache_min_tokens(self, model, expected_min):
        """
        Test that get_context_cache_min_tokens returns correct minimum
        token count for different models.
        """
        min_tokens = get_context_cache_min_tokens(model)
        assert min_tokens == expected_min, f"Expected {expected_min} for {model}, got {min_tokens}"

    def test_estimate_message_tokens_strings(self):
        """
        Test estimate_message_tokens with simple string content.
        Each char ≈ 0.25 tokens. 100 chars = 25 tokens.
        """
        messages = [
            {"role": "user", "content": "A" * 100},  # ~25 tokens
        ]  # type: ignore[list-item]
        estimated = estimate_message_tokens(messages)  # type: ignore[arg-type]
        # Should be approximately 25 tokens (allow some margin for overhead)
        assert 20 <= estimated <= 30, f"Expected ~25 tokens, got {estimated}"

    def test_estimate_message_tokens_nested_content(self):
        """
        Test estimate_message_tokens with nested content structure.
        """
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "B" * 100}],  # ~25 tokens
            }
        ]  # type: ignore[list-item]
        estimated = estimate_message_tokens(messages)  # type: ignore[arg-type]
        # Should be approximately 25 tokens + nesting overhead
        assert 20 <= estimated <= 35, f"Expected ~25-30 tokens, got {estimated}"

    def test_estimate_message_tokens_with_images(self):
        """
        Test estimate_message_tokens with image content.
        Images are counted as 800 chars (roughly 200 tokens) each regardless of content length.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Short text"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                ],
            }
        ]  # type: ignore[list-item]
        estimated = estimate_message_tokens(messages)  # type: ignore[arg-type]
        # Image = 800 chars / 4 = ~200 tokens
        # Text = ~3 tokens
        # Overhead = 4 tokens
        # Total = ~207 tokens
        assert 190 <= estimated <= 220, f"Expected ~207 tokens for image+text, got {estimated}"

    def test_estimate_message_tokens_empty_content(self):
        """
        Test estimate_message_tokens with empty messages list.
        Should return 0 tokens.
        """
        messages: List[AllMessageValues] = []
        estimated = estimate_message_tokens(messages)  # type: ignore[arg-type]
        assert estimated == 0, f"Expected 0 tokens for empty list, got {estimated}"
