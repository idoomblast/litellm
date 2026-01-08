"""
Unit tests for Chutes chat transformation.

Tests the configuration and parameter handling for Chutes models.
"""

from typing import List, cast

import pytest
from litellm.llms.chutes.chat.transformation import ChutesChatConfig
from litellm.types.llms.openai import AllMessageValues


class TestChutesChatConfig:
    """Test Chutes chat configuration."""

    def setup_method(self):
        self.config = ChutesChatConfig()
        self.model = "chutes/model-name"

    def test_get_supported_openai_params(self):
        """Test that standard OpenAI params are supported including thinking."""
        params = self.config.get_supported_openai_params(self.model)
        # Chutes supports standard OpenAI optional params
        assert "temperature" in params
        assert "max_tokens" in params
        assert "top_p" in params
        assert "stream" in params
        # Chutes supports thinking parameter
        assert "thinking" in params

    def test_thinking_parameter_enabled(self):
        """Test that thinking parameter with enabled type is passed through."""
        non_default_params = {
            "thinking": {"type": "enabled"},
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_thinking_parameter_disabled(self):
        """Test that thinking parameter with disabled type is passed through."""
        non_default_params = {
            "thinking": {"type": "disabled"},
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "disabled"}

    def test_map_openai_params_passes_through(self):
        """Test that standard params are passed through correctly."""
        non_default_params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1000
        assert result["top_p"] == 0.9

    def test_transform_messages_passes_through(self):
        """Test that messages are passed through without transformation."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, how can I help?"},
        ]

        # Cast messages to AllMessageValues type for type checking
        from litellm.types.llms.openai import AllMessageValues
        from typing import cast
        typed_messages = cast(List[AllMessageValues], messages)

        result = self.config._transform_messages(
            messages=typed_messages, model=self.model, is_async=False
        )

        # Messages should be unchanged (OpenAI format)
        assert result[0]["role"] == "user"
        assert result[0].get("content") == "Hello, how are you?"
        assert result[1]["role"] == "assistant"
        assert result[1].get("content") == "I'm doing well, how can I help?"

    def test_get_openai_compatible_provider_info_default_api_base(self):
        """Test that default API base is correctly set."""
        api_base, api_key = self.config._get_openai_compatible_provider_info(
            api_base=None, api_key=None
        )

        assert api_base == "https://llm.chutes.ai/v1"
        assert api_key == ""

    def test_get_openai_compatible_provider_info_custom_api_base(self):
        """Test that custom API base is used when provided."""
        custom_base = "https://custom.chutes.ai/v1"
        api_base, api_key = self.config._get_openai_compatible_provider_info(
            api_base=custom_base, api_key="test-key"
        )

        assert api_base == custom_base
        assert api_key == "test-key"

    def test_get_openai_compatible_provider_info_custom_api_key(self):
        """Test that custom API key is used when provided."""
        custom_key = "my-custom-api-key"
        api_base, api_key = self.config._get_openai_compatible_provider_info(
            api_base=None, api_key=custom_key
        )

        assert api_base == "https://llm.chutes.ai/v1"
        assert api_key == custom_key