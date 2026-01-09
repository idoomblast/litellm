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
        """Test that thinking parameter with enabled type maps to chat_template_kwargs."""
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

        # Verify the new mapping to chat_template_kwargs
        assert "chat_template_kwargs" in result
        assert result["chat_template_kwargs"]["enable_thinking"] is True

    def test_thinking_parameter_disabled(self):
        """Test that thinking parameter with disabled type maps to chat_template_kwargs."""
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

        # Verify the new mapping to chat_template_kwargs
        assert "chat_template_kwargs" in result
        assert result["chat_template_kwargs"]["enable_thinking"] is False

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

    def test_reasoning_effort_low_medium_high(self):
        """Test that reasoning_effort values 'low', 'medium', 'high' map to enable_thinking=True."""
        for effort in ["low", "medium", "high"]:
            non_default_params = {
                "reasoning_effort": effort,
            }
            optional_params = {}

            result = self.config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=self.model,
                drop_params=False,
            )

            assert "chat_template_kwargs" in result
            assert result["chat_template_kwargs"]["enable_thinking"] is True

    def test_reasoning_effort_none_minimal(self):
        """Test that reasoning_effort values 'none', 'minimal' map to enable_thinking=False."""
        for effort in ["none", "minimal"]:
            non_default_params = {
                "reasoning_effort": effort,
            }
            optional_params = {}

            result = self.config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=self.model,
                drop_params=False,
            )

            assert "chat_template_kwargs" in result
            assert result["chat_template_kwargs"]["enable_thinking"] is False

    def test_thinking_with_existing_chat_template_kwargs(self):
        """Test that thinking parameter properly merges with existing chat_template_kwargs."""
        non_default_params = {
            "thinking": {"type": "enabled"},
        }
        optional_params = {
            "chat_template_kwargs": {"existing_param": "value"},
        }

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should merge, not overwrite
        assert "chat_template_kwargs" in result
        assert result["chat_template_kwargs"]["enable_thinking"] is True
        assert result["chat_template_kwargs"]["existing_param"] == "value"

    def test_no_chat_template_kwargs_when_no_thinking(self):
        """Test that no chat_template_kwargs is created when no thinking/reasoning_effort provided."""
        non_default_params = {
            "temperature": 0.5,
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should not have chat_template_kwargs when no thinking params
        assert "chat_template_kwargs" not in result

    def test_thinking_direct_boolean(self):
        """Test that direct boolean thinking values work correctly."""
        # Test thinking=True
        non_default_params_true = {
            "thinking": True,
        }
        result_true = self.config.map_openai_params(
            non_default_params=non_default_params_true,
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result_true["chat_template_kwargs"]["enable_thinking"] is True

        # Test thinking=False
        non_default_params_false = {
            "thinking": False,
        }
        result_false = self.config.map_openai_params(
            non_default_params=non_default_params_false,
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result_false["chat_template_kwargs"]["enable_thinking"] is False