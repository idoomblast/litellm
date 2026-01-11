"""
Unit tests for Xiaomi MiMo chat transformation.

Tests the thinking and reasoning_effort parameter handling for Xiaomi MiMo models.
"""

import pytest
from litellm.llms.xiaomi_mimo.chat.transformation import XiaomiMiMoChatConfig


class TestXiaomiMiMoThinkingParams:
    """Test thinking and reasoning_effort parameter handling for Xiaomi MiMo."""

    def setup_method(self):
        self.config = XiaomiMiMoChatConfig()
        self.model = "xiaomi_mimo/mimo-v2-flash"

    def test_get_supported_openai_params_includes_thinking(self):
        """Test that thinking and reasoning_effort are in supported params."""
        params = self.config.get_supported_openai_params(self.model)
        assert "thinking" in params
        assert "reasoning_effort" in params

    def test_map_thinking_enabled(self):
        """Test that thinking={"type": "enabled"} is passed through correctly."""
        non_default_params = {"thinking": {"type": "enabled"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_thinking_disabled(self):
        """Test that thinking={"type": "disabled"} is passed through correctly."""
        non_default_params = {"thinking": {"type": "disabled"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "disabled"}

    def test_map_reasoning_effort_none_to_disabled(self):
        """Test that reasoning_effort='none' maps to thinking disabled."""
        non_default_params = {"reasoning_effort": "none"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "disabled"}

    def test_map_reasoning_effort_minimal_to_disabled(self):
        """Test that reasoning_effort='minimal' maps to thinking disabled."""
        non_default_params = {"reasoning_effort": "minimal"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "disabled"}

    def test_map_reasoning_effort_low_to_enabled(self):
        """Test that reasoning_effort='low' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "low"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_medium_to_enabled(self):
        """Test that reasoning_effort='medium' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "medium"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_high_to_enabled(self):
        """Test that reasoning_effort='high' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_thinking_takes_precedence_over_reasoning_effort(self):
        """Test that explicit thinking parameter takes precedence over reasoning_effort."""
        non_default_params = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "none"  # This should be ignored
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should use thinking parameter, ignore reasoning_effort
        assert result["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in result

    def test_map_no_thinking_or_reasoning_effort(self):
        """Test that no thinking or reasoning_effort params result in no thinking param."""
        non_default_params = {}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result
        assert "reasoning_effort" not in result