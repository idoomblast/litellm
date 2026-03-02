from litellm.llms.alibaba_coding.chat.transformation import (
    DEFAULT_USER_AGENT,
    AlibabaCodingChatConfig,
)


def test_default_user_agent_constant_value():
    assert DEFAULT_USER_AGENT == "opencode/1.2.6 ai-sdk/provider-utils/3.0.21 runtime/bun/1.3.9"


def test_user_agent_override_mechanism():
    config = AlibabaCodingChatConfig()

    headers = config.validate_environment(
        headers={},
        model="alibaba_coding/model",
        messages=[],
        optional_params={"user_agent": "optional-param-agent"},
        litellm_params={"user_agent": "litellm-agent"},
        api_key="test-key",
        api_base=None,
    )
    assert headers["User-Agent"] == "litellm-agent"

    existing_headers = config.validate_environment(
        headers={"User-Agent": "already-set-header"},
        model="alibaba_coding/model",
        messages=[],
        optional_params={"user_agent": "ignored-optional-agent"},
        litellm_params={"user_agent": "ignored-litellm-agent"},
        api_key="test-key",
        api_base=None,
    )
    assert existing_headers["User-Agent"] == "already-set-header"
