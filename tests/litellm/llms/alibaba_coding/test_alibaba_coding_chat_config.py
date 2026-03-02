from litellm.llms.alibaba_coding.chat.transformation import AlibabaCodingChatConfig


def test_alibaba_coding_chat_config_instantiation():
    config = AlibabaCodingChatConfig()
    assert config is not None


def test_get_openai_compatible_provider_info_defaults(monkeypatch):
    monkeypatch.delenv("ALIBABA_CODING_API_BASE", raising=False)
    monkeypatch.delenv("ALIBABA_CODING_API_KEY", raising=False)

    config = AlibabaCodingChatConfig()
    api_base, api_key = config._get_openai_compatible_provider_info(
        api_base=None,
        api_key=None,
    )

    assert api_base == "https://coding-intl.dashscope.aliyuncs.com/v1"
    assert api_key == ""


def test_get_openai_compatible_provider_info_custom_values():
    config = AlibabaCodingChatConfig()
    api_base, api_key = config._get_openai_compatible_provider_info(
        api_base="https://custom.alibaba.example/v1",
        api_key="test-api-key",
    )

    assert api_base == "https://custom.alibaba.example/v1"
    assert api_key == "test-api-key"


def test_get_supported_openai_params_includes_thinking_reasoning_effort():
    config = AlibabaCodingChatConfig()
    params = config.get_supported_openai_params("alibaba_coding/model")

    assert "thinking" in params
    assert "reasoning_effort" in params


def test_map_openai_params_thinking_reasoning_effort():
    config = AlibabaCodingChatConfig()

    test_cases = [
        ({"thinking": True}, True),
        ({"thinking": {"type": "enabled"}}, True),
        ({"thinking": {"type": "disabled"}}, False),
        ({"reasoning_effort": "high"}, True),
        ({"reasoning_effort": "minimal"}, False),
        ({"thinking": False, "reasoning_effort": "high"}, False),
    ]

    for non_default_params, expected_enable_thinking in test_cases:
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="alibaba_coding/model",
            drop_params=False,
        )

        assert "chat_template_kwargs" in result
        assert (
            result["chat_template_kwargs"]["enable_thinking"]
            == expected_enable_thinking
        )
