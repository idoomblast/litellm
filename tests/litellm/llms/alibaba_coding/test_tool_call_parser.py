from litellm.llms.alibaba_coding.chat.tool_call_parser import (
    TOOL_CALL_ARGUMENT_BEGIN,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    TOOL_CALLS_SECTION_BEGIN,
    TOOL_CALLS_SECTION_END,
    is_native_tool_call_model,
    parse_tool_calls_from_content,
    strip_native_tool_tokens,
)


def test_is_native_tool_call_model():
    assert is_native_tool_call_model("moonshotai/Kimi-K2") is True
    assert is_native_tool_call_model("KIMI-k2-instruct") is True
    assert is_native_tool_call_model("kimi-1") is False
    assert is_native_tool_call_model("gpt-4o") is False


def test_parse_tool_calls_from_content_with_tokens():
    content = (
        "Before tool call\n"
        f"{TOOL_CALLS_SECTION_BEGIN}\n"
        f"{TOOL_CALL_BEGIN}functions.get_weather:0{TOOL_CALL_ARGUMENT_BEGIN}"
        f'{{"city":"Tokyo"}}{TOOL_CALL_END}\n'
        f"{TOOL_CALLS_SECTION_END}\n"
        "After tool call"
    )

    tool_calls, cleaned = parse_tool_calls_from_content(content)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "functions.get_weather:0"
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == '{"city":"Tokyo"}'
    assert "Before tool call" in cleaned
    assert "After tool call" in cleaned
    assert TOOL_CALL_BEGIN not in cleaned


def test_strip_native_tool_tokens_cleans_content():
    content = (
        "Result starts. "
        f"{TOOL_CALLS_SECTION_BEGIN}"
        f"{TOOL_CALL_BEGIN}functions.lookup:1{TOOL_CALL_ARGUMENT_BEGIN}{{\"id\":123}}{TOOL_CALL_END}"
        f"{TOOL_CALLS_SECTION_END}"
        " Result ends."
    )

    cleaned = strip_native_tool_tokens(content)

    assert cleaned == "Result starts.  Result ends."
    assert TOOL_CALLS_SECTION_BEGIN not in cleaned
    assert TOOL_CALL_BEGIN not in cleaned
