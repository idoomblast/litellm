import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Chutes

## Overview

| Property | Details |
|-------|-------|
| Description | Chutes is a cloud-native AI deployment platform that allows you to deploy, run, and scale LLM applications with OpenAI-compatible APIs using pre-built templates for popular frameworks like vLLM and SGLang. |
| Provider Route on LiteLLM | `chutes/` |
| Link to Provider Doc | [Chutes Website ↗](https://chutes.ai) |
| Base URL | `https://llm.chutes.ai/v1/` |
| Supported Operations | [`/chat/completions`](#usage---litellm-python-sdk), Embeddings |

<br />

## What is Chutes?

Chutes is a powerful AI deployment and serving platform that provides:
- **Pre-built Templates**: Ready-to-use configurations for vLLM, SGLang, diffusion models, and embeddings
- **OpenAI-Compatible APIs**: Use standard OpenAI SDKs and clients
- **Multi-GPU Scaling**: Support for large models across multiple GPUs
- **Streaming Responses**: Real-time model outputs
- **Custom Configurations**: Override any parameter for your specific needs
- **Performance Optimization**: Pre-configured optimization settings

## Supported Models

Chutes provides access to various LLM models through its platform:

### Reasoning Models
These models support extended reasoning capabilities via the `thinking` parameter:

| Model | Input Tokens | Output Tokens | Input Cost | Output Cost |
|-------|--------------|---------------|------------|-------------|
| `chutes/MiniMaxAI/MiniMax-M2.1-TEE` | 202,752 | 65,535 | $0.40M | $1.50M |
| `chutes/moonshotai/Kimi-K2-Thinking-TEE` | 262,144 | 65,535 | $0.40M | $1.75M |
| `chutes/zai-org/GLM-4.6-TEE` | 202,752 | 65,536 | $0.35M | $1.50M |
| `chutes/zai-org/GLM-4.7-TEE` | 202,752 | 65,535 | $0.40M | $1.50M |
| `chutes/XiaomiMiMo/MiMo-V2-Flash` | 262,144 | 65,536 | $0.40M | $1.50M |
| `chutes/deepseek-ai/DeepSeek-V3.1-TEE` | 163,840 | 65,536 | $0.20M | $0.80M |
| `chutes/deepseek-ai/DeepSeek-V3.2-TEE` | 163,840 | 65,536 | $0.25M | $0.38M |
| `chutes/Qwen/Qwen3-32B` | 40,960 | 40,960 | $0.08M | $0.24M |
| `chutes/Qwen/Qwen3-14B` | 40,960 | 40,960 | $0.05M | $0.22M |

### Standard Models
These models do not support extended reasoning:

| Model | Input Tokens | Output Tokens | Input Cost | Output Cost |
|-------|--------------|---------------|------------|-------------|
| `chutes/moonshotai/Kimi-K2-Instruct-0905` | 262,144 | 262,144 | $0.39M | $1.90M |
| `chutes/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE` | 262,144 | 262,144 | $0.22M | $0.95M |
| `chutes/Qwen/Qwen2.5-Coder-32B-Instruct` | 32,768 | 32,768 | $0.03M | $0.11M |

*Note: Costs shown per 1M tokens. M = Million*

All models support:
- ✅ **Function Calling** & **Parallel Function Calling**
- ✅ **Response Schema** (JSON mode)
- ❌ **Vision** (currently not supported)
- ❌ **Web Search** (currently not supported)
- ❌ **Prompt Caching** (currently not supported)

## Required Variables

```python showLineNumbers title="Environment Variables"
import os

os.environ["CHUTES_API_KEY"] = ""  # your Chutes API key
# Optional: Custom API base URLs
# os.environ["CHUTES_API_BASE"] = "https://llm.chutes.ai/v1"
```

Get your Chutes API key from [chutes.ai](https://chutes.ai).

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Chutes Non-streaming Completion"
import os
from litellm import completion

os.environ["CHUTES_API_KEY"] = "your-chutes-api-key"

messages = [{"content": "What is the capital of France?", "role": "user"}]

response = completion(
    model="chutes/deepseek-ai/DeepSeek-V3.2-TEE",
    messages=messages
)

print(response.choices[0].message.content)
```

### Streaming

```python showLineNumbers title="Chutes Streaming Completion"
import os
from litellm import completion

os.environ["CHUTES_API_KEY"] = "your-chutes-api-key"

messages = [{"content": "Write a short poem about AI", "role": "user"}]

response = completion(
    model="chutes/XiaomiMiMo/MiMo-V2-Flash",
    messages=messages,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Using the `thinking` Parameter (Reasoning Models)

For models with extended reasoning capabilities, you can use the `thinking` parameter:

```python showLineNumbers title="Chutes with Thinking Parameter"
import os
from litellm import completion

os.environ["CHUTES_API_KEY"] = "your-chutes-api-key"

# Enable extended reasoning
response = completion(
    model="chutes/XiaomiMiMo/MiMo-V2-Flash",
    messages=[{
        "role": "user", 
        "content": "Solve this complex math problem step by step: What is 1234 * 5678?"
    }],
    thinking={"type": "enabled"}
)

print(response.choices[0].message.content)
```

### Using `reasoning_effort`

Chutes also supports `reasoning_effort` which automatically maps to the `thinking` parameter:

```python showLineNumbers title="Chutes with Reasoning Effort"
import os
from litellm import completion

os.environ["CHUTES_API_KEY"] = "your-chutes-api-key"

# Reasoning effort options: "low", "medium", "high", "minimal", "none"
response = completion(
    model="chutes/deepseek-ai/DeepSeek-V3.1-TEE",
    messages=[{
        "role": "user",
        "content": "Analyze the economic impact of AI"
    }],
    reasoning_effort="high"  # Maps to {"type": "enabled"}
)

print(response.choices[0].message.content)
```

## Usage - LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export CHUTES_API_KEY="your-chutes-api-key"
```

### 2. Start the proxy

Create a `config.yaml` file:

```yaml showLineNumbers
model_list:
  - model_name: chutes-deepseek
    litellm_params:
      model: chutes/deepseek-ai/DeepSeek-V3.2-TEE
      api_key: os.environ/CHUTES_API_KEY
  
  - model_name: chutes-mimo
    litellm_params:
      model: chutes/XiaomiMiMo/MiMo-V2-Flash
      api_key: os.environ/CHUTES_API_KEY
```

Then start the proxy:

```bash
litellm --config /path/to/config.yaml
```

### 3. Test it

<Tabs>
<TabItem value="python" label="Python SDK">

```python showLineNumbers
from litellm import completion

response = completion(
    model="chutes-deepseek",  # This maps to chutes/deepseek-ai/DeepSeek-V3.2-TEE
    messages=[{"role": "user", "content": "What is 2+2?"}]
)

print(response.choices[0].message.content)
```
</TabItem>

<TabItem value="curl" label="Curl Request">

```bash showLineNumbers
curl --location 'http://0.0.0.0:4000/chat/completions' \\
--header 'Content-Type: application/json' \\
--header 'Authorization: Bearer <your-litellm-proxy-key>' \\
--data '{
    "model": "chutes-mimo",
    "messages": [
        {
            "role": "user",
            "content": "What is 2+2?"
        }
    ]
}'
```
</TabItem>

<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers
from openai import OpenAI

client = OpenAI(
    api_key="your-litellm-proxy-key",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="chutes-deepseek",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)

print(response.choices[0].message.content)
```
</TabItem>
</Tabs>

## Function Calling

Chutes supports standard OpenAI function calling:

```python showLineNumbers title="Chutes Function Calling"
import os
from litellm import completion
import json

os.environ["CHUTES_API_KEY"] = "your-chutes-api-key"

# Define the function
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature"
                }
            },
            "required": ["location"]
        }
    }
}]

response = completion(
    model="chutes/deepseek-ai/DeepSeek-V3.2-TEE",
    messages=[{
        "role": "user", 
        "content": "What's the weather like in Tokyo?"
    }],
    tools=tools,
    tool_choice="auto"
)

# Check if the model wants to call a function
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Function called: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
```

## Response Schema (JSON Mode)

Chutes supports structured JSON outputs:

```python showLineNumbers title="Chutes Response Schema"
import os
from litellm import completion
import json

os.environ["CHUTES_API_KEY"] = "your-chutes-api-key"

response = completion(
    model="chutes/Qwen/Qwen3-32B",
    messages=[{
        "role": "user",
        "content": "Extract the name and age from this text: 'John is 25 years old'"
    }],
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
    }
)

result = json.loads(response.choices[0].message.content)
print(result)  # {"name": "John", "age": 25}
```

## Supported OpenAI Parameters

Chutes supports all standard OpenAI-compatible parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | array | **Required**. Array of message objects with 'role' and 'content' |
| `model` | string | **Required**. Model ID or HuggingFace model identifier |
| `stream` | boolean | Optional. Enable streaming responses |
| `temperature` | float | Optional. Sampling temperature (0.0 to 2.0) |
| `top_p` | float | Optional. Nucleus sampling parameter (0.0 to 1.0) |
| `max_tokens` | integer | Optional. Maximum tokens to generate |
| `frequency_penalty` | float | Optional. Penalize frequent tokens (-2.0 to 2.0) |
| `presence_penalty` | float | Optional. Penalize tokens based on presence (-2.0 to 2.0) |
| `stop` | string/array | Optional. Stop sequences |
| `tools` | array | Optional. List of available tools/functions |
| `tool_choice` | string/object | Optional. Control tool/function calling |
| `response_format` | object | Optional. Response format specification |
| `thinking` | object/boolean | **LiteLLM-specific**. Enable/disable extended reasoning for reasoning models. Maps to `chat_template_kwargs.enable_thinking` internally. Supports both boolean (`true`/`false`) and dict format (`{"type": "enabled"|"disabled"}`) |
| `reasoning_effort` | string | **LiteLLM-specific**. Control reasoning depth ("low", "medium", "high", "none", "minimal"). Maps to `chat_template_kwargs.enable_thinking` internally. |

## Internal Parameter Mapping

LiteLLM automatically transforms OpenAI parameters to Chutes API format internally. You don't need to change your code - just use the standard OpenAI parameters.

| LiteLLM Parameter | Maps to Chutes API | Example |
|-------------------|-------------------|---------|
| `thinking={"type": "enabled"}` | `chat_template_kwargs.enable_thinking=true` | Standard OpenAI reasoning format |
| `thinking={"type": "disabled"}` | `chat_template_kwargs.enable_thinking=false` | Disable reasoning |
| `thinking=true` | `chat_template_kwargs.enable_thinking=true` | Direct boolean format |
| `thinking=false` | `chat_template_kwargs.enable_thinking=false` | Direct boolean format |
| `reasoning_effort="low"|"medium"|"high"` | `chat_template_kwargs.enable_thinking=true` | Simplified reasoning control |
| `reasoning_effort="none"|"minimal"` | `chat_template_kwargs.enable_thinking=false` | Disable reasoning |

## Support Frameworks

Chutes provides optimized templates for popular AI frameworks:

### vLLM (High-Performance LLM Serving)
- OpenAI-compatible endpoints
- Multi-GPU scaling support
- Advanced optimization settings
- Best for production workloads

### SGLang (Advanced LLM Serving)
- Structured generation capabilities
- Advanced features and controls
- Custom configuration options
- Best for complex use cases

### Diffusion Models (Image Generation)
- Pre-configured image generation templates
- Optimized settings for best results
- Support for popular diffusion models

### Embedding Models
- Text embedding templates
- Vector search optimization
- Support for popular embedding models

## Authentication

Chutes supports multiple authentication methods:
- API Key via `X-API-Key` header
- Bearer token via `Authorization` header

Example for LiteLLM (uses environment variable):
```python
os.environ["CHUTES_API_KEY"] = "your-api-key"
```

You can also pass the API key directly:
```python
response = completion(
    model="chutes/deepseek-ai/DeepSeek-V3.2-TEE",
    messages=[...],
    api_key="your-api-key"
)
```

## Performance Optimization

Chutes offers hardware selection and optimization:
- **Small Models (7B-13B)**: 1 GPU with 24GB VRAM
- **Medium Models (30B-70B)**: 4 GPUs with 80GB VRAM each
- **Large Models (100B+)**: 8 GPUs with 140GB+ VRAM each

Engine optimization parameters available for fine-tuning performance.

## Deployment Options

Chutes provides flexible deployment:
- **Quick Setup**: Use pre-built templates for instant deployment
- **Custom Images**: Deploy with custom Docker images
- **Scaling**: Configure max instances and auto-scaling thresholds
- **Hardware**: Choose specific GPU types and configurations

Additional configuration options through environment variables:
```bash
# Optional: Custom API base URL (default: https://llm.chutes.ai/v1)
export CHUTES_API_BASE="https://llm.chutes.ai/v1"
```

## Additional Resources

- [Chutes Documentation](https://chutes.ai/docs)
- [Chutes Getting Started](https://chutes.ai/docs/getting-started/running-a-chute)
- [Chutes API Reference](https://chutes.ai/docs/sdk-reference)
- [LiteLLM Documentation](https://docs.litellm.ai/)
