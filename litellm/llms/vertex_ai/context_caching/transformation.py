"""
Transformation logic for context caching. 

Why separate file? Make it easy to see how transformation works
"""

import re
from typing import List, Optional, Tuple, Literal, Dict

from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.vertex_ai import CachedContentRequestBody
from litellm.utils import is_cached_message

from ..common_utils import get_supports_system_message
from ..gemini.transformation import (
    _gemini_convert_messages_with_history,
    _transform_system_message,
)


# Context Caching Minimum Token Requirements
# Based on: https://ai.google.dev/gemini-api/docs/caching
# https://cloud.google.com/blog/products/ai-machine-learning/vertex-ai-context-caching
CONTEXT_CACHE_MIN_TOKENS: Dict[str, int] = {
    # Gemini 2.5 models - reduced minimums
    "gemini-2.5-flash": 1024,
    "gemini-2.5-pro": 2048,
    "gemini-2.5-flash-lite": 512,
    # Gemini 3 models
    "gemini-3-flash-preview": 1024,
    "gemini-3-flash": 1024,
    "gemini-3-pro-preview": 4096,
    "gemini-3-pro": 4096,
    # Older models - higher minimums
    "gemini-1.5-pro": 32768,
    "gemini-1.5-pro-001": 32768,
    "gemini-1.5-pro-002": 32768,
    "gemini-1.5-flash": 32768,
    "gemini-1.5-flash-001": 32768,
    "gemini-1.5-flash-002": 32768,
    "gemini-1.5-flash-8b": 32768,
}


def get_context_cache_min_tokens(model: str) -> int:
    """
    Get the minimum token count required for context caching for a given model.
    
    Args:
        model: The model name (e.g., "gemini-2.5-flash", "vertex_ai/gemini-2.5-pro")
        
    Returns:
        int: Minimum number of tokens required for context caching
        
    Examples:
        >>> get_context_cache_min_tokens("gemini-2.5-flash")
        1024
        >>> get_context_cache_min_tokens("vertex_ai/gemini-2.5-pro")
        2048
    """
    # Clean up model name - remove provider prefix
    clean_model = model
    if "/" in model:
        clean_model = model.split("/")[-1]
    
    # Look up exact match first
    if clean_model in CONTEXT_CACHE_MIN_TOKENS:
        return CONTEXT_CACHE_MIN_TOKENS[clean_model]
    
    # Try partial match for versioned models (e.g., gemini-2.5-flash-001)
    for model_pattern, min_tokens in CONTEXT_CACHE_MIN_TOKENS.items():
        if clean_model.startswith(model_pattern):
            return min_tokens
    
    # Default fallback for unknown models - use safest (highest) minimum
    return 32768


def estimate_message_tokens(messages: List[AllMessageValues]) -> int:
    """
    Estimate token count for a list of messages without making an API call.
    
    Uses a character-based approximation: roughly 4 characters per token for English text.
    This is a simple estimation suitable for validation purposes before making actual API calls.
    
    Note: This is an approximation. Actual token count may vary slightly.
    For exact counts, use the provider's countTokens endpoint.
    
    Args:
        messages: List of messages in OpenAI format
        
    Returns:
        int: Estimated token count
        
    Examples:
        >>> estimate_message_tokens([{"role": "user", "content": "Hello world"}])
        3
        >>> estimate_message_tokens([{"role": "user", "content": [{"type": "text", "text": "Hello"}]}])
        1
    """
    total_chars = 0
    
    for message in messages:
        content = message.get("content")
        
        if content is None:
            continue
            
        # Handle string content
        if isinstance(content, str):
            total_chars += len(content)
        # Handle list content (e.g., with cache_control)
        elif isinstance(content, list):
            for content_item in content:
                if isinstance(content_item, dict):
                    # Check for text content - only add if text is non-empty
                    text = content_item.get("text", "")
                    if text:  # Only count non-empty text
                        total_chars += len(text)
                    # Handle image/video content - approximate as more tokens
                    elif content_item.get("type") == "image_url":
                        total_chars += 800  # Rough estimate for images
                    elif content_item.get("type") == "image":
                        total_chars += 800  # Alternative image type
                    elif content_item.get("type") in ["video", "audio"]:
                        total_chars += 2000  # Rough estimate for media
                    # Handle cache_control in nested content (ignore for token count)
                    elif content_item.get("cache_control") is not None:
                        continue
    
    # Character-based approximation: roughly 4 characters per token
    # This works reasonably well for English text
    estimated_tokens = total_chars / 4.0
    
    # Add overhead for message structure (role, formatting, etc.)
    # Each message has some overhead for metadata
    estimated_tokens += len(messages) * 4
    
    return int(estimated_tokens)


def get_first_continuous_block_idx(
    filtered_messages: List[Tuple[int, AllMessageValues]]  # (idx, message)
) -> int:
    """
    Find the array index that ends the first continuous sequence of message blocks.

    Args:
        filtered_messages: List of tuples containing (index, message) pairs

    Returns:
        int: The array index where the first continuous sequence ends
    """
    if not filtered_messages:
        return -1

    if len(filtered_messages) == 1:
        return 0

    current_value = filtered_messages[0][0]

    # Search forward through the array indices
    for i in range(1, len(filtered_messages)):
        if filtered_messages[i][0] != current_value + 1:
            return i - 1
        current_value = filtered_messages[i][0]

    # If we made it through the whole list, return the last index
    return len(filtered_messages) - 1


def extract_ttl_from_cached_messages(messages: List[AllMessageValues]) -> Optional[str]:
    """
    Extract TTL from cached messages. Returns the first valid TTL found.
    
    Args:
        messages: List of messages to extract TTL from
        
    Returns:
        Optional[str]: TTL string in format "3600s" or None if not found/invalid
    """
    for message in messages:
        if not is_cached_message(message):
            continue
            
        content = message.get("content")
        if not content or isinstance(content, str):
            continue
            
        for content_item in content:
            # Type check to ensure content_item is a dictionary before calling .get()
            if not isinstance(content_item, dict):
                continue
                
            cache_control = content_item.get("cache_control")
            if not cache_control or not isinstance(cache_control, dict):
                continue
                
            if cache_control.get("type") != "ephemeral":
                continue
                
            ttl = cache_control.get("ttl")
            if ttl and _is_valid_ttl_format(ttl):
                return str(ttl)
    
    return None


def _is_valid_ttl_format(ttl: str) -> bool:
    """
    Validate TTL format. Should be a string ending with 's' for seconds.
    Examples: "3600s", "7200s", "1.5s"
    
    Args:
        ttl: TTL string to validate
        
    Returns:
        bool: True if valid format, False otherwise
    """
    if not isinstance(ttl, str):
        return False
    
    # TTL should end with 's' and contain a valid number before it
    pattern = r'^([0-9]*\.?[0-9]+)s$'
    match = re.match(pattern, ttl)
    
    if not match:
        return False
    
    try:
        # Ensure the numeric part is valid and positive
        numeric_part = float(match.group(1))
        return numeric_part > 0
    except ValueError:
        return False


def separate_cached_messages(
    messages: List[AllMessageValues],
) -> Tuple[List[AllMessageValues], List[AllMessageValues]]:
    """
    Returns separated cached and non-cached messages.

    Args:
        messages: List of messages to be separated.

    Returns:
        Tuple containing:
        - cached_messages: List of cached messages.
        - non_cached_messages: List of non-cached messages.
    """
    cached_messages: List[AllMessageValues] = []
    non_cached_messages: List[AllMessageValues] = []

    # Extract cached messages and their indices
    filtered_messages: List[Tuple[int, AllMessageValues]] = []
    for idx, message in enumerate(messages):
        if is_cached_message(message=message):
            filtered_messages.append((idx, message))

    # Validate only one block of continuous cached messages
    last_continuous_block_idx = get_first_continuous_block_idx(filtered_messages)
    # Separate messages based on the block of cached messages
    if filtered_messages and last_continuous_block_idx is not None:
        first_cached_idx = filtered_messages[0][0]
        last_cached_idx = filtered_messages[last_continuous_block_idx][0]

        cached_messages = messages[first_cached_idx : last_cached_idx + 1]
        non_cached_messages = (
            messages[:first_cached_idx] + messages[last_cached_idx + 1 :]
        )
    else:
        non_cached_messages = messages

    return cached_messages, non_cached_messages


def remove_cache_control_from_messages(
    messages: List[AllMessageValues],
) -> List[AllMessageValues]:
    """
    Remove cache_control parameters from all messages.

    This is used to disable context caching when requested content is too small
    for model minimum requirements, allowing the request to proceed without caching.

    Args:
        messages: List of messages with potential cache_control parameters

    Returns:
        List of messages with cache_control removed from content arrays

    Examples:
        >>> messages = [{"role": "user", "content": [{"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}]}]
        >>> cleaned = remove_cache_control_from_messages(messages)
        >>> "cache_control" not in cleaned[0]["content"][0]
        True
    """
    from typing import cast

    cleaned_messages: List[AllMessageValues] = []

    for message in messages:
        # Create a copy of the message to avoid mutating the original
        cleaned_message = dict(message)  # type: ignore

        content = cleaned_message.get("content")

        if content is None:
            cleaned_messages.append(cast(AllMessageValues, cleaned_message))
            continue

        # Handle list content (where cache_control is typically found)
        if isinstance(content, list):
            cleaned_content = []
            for content_item in content:
                if isinstance(content_item, dict):
                    # Remove cache_control if present
                    cleaned_item = dict(content_item)
                    cleaned_item.pop("cache_control", None)
                    cleaned_content.append(cleaned_item)
                else:
                    cleaned_content.append(content_item)
            cleaned_message["content"] = cleaned_content  # type: ignore

        cleaned_messages.append(cast(AllMessageValues, cleaned_message))

    return cleaned_messages


def transform_openai_messages_to_gemini_context_caching(
    model: str,
    messages: List[AllMessageValues],
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    cache_key: str,
    vertex_project: Optional[str],
    vertex_location: Optional[str],
) -> CachedContentRequestBody:
    # Extract TTL from cached messages BEFORE system message transformation
    ttl = extract_ttl_from_cached_messages(messages)
    
    supports_system_message = get_supports_system_message(
        model=model, custom_llm_provider=custom_llm_provider
    )

    transformed_system_messages, new_messages = _transform_system_message(
        supports_system_message=supports_system_message, messages=messages
    )

    transformed_messages = _gemini_convert_messages_with_history(messages=new_messages, model=model)
    
    model_name = "models/{}".format(model)

    if custom_llm_provider == "vertex_ai" or custom_llm_provider == "vertex_ai_beta":
        model_name = f"projects/{vertex_project}/locations/{vertex_location}/publishers/google/{model_name}"

    data = CachedContentRequestBody(
        contents=transformed_messages,
        model=model_name,
        displayName=cache_key,
    )
    
    # Add TTL if present and valid
    if ttl:
        data["ttl"] = ttl
    
    if transformed_system_messages is not None:
        data["system_instruction"] = transformed_system_messages

    return data
