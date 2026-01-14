"""
Test ZAI streaming response cleanup for GLM-4.7 model
Tests that <tool_call> XML tags are properly removed from streaming chunks
"""

import pytest
from litellm.llms.vertex_ai.vertex_ai_partner_models.zai.transformation import (
    ZAIChatCompletionStreamingHandler,
)


class TestZAIStreamingCleanup:
    """Test cleanup of <tool_call> tags from ZAI GLM-4.7 streaming responses"""

    def test_content_tool_call_tags_cleanup(self):
        """Test aggressive cleanup of content containing <tool_call> tags"""
        handler = ZAIChatCompletionStreamingHandler(
            streaming_response=iter([]), sync_stream=True
        )
        
        # Test case 1: Content with <tool_call> tags should be removed entirely
        chunk_str = 'data: {"id":"test","created":123,"model":"zai-org/glm-4.7-maas","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"<tool_call>file"}}]}'
        cleaned = handler._clean_chunk_data(chunk_str)
        
        # Parse the cleaned JSON to check content
        json_start = cleaned.find("{")
        json_data = cleaned[json_start:]
        import json
        data = json.loads(json_data)
        
        # Content should be empty after cleanup
        assert data["choices"][0]["delta"]["content"] == ""
        
    def test_function_name_tool_call_tags_cleanup(self):
        """Test cleanup of <tool_call> tags from function names"""
        handler = ZAIChatCompletionStreamingHandler(
            streaming_response=iter([]), sync_stream=True
        )
        
        # Test case 2: Function name with <tool_call> tags should be cleaned but keep the name
        chunk_str = 'data: {"id":"test","created":123,"model":"zai-org/glm-4.7-maas","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_test","function":{"arguments":"","name":"<tool_call>read_file"},"type":"function","index":0}]}}]}'
        cleaned = handler._clean_chunk_data(chunk_str)
        
        # Parse the cleaned JSON to check function name
        json_start = cleaned.find("{")
        json_data = cleaned[json_start:]
        import json
        data = json.loads(json_data)
        
        # Function name should be cleaned but keep the actual name
        assert data["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "read_file"
        
    def test_multiple_tool_call_tags(self):
        """Test cleanup of multiple consecutive <tool_call> tags"""
        handler = ZAIChatCompletionStreamingHandler(
            streaming_response=iter([]), sync_stream=True
        )
        
        # Test case 3: Multiple consecutive tool_call tags
        chunk_str = 'data: {"id":"test","created":123,"model":"zai-org/glm-4.7-maas","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"<tool_call><tool_call>"}}]}'
        cleaned = handler._clean_chunk_data(chunk_str)
        
        # Parse the cleaned JSON
        json_start = cleaned.find("{")
        json_data = cleaned[json_start:]
        import json
        data = json.loads(json_data)
        
        # Content should be empty after cleanup
        assert data["choices"][0]["delta"]["content"] == ""
        
    def test_normal_content_unchanged(self):
        """Test that normal content without <tool_call> tags is unchanged"""
        handler = ZAIChatCompletionStreamingHandler(
            streaming_response=iter([]), sync_stream=True
        )
        
        # Test case 4: Normal content should be unchanged
        chunk_str = 'data: {"id":"test","created":123,"model":"zai-org/glm-4.7-maas","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"This is normal content"}}]}'
        cleaned = handler._clean_chunk_data(chunk_str)
        
        # Parse the cleaned JSON
        json_start = cleaned.find("{")
        json_data = cleaned[json_start:]
        import json
        data = json.loads(json_data)
        
        # Content should remain unchanged
        assert data["choices"][0]["delta"]["content"] == "This is normal content"
        
    def test_empty_chunk(self):
        """Test that empty chunks are handled gracefully"""
        handler = ZAIChatCompletionStreamingHandler(
            streaming_response=iter([]), sync_stream=True
        )
        
        # Test case 5: Empty chunk
        chunk_str = ""
        cleaned = handler._clean_chunk_data(chunk_str)
        assert cleaned == ""
        
    def test_chunk_without_json(self):
        """Test that chunks without valid JSON are handled gracefully"""
        handler = ZAIChatCompletionStreamingHandler(
            streaming_response=iter([]), sync_stream=True
        )
        
        # Test case 6: Non-JSON chunk with tool_call tags
        chunk_str = "some random text with <tool_call>tags</tool_call>"
        cleaned = handler._clean_chunk_data(chunk_str)
        
        # Should clean up the tags even without JSON structure
        assert "<tool_call>" not in cleaned
        assert "</tool_call>" not in cleaned
        
    def test_iterator_sync_processing(self):
        """Test that sync iterator properly cleans chunks"""
        # Mock streaming response with problematic chunks
        mock_chunks = [
            'data: {"id":"1","created":123,"model":"zai-org/glm-4.7-maas","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"<tool_call>file"}}]}',
            'data: {"id":"2","created":123,"model":"zai-org/glm-4.7-maas","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" normal text"}}]}',
        ]
        
        handler = ZAIChatCompletionStreamingHandler(
            streaming_response=iter(mock_chunks), sync_stream=True
        )
        
        # Process chunks
        chunks = list(handler)
        
        # Should have processed both chunks
        assert len(chunks) == 2
        
        # First chunk should have empty content after cleanup
        assert chunks[0].choices[0].delta.content == ""
        
        # Second chunk should have normal text
        assert chunks[1].choices[0].delta.content == " normal text"
        
    @pytest.mark.asyncio
    async def test_iterator_async_processing(self):
        """Test that async iterator properly cleans chunks"""
        # Mock async streaming response with problematic chunks
        async def mock_async_iterator():
            yield 'data: {"id":"1","created":123,"model":"zai-org/glm-4.7-maas","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"<tool_call>list"}}]}'
            yield 'data: {"id":"2","created":123,"model":"zai-org/glm-4.7-maas","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" normal async text"}}]}'
        
        handler = ZAIChatCompletionStreamingHandler(
            streaming_response=mock_async_iterator(), sync_stream=False
        )
        
        # Process chunks asynchronously
        chunks = []
        async for chunk in handler:
            chunks.append(chunk)
        
        # Should have processed both chunks
        assert len(chunks) == 2
        
        # First chunk should have empty content after cleanup
        assert chunks[0].choices[0].delta.content == ""
        
        # Second chunk should have normal text
        assert chunks[1].choices[0].delta.content == " normal async text"
