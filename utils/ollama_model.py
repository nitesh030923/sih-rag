"""
Custom Ollama model implementation for pydantic-ai with native API support.
"""

import json
import logging
from typing import Any, AsyncIterator, Optional
import httpx
from pydantic_ai.models import Model, ModelSettings
from pydantic_ai.messages import (
    Message,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)

logger = logging.getLogger(__name__)


class OllamaModel(Model):
    """Custom Ollama model using native API."""
    
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/v1').rstrip('/')  # Remove OpenAI API suffix
        self.http_client = httpx.AsyncClient(timeout=120.0)
    
    async def request(
        self,
        messages: list[Message],
        model_settings: Optional[ModelSettings] = None,
    ) -> ModelResponse:
        """
        Make a request to Ollama using native API with simplified tool handling.
        """
        try:
            # Convert messages to Ollama format
            prompt = self._build_prompt(messages)
            
            # Make request to Ollama
            response = await self.http_client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": getattr(model_settings, 'temperature', 0.7) if model_settings else 0.7,
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract response text
            response_text = result.get("response", "")
            
            # Return as ModelResponse with TextPart
            return ModelResponse(
                parts=[TextPart(content=response_text)],
                timestamp=None
            )
            
        except Exception as e:
            logger.error(f"Ollama request failed: {e}", exc_info=True)
            raise
    
    async def request_stream(
        self,
        messages: list[Message],
        model_settings: Optional[ModelSettings] = None,
    ) -> AsyncIterator[ModelResponse]:
        """
        Stream responses from Ollama.
        """
        try:
            prompt = self._build_prompt(messages)
            
            async with self.http_client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": getattr(model_settings, 'temperature', 0.7) if model_settings else 0.7,
                    }
                }
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if chunk := data.get("response"):
                                yield ModelResponse(
                                    parts=[TextPart(content=chunk)],
                                    timestamp=None
                                )
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}", exc_info=True)
            raise
    
    def _build_prompt(self, messages: list[Message]) -> str:
        """
        Build a prompt from message history.
        Includes system prompt and conversation history.
        """
        parts = []
        
        for msg in messages:
            role = getattr(msg, 'role', 'user')
            
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        parts.append(f"{role.upper()}: {part.content}")
                    elif isinstance(part, ToolReturnPart):
                        # Include tool results in context
                        parts.append(f"CONTEXT: {part.content}")
            elif hasattr(msg, 'content'):
                content = msg.content
                if isinstance(content, str):
                    parts.append(f"{role.upper()}: {content}")
        
        return "\n\n".join(parts)
    
    async def close(self):
        """Close the HTTP client."""
        await self.http_client.aclose()
