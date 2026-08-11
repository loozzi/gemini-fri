"""Pydantic schemas for the Ollama protocol.

Only the fields this server can honour are declared; the rest of Ollama's
surface (keep_alive, think, raw, context) is accepted and ignored.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel


class OllamaFunctionCall(BaseModel):
    name: str
    # Ollama carries arguments as an object, where OpenAI uses a JSON string.
    arguments: Dict[str, Any] = {}


class OllamaToolCall(BaseModel):
    function: OllamaFunctionCall


class OllamaMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = ""
    images: Optional[List[str]] = None  # bare base64, no data: prefix
    tool_calls: Optional[List[OllamaToolCall]] = None
    tool_name: Optional[str] = None


class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaMessage] = []
    tools: Optional[List[dict]] = None
    # "json" or a JSON Schema object
    format: Optional[Union[str, dict]] = None
    options: Optional[Dict[str, Any]] = None
    # Ollama streams by default — the opposite of the OpenAI surface.
    stream: bool = True
    keep_alive: Optional[Union[str, int]] = None
    think: Optional[Union[bool, str]] = None


class OllamaShowRequest(BaseModel):
    model: str = ""
    name: Optional[str] = None  # older clients send `name`
    verbose: Optional[bool] = None

    def target(self) -> str:
        return self.model or self.name or ""
