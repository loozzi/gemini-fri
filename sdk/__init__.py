from .core.models import ChatCompletionRequest, ChatCompletionResponse
from .resources.chat import ChatCompletions


class _ChatNamespace:
    def __init__(self, api_key: str):
        self.completions = ChatCompletions(api_key=api_key)


class OpenAICompatClient:
    def __init__(self, api_key: str = ""):
        self.chat = _ChatNamespace(api_key=api_key)


__all__ = ["OpenAICompatClient", "ChatCompletionRequest", "ChatCompletionResponse"]
