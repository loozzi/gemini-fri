import time
import uuid
from typing import AsyncIterator, List, Union

from sdk.core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceDelta,
    Message,
    Usage,
)
from gemini_live_text import chat_once, chat_stream, DEFAULT_API_KEY, DEFAULT_SYSTEM_PROMPT


def _build_gemini_prompt(messages: List[Message]) -> tuple[str, str]:
    """Convert OpenAI-format messages → (system_prompt, user_message)."""
    system_parts: list[str] = []
    conversation_parts: list[str] = []

    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if msg.role == "system":
            system_parts.append(content)
        elif msg.role == "user":
            conversation_parts.append(f"User: {content}")
        elif msg.role == "assistant":
            conversation_parts.append(f"Assistant: {content}")

    system_prompt = "\n".join(system_parts) if system_parts else DEFAULT_SYSTEM_PROMPT
    user_message = "\n".join(conversation_parts)
    return system_prompt, user_message


class ChatCompletions:
    def __init__(self, api_key: str = DEFAULT_API_KEY):
        self._api_key = api_key

    async def create(
        self,
        request: ChatCompletionRequest,
    ) -> Union[ChatCompletionResponse, AsyncIterator[dict]]:
        system_prompt, user_message = _build_gemini_prompt(request.messages)

        if request.stream:
            return self._stream(user_message, system_prompt, request.model)

        return await self._complete(user_message, system_prompt, request.model)

    async def _complete(
        self, user_message: str, system_prompt: str, model: str
    ) -> ChatCompletionResponse:
        text = await chat_once(
            message=user_message,
            api_key=self._api_key,
            system_prompt=system_prompt,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=len(user_message.split()),
                completion_tokens=len(text.split()),
                total_tokens=len(user_message.split()) + len(text.split()),
            ),
        )

    async def _stream(
        self, user_message: str, system_prompt: str, model: str
    ) -> AsyncIterator[dict]:
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        # yield first chunk with role
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }

        async for text_chunk in chat_stream(
            message=user_message,
            api_key=self._api_key,
            system_prompt=system_prompt,
        ):
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": text_chunk}, "finish_reason": None}],
            }

        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
