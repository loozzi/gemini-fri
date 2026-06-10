import json
import logging
import time
import uuid
from typing import AsyncIterator, List, Optional, Union

from tenacity import (
    AsyncRetrying,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from sdk.core.exceptions import AuthError
from sdk.core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceDelta,
    Message,
    ToolCall,
    ToolCallFunction,
    Usage,
)
from gemini_live_text import (
    chat_once,
    chat_once_ex,
    chat_stream,
    DEFAULT_API_KEY,
    DEFAULT_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

_RETRY_CONFIG = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_not_exception_type(AuthError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _build_gemini_prompt(messages: List[Message]) -> tuple[str, str]:
    """Convert OpenAI-format messages → (system_prompt, conversation_text)."""
    system_parts: list[str] = []
    conversation_parts: list[str] = []

    for msg in messages:
        if msg.role == "system":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            system_parts.append(content)
        elif msg.role == "user":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            conversation_parts.append(f"User: {content}")
        elif msg.role == "assistant":
            if msg.tool_calls:
                calls_str = "; ".join(
                    f"{tc.function.name}({tc.function.arguments})"
                    for tc in msg.tool_calls
                )
                conversation_parts.append(f"Assistant called: {calls_str}")
            elif msg.content:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                conversation_parts.append(f"Assistant: {content}")
        elif msg.role == "tool":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            conversation_parts.append(f"Tool result [{msg.tool_call_id}]: {content}")

    system_prompt = "\n".join(system_parts) if system_parts else DEFAULT_SYSTEM_PROMPT
    user_message = "\n".join(conversation_parts)
    return system_prompt, user_message


def _convert_tools(openai_tools: List[dict]) -> list:
    """Convert OpenAI tools format → Gemini FunctionDeclaration list."""
    from google.genai import types as gtypes

    type_map = {
        "string": "STRING",
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
    }

    function_declarations = []
    for tool in openai_tools:
        if tool.get("type") != "function":
            continue
        func = tool["function"]
        params = func.get("parameters", {})

        properties: dict = {}
        for prop_name, prop_schema in params.get("properties", {}).items():
            prop_type = type_map.get(prop_schema.get("type", "string").lower(), "STRING")
            prop_kwargs: dict = {"type": prop_type}
            if desc := prop_schema.get("description"):
                prop_kwargs["description"] = desc
            properties[prop_name] = gtypes.Schema(**prop_kwargs)

        gemini_params = None
        if properties:
            gemini_params = gtypes.Schema(
                type="OBJECT",
                properties=properties,
                required=params.get("required", []),
            )

        function_declarations.append(
            gtypes.FunctionDeclaration(
                name=func["name"],
                description=func.get("description", ""),
                parameters=gemini_params,
            )
        )

    if not function_declarations:
        return []
    return [gtypes.Tool(function_declarations=function_declarations)]


# ─── ChatCompletions ──────────────────────────────────────────────────────────


class ChatCompletions:
    def __init__(self, api_key: str = DEFAULT_API_KEY):
        self._api_key = api_key

    async def create(
        self,
        request: ChatCompletionRequest,
    ) -> Union[ChatCompletionResponse, AsyncIterator[dict]]:
        system_prompt, user_message = _build_gemini_prompt(request.messages)
        gemini_tools = _convert_tools(request.tools) if request.tools else []

        if request.stream:
            return self._stream(user_message, system_prompt, request.model)

        return await self._complete(user_message, system_prompt, request.model, gemini_tools)

    async def _complete(
        self,
        user_message: str,
        system_prompt: str,
        model: str,
        tools: Optional[list] = None,
    ) -> ChatCompletionResponse:
        result = None
        async for attempt in AsyncRetrying(**_RETRY_CONFIG):
            with attempt:
                if tools:
                    text, function_calls = await chat_once_ex(
                        message=user_message,
                        api_key=self._api_key,
                        system_prompt=system_prompt,
                        tools=tools,
                    )
                else:
                    text = await chat_once(
                        message=user_message,
                        api_key=self._api_key,
                        system_prompt=system_prompt,
                    )
                    function_calls = []
                result = (text, function_calls)

        text, function_calls = result

        if function_calls:
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                object="chat.completion",
                created=int(time.time()),
                model=model,
                choices=[
                    Choice(
                        index=0,
                        message=Message(
                            role="assistant",
                            content=text or None,
                            tool_calls=[
                                ToolCall(
                                    id=fc["id"],
                                    function=ToolCallFunction(
                                        name=fc["name"],
                                        arguments=json.dumps(fc["args"]),
                                    ),
                                )
                                for fc in function_calls
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=Usage(
                    prompt_tokens=len(user_message.split()),
                    completion_tokens=0,
                    total_tokens=len(user_message.split()),
                ),
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

        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }

        # Retry only the connection/generator setup; individual chunks are not retried
        attempt_count = 0
        last_exc: BaseException | None = None
        while attempt_count < 3:
            try:
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
                break  # success
            except AuthError:
                raise
            except Exception as exc:
                attempt_count += 1
                last_exc = exc
                logger.warning("Stream attempt %d failed: %s", attempt_count, exc)
                if attempt_count >= 3:
                    raise
                wait_sec = min(2 ** attempt_count, 30)
                import asyncio
                await asyncio.sleep(wait_sec)

        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
