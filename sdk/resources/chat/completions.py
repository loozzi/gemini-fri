import asyncio
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

from sdk.core.exceptions import AuthError, RateLimitError
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


# ─── Token Bucket ─────────────────────────────────────────────────────────────


class TokenBucket:
    """Client-side rate limiter: proactively throttle to stay under TPM limit."""

    def __init__(self, tpm: int = 65_000):
        self._capacity = tpm
        self._tokens = float(tpm)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._capacity,
                self._tokens + elapsed / 60.0 * self._capacity,
            )
            self._last_refill = now

            if tokens > self._tokens:
                wait_sec = (tokens - self._tokens) / self._capacity * 60.0
                logger.debug(
                    "Token bucket throttling %.1fs (need %d, have %.0f)",
                    wait_sec, tokens, self._tokens,
                )
                await asyncio.sleep(wait_sec)
                self._tokens = 0.0
            else:
                self._tokens -= tokens


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ─── Retry ────────────────────────────────────────────────────────────────────


_wait_default = wait_exponential(multiplier=1, min=2, max=30)
_wait_rate_limit = wait_exponential(multiplier=2, min=30, max=120)


def _smart_wait(retry_state):
    if isinstance(retry_state.outcome.exception(), RateLimitError):
        return _wait_rate_limit(retry_state)
    return _wait_default(retry_state)


_RETRY_CONFIG = dict(
    stop=stop_after_attempt(5),
    wait=_smart_wait,
    retry=retry_if_not_exception_type(AuthError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


# ─── Exception translation ────────────────────────────────────────────────────


def _translate_exc(exc: Exception) -> Exception:
    """Convert google-genai SDK exceptions to our SDK exceptions."""
    try:
        from google.genai import errors as _gerrors
        if isinstance(exc, (_gerrors.ClientError, _gerrors.APIError)):
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            msg = str(exc)
            if code == 429 or "429" in msg:
                return RateLimitError(429, msg)
            if code == 401 or "401" in msg:
                return AuthError(401, msg)
    except (ImportError, AttributeError):
        pass
    return exc


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

    def convert_schema(schema: dict) -> gtypes.Schema:
        raw_type = schema.get("type", "string").lower()
        prop_type = type_map.get(raw_type, "STRING")
        kwargs: dict = {"type": prop_type}
        if desc := schema.get("description"):
            kwargs["description"] = desc
        if prop_type == "ARRAY":
            items_schema = schema.get("items", {})
            kwargs["items"] = convert_schema(items_schema)
        if prop_type == "OBJECT" and schema.get("properties"):
            kwargs["properties"] = {
                k: convert_schema(v) for k, v in schema["properties"].items()
            }
            if schema.get("required"):
                kwargs["required"] = schema["required"]
        return gtypes.Schema(**kwargs)

    function_declarations = []
    for tool in openai_tools:
        if tool.get("type") != "function":
            continue
        func = tool["function"]
        params = func.get("parameters", {})

        properties: dict = {}
        for prop_name, prop_schema in params.get("properties", {}).items():
            properties[prop_name] = convert_schema(prop_schema)

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
    def __init__(self, api_key: str = DEFAULT_API_KEY, tpm: int = 65_000):
        self._api_key = api_key
        self._bucket = TokenBucket(tpm)

    async def create(
        self,
        request: ChatCompletionRequest,
    ) -> Union[ChatCompletionResponse, AsyncIterator[dict]]:
        system_prompt, user_message = _build_gemini_prompt(request.messages)
        gemini_tools = _convert_tools(request.tools) if request.tools else []

        gen_kwargs = dict(
            temperature=request.temperature if request.temperature != 1.0 else None,
            max_output_tokens=request.max_tokens,
            top_p=request.top_p if request.top_p != 1.0 else None,
        )

        if request.stream:
            return self._stream(user_message, system_prompt, request.model, **gen_kwargs)

        return await self._complete(user_message, system_prompt, request.model, gemini_tools, **gen_kwargs)

    async def _complete(
        self,
        user_message: str,
        system_prompt: str,
        model: str,
        tools: Optional[list] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> ChatCompletionResponse:
        await self._bucket.consume(_estimate_tokens(user_message))

        result = None
        async for attempt in AsyncRetrying(**_RETRY_CONFIG):
            with attempt:
                try:
                    if tools:
                        text, function_calls = await chat_once_ex(
                            message=user_message,
                            api_key=self._api_key,
                            system_prompt=system_prompt,
                            tools=tools,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            top_p=top_p,
                        )
                    else:
                        text = await chat_once(
                            message=user_message,
                            api_key=self._api_key,
                            system_prompt=system_prompt,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            top_p=top_p,
                        )
                        function_calls = []
                except Exception as exc:
                    translated = _translate_exc(exc)
                    if translated is not exc:
                        raise translated from exc
                    raise
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
        self,
        user_message: str,
        system_prompt: str,
        model: str,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> AsyncIterator[dict]:
        await self._bucket.consume(_estimate_tokens(user_message))

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }

        attempt_count = 0
        while attempt_count < 5:
            try:
                async for text_chunk in chat_stream(
                    message=user_message,
                    api_key=self._api_key,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    top_p=top_p,
                ):
                    yield {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": text_chunk}, "finish_reason": None}],
                    }
                break
            except AuthError:
                raise
            except Exception as exc:
                exc = _translate_exc(exc)
                attempt_count += 1
                logger.warning("Stream attempt %d failed: %s", attempt_count, exc)
                if attempt_count >= 5:
                    raise exc
                if isinstance(exc, RateLimitError):
                    wait_sec = min(30 * (2 ** (attempt_count - 1)), 120)
                else:
                    wait_sec = min(2 ** attempt_count, 30)
                await asyncio.sleep(wait_sec)

        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
