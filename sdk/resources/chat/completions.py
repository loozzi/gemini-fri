import asyncio
import json
import logging
import re
import time
import uuid
from typing import AsyncIterator, List, NamedTuple, Optional, Union

from tenacity import (
    AsyncRetrying,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from google.genai import types as gtypes

from sdk.core.content import build_parts, estimate_tokens, parts_text
from sdk.core.exceptions import AuthError, InvalidRequestError, RateLimitError
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
from sdk.providers.gemini_live import (
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


_TYPE_MAP = {
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def _resolve_refs(node, defs: dict, depth: int = 0):
    """Inline $ref pointers so nested Pydantic models survive conversion."""
    if depth > 16:  # recursive schema — stop rather than loop forever
        return {"type": "string"}
    if isinstance(node, list):
        return [_resolve_refs(item, defs, depth + 1) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        target = defs.get(node["$ref"].rsplit("/", 1)[-1], {})
        return _resolve_refs(target, defs, depth + 1)
    return {
        key: _resolve_refs(value, defs, depth + 1)
        for key, value in node.items()
        if key != "$defs"
    }


def _convert_schema(schema: dict) -> "gtypes.Schema":
    # Optional[T] arrives as anyOf [T, null] (or type: [T, "null"]). Gemini has
    # no union type, so keep the first non-null variant.
    if "anyOf" in schema:
        variants = [v for v in schema["anyOf"] if v.get("type") != "null"] or [{}]
        merged = dict(variants[0])
        if schema.get("description") and "description" not in merged:
            merged["description"] = schema["description"]
        schema = merged

    raw_type = schema.get("type", "string")
    if isinstance(raw_type, list):
        raw_type = next((t for t in raw_type if t != "null"), "string")
    prop_type = _TYPE_MAP.get(str(raw_type).lower(), "STRING")

    kwargs: dict = {"type": prop_type}
    if desc := schema.get("description"):
        kwargs["description"] = desc
    if prop_type == "ARRAY":
        kwargs["items"] = _convert_schema(schema.get("items", {}))
    if prop_type == "OBJECT" and schema.get("properties"):
        kwargs["properties"] = {
            k: _convert_schema(v) for k, v in schema["properties"].items()
        }
        if schema.get("required"):
            kwargs["required"] = schema["required"]
    return gtypes.Schema(**kwargs)


def _object_schema(params: dict) -> "Optional[gtypes.Schema]":
    """Convert a JSON Schema object node → Gemini Schema, or None if empty."""
    params = _resolve_refs(params, params.get("$defs", {}))
    properties = {
        name: _convert_schema(prop)
        for name, prop in params.get("properties", {}).items()
    }
    if not properties:
        return None
    return gtypes.Schema(
        type="OBJECT",
        properties=properties,
        required=params.get("required", []),
    )


def _convert_tools(openai_tools: List[dict]) -> list:
    """Convert OpenAI tools format → Gemini FunctionDeclaration list."""
    function_declarations = []
    for tool in openai_tools:
        if tool.get("type") != "function":
            continue
        func = tool["function"]
        function_declarations.append(
            gtypes.FunctionDeclaration(
                name=func["name"],
                description=func.get("description", ""),
                parameters=_object_schema(func.get("parameters", {})),
            )
        )

    if not function_declarations:
        return []
    return [gtypes.Tool(function_declarations=function_declarations)]


# ─── Structured output (response_format) ──────────────────────────────────────
#
# The Live API rejects `response_schema` outright ("response_schema not
# supported in generation config") and this model refuses TEXT modality, so
# native structured output is unavailable. Instead the schema is declared as a
# function and the model is told to answer by calling it — the tool-calling
# path already works over the audio-transcription session. If the model answers
# in prose anyway, parsing the text as JSON is the fallback.


class _Structured(NamedTuple):
    tool: Optional[object]  # gtypes.Tool, or None for schema-less json_object
    name: Optional[str]     # function name to look for in the tool call
    directive: str          # appended to the system prompt


_SCHEMA_DIRECTIVE = (
    "\n\nAnswer only by calling the function `{name}` with the extracted data. "
    "Do not reply with prose or explanation. "
    "Keep field values in the same language as the source content."
)

_JSON_DIRECTIVE = (
    "\n\nReply with a single valid JSON object and nothing else: "
    "no markdown fences, no preamble, no explanation. "
    "Keep values in the same language as the source content."
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _build_structured(response_format: Optional[dict]) -> Optional[_Structured]:
    """Interpret OpenAI's `response_format` field."""
    if not response_format:
        return None
    if not isinstance(response_format, dict):
        raise InvalidRequestError(400, "response_format must be an object")

    kind = response_format.get("type")
    if kind in (None, "text"):
        return None

    if kind == "json_object":
        return _Structured(tool=None, name=None, directive=_JSON_DIRECTIVE)

    if kind != "json_schema":
        raise InvalidRequestError(
            400,
            f"response_format.type {kind!r} is not supported; "
            "expected 'text', 'json_object' or 'json_schema'",
        )

    spec = response_format.get("json_schema")
    if not isinstance(spec, dict):
        raise InvalidRequestError(
            400, "response_format.json_schema must be an object"
        )

    name = spec.get("name")
    if not isinstance(name, str) or not name:
        raise InvalidRequestError(
            400, "response_format.json_schema.name must be a non-empty string"
        )

    schema = spec.get("schema")
    if not isinstance(schema, dict):
        raise InvalidRequestError(
            400, "response_format.json_schema.schema must be an object"
        )

    tool = gtypes.Tool(function_declarations=[
        gtypes.FunctionDeclaration(
            name=name,
            description=spec.get("description")
            or schema.get("description")
            or "Return the extracted data in this exact shape.",
            parameters=_object_schema(schema),
        )
    ])
    return _Structured(tool=tool, name=name, directive=_SCHEMA_DIRECTIVE.format(name=name))


def _completion(
    model: str,
    message: Message,
    finish_reason: str,
    prompt_words: int,
    completion_words: int = 0,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
        usage=Usage(
            prompt_tokens=prompt_words,
            completion_tokens=completion_words,
            total_tokens=prompt_words + completion_words,
        ),
    )


def _parse_json_text(text: str) -> Optional[dict]:
    """Best-effort JSON out of a prose answer, tolerating markdown fences."""
    if not text:
        return None
    candidate = _FENCE_RE.sub("", text.strip())
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


# ─── ChatCompletions ──────────────────────────────────────────────────────────


class ChatCompletions:
    def __init__(self, api_key: str = DEFAULT_API_KEY, tpm: int = 65_000):
        self._api_key = api_key
        self._bucket = TokenBucket(tpm)

    async def create(
        self,
        request: ChatCompletionRequest,
    ) -> Union[ChatCompletionResponse, AsyncIterator[dict]]:
        system_prompt, parts = build_parts(request.messages)
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        gemini_tools = _convert_tools(request.tools) if request.tools else []

        structured = _build_structured(request.response_format)
        if structured:
            system_prompt += structured.directive
            if structured.tool is not None:
                gemini_tools = [*gemini_tools, structured.tool]

        gen_kwargs = dict(
            temperature=request.temperature if request.temperature != 1.0 else None,
            max_output_tokens=request.max_tokens,
            top_p=request.top_p if request.top_p != 1.0 else None,
        )

        # The streaming path cannot carry tools or a schema, so those requests
        # run to completion first and are replayed as chunks. Streaming them
        # directly would silently drop what the client asked for.
        if request.stream and structured is None and not gemini_tools:
            return self._stream(parts, system_prompt, request.model, **gen_kwargs)

        response = await self._complete(
            parts, system_prompt, request.model, gemini_tools, structured, **gen_kwargs
        )

        if request.stream:
            return self._replay(response)

        return response

    async def _complete(
        self,
        parts: List[gtypes.Part],
        system_prompt: str,
        model: str,
        tools: Optional[list] = None,
        structured: Optional[_Structured] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> ChatCompletionResponse:
        await self._bucket.consume(estimate_tokens(parts))
        prompt_words = len(parts_text(parts).split())

        result = None
        async for attempt in AsyncRetrying(**_RETRY_CONFIG):
            with attempt:
                try:
                    if tools:
                        text, function_calls = await chat_once_ex(
                            parts=parts,
                            api_key=self._api_key,
                            system_prompt=system_prompt,
                            tools=tools,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            top_p=top_p,
                        )
                    else:
                        text = await chat_once(
                            parts=parts,
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

        if structured is not None:
            payload = None
            if structured.name:
                payload = next(
                    (fc["args"] for fc in function_calls if fc["name"] == structured.name),
                    None,
                )
            # Model may ignore the tool and answer in prose; it is often still JSON.
            if payload is None and not function_calls:
                payload = _parse_json_text(text)

            if payload is not None:
                content = json.dumps(payload, ensure_ascii=False)
                return _completion(
                    model,
                    Message(role="assistant", content=content),
                    "stop",
                    prompt_words,
                    len(content.split()),
                )
            if not function_calls:
                logger.warning(
                    "response_format requested but the model returned neither a %s call "
                    "nor JSON; passing the raw text through",
                    structured.name or "json_object",
                )

        if function_calls:
            return _completion(
                model,
                Message(
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
                "tool_calls",
                prompt_words,
            )

        return _completion(
            model,
            Message(role="assistant", content=text),
            "stop",
            prompt_words,
            len(text.split()),
        )

    async def _replay(self, response: ChatCompletionResponse) -> AsyncIterator[dict]:
        """Emit a finished completion as SSE chunks."""
        choice = response.choices[0]
        base = {
            "id": response.id,
            "object": "chat.completion.chunk",
            "created": response.created,
            "model": response.model,
        }

        yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
        if choice.message and choice.message.content:
            yield {
                **base,
                "choices": [
                    {"index": 0, "delta": {"content": choice.message.content}, "finish_reason": None}
                ],
            }
        if choice.message and choice.message.tool_calls:
            yield {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": i,
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for i, tc in enumerate(choice.message.tool_calls)
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        yield {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice.finish_reason}],
        }

    async def _stream(
        self,
        parts: List[gtypes.Part],
        system_prompt: str,
        model: str,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> AsyncIterator[dict]:
        await self._bucket.consume(estimate_tokens(parts))

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
                    parts=parts,
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
