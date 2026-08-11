"""Translate between the Ollama protocol and the OpenAI-shaped core.

Nothing here talks to Gemini. Requests are rewritten into
`ChatCompletionRequest` and handed to `ChatCompletions`, so images, tools,
response_format, retry and the token bucket are reused as-is; the results are
then reshaped into Ollama's NDJSON.
"""

import base64
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, List, Optional

from sdk.core.content import sniff_image_mime
from sdk.core.exceptions import InvalidRequestError
from sdk.core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    ToolCall,
    ToolCallFunction,
)
from sdk.core.ollama_models import OllamaChatRequest, OllamaMessage
from sdk.providers.gemini_live import MODEL

OLLAMA_MODEL_TAG = f"{MODEL}:latest"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ─── Ollama → OpenAI ──────────────────────────────────────────────────────────


def _image_data_url(raw: str, where: str) -> str:
    """Ollama sends bare base64, so the mime type must be read off the bytes."""
    payload = raw.strip()
    if payload.startswith("data:"):
        return payload  # some clients send a full data URL anyway

    try:
        data = base64.b64decode(payload)
    except Exception as exc:
        raise InvalidRequestError(400, f"{where}: malformed base64 image") from exc

    mime_type = sniff_image_mime(data)
    if mime_type is None:
        raise InvalidRequestError(
            400,
            f"{where}: unrecognised image format; expected PNG, JPEG, WEBP, HEIC or HEIF",
        )
    return f"data:{mime_type};base64,{payload}"


def _to_openai_message(msg: OllamaMessage, index: int) -> Message:
    content: Any = msg.content or ""

    if msg.images:
        parts: List[dict] = []
        if msg.content:
            parts.append({"type": "text", "text": msg.content})
        for image_index, image in enumerate(msg.images):
            url = _image_data_url(image, f"messages[{index}].images[{image_index}]")
            parts.append({"type": "image_url", "image_url": {"url": url}})
        content = parts

    tool_calls = None
    if msg.tool_calls:
        tool_calls = [
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                function=ToolCallFunction(
                    name=call.function.name,
                    arguments=json.dumps(call.function.arguments, ensure_ascii=False),
                ),
            )
            for call in msg.tool_calls
        ]
        content = content or None

    return Message(
        role=msg.role,
        content=content,
        name=msg.tool_name,
        # Ollama has no call ids; the tool name is the only handle available.
        tool_call_id=msg.tool_name if msg.role == "tool" else None,
        tool_calls=tool_calls,
    )


def _to_response_format(fmt: Any) -> Optional[dict]:
    if fmt is None or fmt == "":
        return None
    if isinstance(fmt, str):
        if fmt.lower() == "json":
            return {"type": "json_object"}
        raise InvalidRequestError(
            400, f"format {fmt!r} is not supported; expected 'json' or a JSON Schema object"
        )
    if isinstance(fmt, dict):
        return {
            "type": "json_schema",
            "json_schema": {"name": fmt.get("title") or "response", "schema": fmt},
        }
    raise InvalidRequestError(400, "format must be 'json' or a JSON Schema object")


def to_chat_request(request: OllamaChatRequest) -> ChatCompletionRequest:
    options = request.options or {}
    tuning: dict = {}
    if (temperature := options.get("temperature")) is not None:
        tuning["temperature"] = temperature
    if (top_p := options.get("top_p")) is not None:
        tuning["top_p"] = top_p
    num_predict = options.get("num_predict")
    if isinstance(num_predict, int) and num_predict > 0:  # -1 means "unlimited"
        tuning["max_tokens"] = num_predict

    return ChatCompletionRequest(
        model=request.model,
        messages=[_to_openai_message(m, i) for i, m in enumerate(request.messages)],
        tools=request.tools or None,
        response_format=_to_response_format(request.format),
        stream=request.stream,
        **tuning,
    )


# ─── OpenAI → Ollama ──────────────────────────────────────────────────────────


def _ollama_tool_calls(message: Message) -> Optional[List[dict]]:
    if not message.tool_calls:
        return None
    calls = []
    for call in message.tool_calls:
        try:
            arguments = json.loads(call.function.arguments)
        except (json.JSONDecodeError, ValueError):
            arguments = {}
        calls.append({"function": {"name": call.function.name, "arguments": arguments}})
    return calls


def _final_chunk(
    model: str,
    started: float,
    done_reason: str = "stop",
    prompt_count: int = 0,
    eval_count: int = 0,
    content: str = "",
    tool_calls: Optional[List[dict]] = None,
) -> dict:
    # Ollama reports durations in nanoseconds. Only the wall clock is known
    # here: nothing is loaded, and the prompt/eval split is not observable, so
    # the whole elapsed time is attributed to eval so tokens/sec stays sane.
    elapsed_ns = int((time.perf_counter() - started) * 1_000_000_000)

    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "model": model,
        "created_at": now_iso(),
        "message": message,
        "done": True,
        "done_reason": done_reason,
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": prompt_count,
        "prompt_eval_duration": 0,
        "eval_count": eval_count,
        "eval_duration": elapsed_ns,
    }


def to_ollama_response(
    response: ChatCompletionResponse, model: str, started: float
) -> dict:
    choice = response.choices[0]
    message = choice.message or Message(role="assistant", content="")
    usage = response.usage
    return _final_chunk(
        model,
        started,
        done_reason="stop",
        prompt_count=usage.prompt_tokens if usage else 0,
        eval_count=usage.completion_tokens if usage else 0,
        content=message.content or "",
        tool_calls=_ollama_tool_calls(message),
    )


async def to_ollama_stream(
    chunks: AsyncIterator[dict], model: str, started: float
) -> AsyncIterator[dict]:
    eval_count = 0

    async for chunk in chunks:
        choices = chunk.get("choices") or [{}]
        delta = choices[0].get("delta") or {}

        if content := delta.get("content"):
            eval_count += len(content.split())
            yield {
                "model": model,
                "created_at": now_iso(),
                "message": {"role": "assistant", "content": content},
                "done": False,
            }

        if raw_calls := delta.get("tool_calls"):
            calls = []
            for call in raw_calls:
                function = call.get("function") or {}
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except (json.JSONDecodeError, ValueError):
                    arguments = {}
                calls.append(
                    {"function": {"name": function.get("name"), "arguments": arguments}}
                )
            yield {
                "model": model,
                "created_at": now_iso(),
                "message": {"role": "assistant", "content": "", "tool_calls": calls},
                "done": False,
            }

    yield _final_chunk(model, started, eval_count=eval_count)
