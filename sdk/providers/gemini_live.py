"""Gemini Live API client owned by the sdk.

Mirrors `gemini_live_text.py` at the repo root, which stays a standalone CLI
demo. Two differences: these functions take a list of `types.Part` rather than
a plain string, and they send it with `send_client_content` instead of
`send_realtime_input`. The latter matters for images — `send_realtime_input`
gives no ordering guarantee, and images carry enough preprocessing cost that a
following text message can reach the model first.
"""

import os
import uuid as _uuid
from typing import AsyncGenerator, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from sdk.core.model_registry import DEFAULT_MODEL_ID, resolve as resolve_model

load_dotenv()

DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY")
# Kept as the module-level default; the servable set lives in sdk.core.model_registry.
MODEL = DEFAULT_MODEL_ID
DEFAULT_SYSTEM_PROMPT = "Bạn là trợ lý AI hữu ích, trả lời ngắn gọn và rõ ràng."


# ─── Setup ────────────────────────────────────────────────────────────────────


def _make_generation_config(
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
) -> Optional[types.GenerationConfig]:
    kwargs = {k: v for k, v in {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "top_p": top_p,
    }.items() if v is not None}
    return types.GenerationConfig(**kwargs) if kwargs else None


def _client(api_key: str) -> genai.Client:
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1alpha"),
    )


def _live_config(
    system_prompt: str,
    tools: Optional[list] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    thinking_budget: Optional[int] = None,
) -> types.LiveConnectConfig:
    """Build the Live session config.

    `thinking_budget` is left unset unless a caller asks: omitting the field
    keeps each model's own default, which is what clients get today. Passing 0
    turns reasoning off, which measurably cuts time-to-first-token on the
    native-audio models (~2.3s → ~1.2s) and does roughly nothing on
    gemini-3.1-flash-live-preview, which barely thinks to begin with.
    """
    extra = {}
    if thinking_budget is not None:
        extra["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(parts=[types.Part(text=system_prompt)]),
        tools=tools or [],
        generation_config=_make_generation_config(temperature, max_output_tokens, top_p),
        **extra,
    )


async def _send_turn(session, parts: List[types.Part]) -> None:
    await session.send_client_content(
        turns=types.Content(role="user", parts=parts),
        turn_complete=True,
    )


# ─── Chat ─────────────────────────────────────────────────────────────────────


async def chat_once(
    parts: List[types.Part],
    api_key: str = DEFAULT_API_KEY,
    model: str = MODEL,
    thinking_budget: Optional[int] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
) -> str:
    client = _client(api_key)
    config = _live_config(
        system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        thinking_budget=thinking_budget,
    )

    response_parts: list[str] = []

    async with client.aio.live.connect(model=resolve_model(model), config=config) as session:
        await _send_turn(session, parts)

        async for response in session.receive():
            server_content = response.server_content
            if server_content is None:
                continue

            if server_content.output_transcription:
                response_parts.append(server_content.output_transcription.text)

            if server_content.turn_complete:
                break

    return "".join(response_parts)


async def chat_stream(
    parts: List[types.Part],
    api_key: str = DEFAULT_API_KEY,
    model: str = MODEL,
    thinking_budget: Optional[int] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
) -> AsyncGenerator[str, None]:
    """Yield text chunks từ Gemini Live API (dùng cho streaming response)."""
    client = _client(api_key)
    config = _live_config(
        system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        thinking_budget=thinking_budget,
    )

    async with client.aio.live.connect(model=resolve_model(model), config=config) as session:
        await _send_turn(session, parts)

        async for response in session.receive():
            server_content = response.server_content
            if server_content is None:
                continue

            if server_content.output_transcription:
                yield server_content.output_transcription.text

            if server_content.turn_complete:
                break


async def chat_once_ex(
    parts: List[types.Part],
    api_key: str = DEFAULT_API_KEY,
    model: str = MODEL,
    thinking_budget: Optional[int] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    tools: Optional[list] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
) -> tuple[str, list[dict]]:
    """
    Extended chat_once with function calling support.
    Returns (text, function_calls) where function_calls is a list of
    {"id": str, "name": str, "args": dict}.
    """
    client = _client(api_key)
    config = _live_config(
        system_prompt,
        tools=tools,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        thinking_budget=thinking_budget,
    )

    text_parts: list[str] = []
    function_calls: list[dict] = []

    async with client.aio.live.connect(model=resolve_model(model), config=config) as session:
        await _send_turn(session, parts)

        async for response in session.receive():
            server_content = response.server_content
            if server_content:
                if server_content.output_transcription:
                    text_parts.append(server_content.output_transcription.text)
                if server_content.turn_complete:
                    break

            if response.tool_call:
                for fc in response.tool_call.function_calls:
                    function_calls.append({
                        "id": getattr(fc, "id", None) or f"call_{_uuid.uuid4().hex[:8]}",
                        "name": fc.name,
                        "args": dict(fc.args) if fc.args else {},
                    })
                break

    return "".join(text_parts), function_calls
