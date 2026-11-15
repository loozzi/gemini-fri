"""Gemini Live API client owned by the sdk.

Mirrors `gemini_live_text.py` at the repo root, which stays a standalone CLI
demo. Two differences: these functions take a list of `types.Part` rather than
a plain string, and they send it with `send_client_content` instead of
`send_realtime_input`. The latter matters for images — `send_realtime_input`
gives no ordering guarantee, and images carry enough preprocessing cost that a
following text message can reach the model first.
"""

import json
import logging
import os
import re
import uuid as _uuid
from typing import AsyncGenerator, List, Optional, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types

from sdk.core.model_registry import DEFAULT_MODEL_ID, resolve as resolve_model
from sdk.core.models import CompletionTokensDetails, PromptTokensDetails, Usage
from sdk.providers import live_sessions

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY")
# Kept as the module-level default; the servable set lives in sdk.core.model_registry.
MODEL = DEFAULT_MODEL_ID
DEFAULT_SYSTEM_PROMPT = "Bạn là trợ lý AI hữu ích, trả lời ngắn gọn và rõ ràng."
_EXTENDED_THINKING = "gemini-3.8-live-extended-thinking"


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
    model: str = MODEL,
) -> types.LiveConnectConfig:
    """Build the Live session config.

    `thinking_budget` is left unset unless a caller asks: omitting the field
    keeps legacy models' own defaults. Passing 0
    turns reasoning off on legacy models, which cuts time-to-first-token on the
    native-audio models (~2.3s → ~1.2s) and does roughly nothing on
    gemini-3.1-flash-live-preview, which barely thinks to begin with. Gemini 3.8
    Live omits thinking config; Extended Thinking uses levels and cannot turn
    reasoning off.
    """
    model_id = resolve_model(model)
    extra = {}
    if model_id == _EXTENDED_THINKING:
        # The live endpoint requires a level even when effort was omitted.
        level = "low"
        if thinking_budget is not None:
            # This model cannot disable reasoning; map minimal/none to low.
            level = (
                "low" if thinking_budget <= 512
                else "medium" if thinking_budget <= 2048
                else "high"
            )
        extra["thinking_config"] = types.ThinkingConfig(thinking_level=level)
    elif model_id != "gemini-3.8-live" and thinking_budget is not None:
        extra["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

    session_tools = [types.Tool.model_validate(tool).model_copy(deep=True) for tool in (tools or [])]
    for tool in session_tools:
        for declaration in tool.function_declarations or []:
            if model_id == _EXTENDED_THINKING:
                declaration.behavior = types.Behavior.NON_BLOCKING
            elif model_id == "gemini-3.8-live":
                # Preserve the bridge's request/response tool lifecycle.
                declaration.behavior = types.Behavior.BLOCKING

    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(parts=[types.Part(text=system_prompt)]),
        tools=session_tools,
        generation_config=_make_generation_config(temperature, max_output_tokens, top_p),
        **extra,
    )


class _InteractionSocket:
    """Keep interactionStatus before google-genai 2.8 drops unknown fields.

    The SDK also ends receive() at every turnComplete, including spoken fillers.
    Keep this adapter local to Extended Thinking sessions until the SDK exposes
    the interaction lifecycle itself.
    """

    def __init__(self, socket):
        self.socket = socket
        self.status = None

    def __getattr__(self, name):
        return getattr(self.socket, name)

    async def recv(self, *args, **kwargs):
        raw = await self.socket.recv(*args, **kwargs)
        payload = json.loads(raw) if raw else {}
        content = payload.get("serverContent") or {}
        self.status = content.get("interactionStatus", payload.get("interactionStatus", self.status))
        if set(payload) == {"interactionStatus"}:
            # The SDK treats unknown-only messages as API errors.
            return json.dumps({"serverContent": {"turnComplete": self.status == "IDLE"}})
        return raw


def _track_interaction(session, model: str) -> None:
    if resolve_model(model) == _EXTENDED_THINKING and not isinstance(session._ws, _InteractionSocket):
        session._ws = _InteractionSocket(session._ws)


async def _responses(session):
    """Read through intermediate utterances until idle or a tool call."""
    socket = getattr(session, "_ws", None)
    while True:
        received = False
        async for response in session.receive():
            received = True
            yield response
            if response.tool_call:
                return
            content = response.server_content
            if isinstance(socket, _InteractionSocket):
                if socket.status == "IDLE":
                    return
            elif content and content.turn_complete:
                return
        if not received or not isinstance(socket, _InteractionSocket):
            return


async def _send_turn(session, parts: List[types.Part]) -> None:
    await session.send_client_content(
        turns=types.Content(role="user", parts=parts),
        turn_complete=True,
    )


# ─── Usage ────────────────────────────────────────────────────────────────────


def _usage(meta: Optional[types.UsageMetadata]) -> Optional[Usage]:
    """Gemini usage metadata → OpenAI usage.

    The Live API sends it on the message that completes a turn. A turn paused
    on a tool call carries none: nothing follows the call until the tool
    response goes back.
    """
    if meta is None:
        return None
    prompt = (meta.prompt_token_count or 0) + (meta.tool_use_prompt_token_count or 0)
    reasoning = meta.thoughts_token_count or 0
    # Gemini keeps thoughts out of response_token_count; OpenAI counts them in.
    completion = (meta.response_token_count or 0) + reasoning
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=meta.cached_content_token_count or 0),
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=reasoning),
    )


def _add_usage(a: Optional[Usage], b: Optional[Usage]) -> Optional[Usage]:
    if a is None or b is None:
        return a or b
    return Usage(
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
        prompt_tokens_details=PromptTokensDetails(
            cached_tokens=a.prompt_tokens_details.cached_tokens + b.prompt_tokens_details.cached_tokens
        ),
        completion_tokens_details=CompletionTokensDetails(
            reasoning_tokens=a.completion_tokens_details.reasoning_tokens
            + b.completion_tokens_details.reasoning_tokens
        ),
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
) -> tuple[str, Optional[Usage]]:
    client = _client(api_key)
    config = _live_config(
        system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        thinking_budget=thinking_budget,
        model=model,
    )

    response_parts: list[str] = []
    usage: Optional[Usage] = None

    async with client.aio.live.connect(model=resolve_model(model), config=config) as session:
        _track_interaction(session, model)
        await _send_turn(session, parts)

        async for response in _responses(session):
            usage = _usage(response.usage_metadata) or usage
            server_content = response.server_content
            if server_content is None:
                continue

            if server_content.output_transcription:
                response_parts.append(server_content.output_transcription.text)

    return "".join(response_parts), usage


async def chat_stream(
    parts: List[types.Part],
    api_key: str = DEFAULT_API_KEY,
    model: str = MODEL,
    thinking_budget: Optional[int] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
) -> AsyncGenerator[Union[str, Usage], None]:
    """Yield text chunks từ Gemini Live API (dùng cho streaming response).

    The last item is the turn's `Usage`, when Gemini reported one.
    """
    usage: Optional[Usage] = None
    client = _client(api_key)
    config = _live_config(
        system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        thinking_budget=thinking_budget,
        model=model,
    )

    async with client.aio.live.connect(model=resolve_model(model), config=config) as session:
        _track_interaction(session, model)
        await _send_turn(session, parts)

        async for response in _responses(session):
            usage = _usage(response.usage_metadata) or usage
            server_content = response.server_content
            if server_content is None:
                continue

            if server_content.output_transcription:
                yield server_content.output_transcription.text

        if usage is not None:
            yield usage


async def _open(api_key: str, model_id: str, config: types.LiveConnectConfig):
    """Enter a Live connection by hand, so it can outlive the request that opened it."""
    connection = _client(api_key).aio.live.connect(model=model_id, config=config)
    session = await connection.__aenter__()
    _track_interaction(session, model_id)
    return connection, session


async def _read_turn(session) -> tuple[str, list, Optional[Usage]]:
    """Read until the model ends its turn or pauses it to call tools."""
    text_parts: list[str] = []
    calls: list = []
    usage: Optional[Usage] = None

    async for response in _responses(session):
        usage = _usage(response.usage_metadata) or usage
        server_content = response.server_content
        if server_content:
            transcription = server_content.output_transcription
            if transcription and transcription.text:
                text_parts.append(transcription.text)

        if response.tool_call:
            calls.extend(response.tool_call.function_calls)
            break

    return "".join(text_parts), calls, usage


# A turn with no letter or digit in it — an empty transcript, "\n\n", a stray
# "```" — is not an answer, yet native-audio models end tool loops that way
# often: 4 of 4 measured agent runs with default thinking. Agent clients show it
# as an empty reply and wait for the user to type "continue", so the server says
# it once instead, in the same session. In those runs every nudge drew a real
# summary; one also re-ran a read-only check.
_NUDGE = "Continue. If the task is already complete, reply with a one-sentence summary of what you did."
_HAS_WORD = re.compile(r"\w")


async def _read_answer(session) -> tuple[str, list, Optional[Usage]]:
    """`_read_turn`, nudging the model once if the turn came back empty.

    If the nudge still yields nothing, the original turn is returned so the
    reply is never worse than what the model first said.
    """
    text, calls, usage = await _read_turn(session)
    if calls or _HAS_WORD.search(text):
        return text, calls, usage

    logger.info("Model ended its turn without an answer; nudging it once")
    await session.send_client_content(
        turns=types.Content(role="user", parts=[types.Part(text=_NUDGE)]),
        turn_complete=True,
    )
    retry_text, retry_calls, retry_usage = await _read_turn(session)
    # Both turns were generated for this request, so both are counted.
    usage = _add_usage(usage, retry_usage)
    if retry_calls or _HAS_WORD.search(retry_text):
        return retry_text, retry_calls, usage
    return text, calls, usage


async def _settle(
    connection, session, model_id: str, text: str, calls: list, usage: Optional[Usage]
) -> tuple[str, list[dict], Optional[Usage]]:
    """Park the session while the model waits on tools; close it otherwise."""
    if not calls:
        await live_sessions.close_connection(connection)
        return text, [], usage

    function_calls: list[dict] = []
    pending: dict[str, live_sessions.PendingCall] = {}
    for fc in calls:
        call_id = fc.id or f"call_{_uuid.uuid4().hex[:8]}"
        function_calls.append({
            "id": call_id,
            "name": fc.name,
            "args": dict(fc.args) if fc.args else {},
        })
        pending[call_id] = live_sessions.PendingCall(name=fc.name, gemini_id=fc.id)

    await live_sessions.park(live_sessions.ParkedSession(
        connection=connection, session=session, model=model_id, pending=pending,
    ))
    return text, function_calls, usage


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
    tool_results: Optional[List[tuple[str, str]]] = None,
) -> tuple[str, list[dict], Optional[Usage]]:
    """
    Extended chat_once with function calling support.
    Returns (text, function_calls, usage) where function_calls is a list of
    {"id": str, "name": str, "args": dict}. `usage` is None when Gemini sent
    none, which is always so for a turn paused on tool calls.

    When the model calls tools its session is parked rather than closed, and
    `tool_results` — the (call id, output) pairs ending the next request's
    history — resume it through `send_tool_response`. That is the only form in
    which the Live API takes tool results: `function_call` parts replayed in
    client content are rejected with 1007, and in the flattened transcript that
    `parts` carries the model does not recognise its own finished steps, so
    agents repeat them or stop mid-task. `parts` is still replayed whenever no
    parked session matches: a first request, an expired session, or a client
    such as Ollama that sends no call ids.
    """
    model_id = resolve_model(model)

    entry = None
    if tool_results:
        entry = await live_sessions.claim([call_id for call_id, _ in tool_results], model_id)

    if entry is not None:
        try:
            await entry.session.send_tool_response(function_responses=[
                types.FunctionResponse(
                    name=entry.pending[call_id].name,
                    response={"output": output},
                    **({"id": entry.pending[call_id].gemini_id} if entry.pending[call_id].gemini_id else {}),
                )
                for call_id, output in tool_results
            ])
            text, calls, usage = await _read_answer(entry.session)
        except Exception as exc:
            # Upstream may have closed the session while the client ran its tools.
            logger.warning("Parked Live session could not resume (%s); replaying history", exc)
            await entry.close()
        else:
            return await _settle(entry.connection, entry.session, model_id, text, calls, usage)

    config = _live_config(
        system_prompt,
        tools=tools,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        thinking_budget=thinking_budget,
        model=model,
    )
    connection, session = await _open(api_key, model_id, config)
    try:
        await _send_turn(session, parts)
        text, calls, usage = await _read_answer(session)
    except BaseException:
        await live_sessions.close_connection(connection)
        raise
    return await _settle(connection, session, model_id, text, calls, usage)
