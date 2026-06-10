"""
Gemini Live API — Interactive Text Chat
Model: gemini-2.0-flash-live-001
Output: Text response (streamed to console)

Cài đặt:
    pip install google-genai python-dotenv

Sử dụng:
    python gemini_live_text.py
    python gemini_live_text.py --system-prompt "Bạn là trợ lý AI hữu ích"
    python gemini_live_text.py --one-shot --message "Giải thích về AI"
"""

import argparse
import asyncio
import os
import uuid as _uuid
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY")
MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_SYSTEM_PROMPT = "Bạn là trợ lý AI hữu ích, trả lời ngắn gọn và rõ ràng."


# ─── Core: one-shot chat ──────────────────────────────────────────────────────


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


async def chat_once(
    message: str,
    api_key: str = DEFAULT_API_KEY,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
) -> str:
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1alpha"),
    )

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(
            parts=[types.Part(text=system_prompt)]
        ),
        generation_config=_make_generation_config(temperature, max_output_tokens, top_p),
    )

    response_parts: list[str] = []

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        await session.send_realtime_input(text=message)

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
    message: str,
    api_key: str = DEFAULT_API_KEY,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
) -> AsyncGenerator[str, None]:
    """Yield text chunks từ Gemini Live API (dùng cho streaming response)."""
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1alpha"),
    )

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(
            parts=[types.Part(text=system_prompt)]
        ),
        generation_config=_make_generation_config(temperature, max_output_tokens, top_p),
    )

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        await session.send_realtime_input(text=message)

        async for response in session.receive():
            server_content = response.server_content
            if server_content is None:
                continue

            if server_content.output_transcription:
                yield server_content.output_transcription.text

            if server_content.turn_complete:
                break


async def chat_once_ex(
    message: str,
    api_key: str = DEFAULT_API_KEY,
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
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1alpha"),
    )

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(
            parts=[types.Part(text=system_prompt)]
        ),
        tools=tools or [],
        generation_config=_make_generation_config(temperature, max_output_tokens, top_p),
    )

    text_parts: list[str] = []
    function_calls: list[dict] = []

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        await session.send_realtime_input(text=message)

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


# ─── Core: interactive session ────────────────────────────────────────────────


async def chat_interactive(
    api_key: str = DEFAULT_API_KEY,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> None:
    """
    Chạy phiên chat tương tác nhiều lượt trong một Live session.
    Gõ 'quit' hoặc 'exit' để thoát, 'clear' để xóa màn hình.
    """
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1alpha"),
    )

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(
            parts=[types.Part(text=system_prompt)]
        ),
    )

    print(f"[✓] Gemini Live Text Chat | Model: {MODEL}")
    print("[i] Gõ 'quit' hoặc 'exit' để thoát\n")

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[✓] Đã thoát.")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("[✓] Đã thoát.")
                break

            if user_input.lower() == "clear":
                os.system("clear" if os.name != "nt" else "cls")
                continue

            await session.send_realtime_input(text=user_input)

            print("Gemini: ", end="", flush=True)
            async for response in session.receive():
                server_content = response.server_content
                if server_content is None:
                    continue

                if server_content.output_transcription:
                    print(server_content.output_transcription.text, end="", flush=True)

                if server_content.turn_complete:
                    print()  # newline sau khi turn kết thúc
                    break


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gemini Live API — Text Chat"
    )
    parser.add_argument(
        "--system-prompt", "-s",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System instruction cho model",
    )
    parser.add_argument(
        "--one-shot", "-1",
        action="store_true",
        help="Chế độ một lượt (không tương tác)",
    )
    parser.add_argument(
        "--message", "-m",
        type=str,
        default=None,
        help="Tin nhắn dùng với --one-shot",
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=DEFAULT_API_KEY,
        help="Gemini API key (hoặc set env GEMINI_API_KEY)",
    )
    return parser.parse_args()


async def _main():
    args = parse_args()

    if args.api_key == "YOUR_API_KEY":
        print("[✗] Chưa set API key!")
        print("    Cách 1: export GEMINI_API_KEY=your_key")
        print("    Cách 2: python gemini_live_text.py --api-key your_key")
        return

    if args.one_shot:
        message = args.message or input("Nhập tin nhắn: ").strip()
        if not message:
            print("[✗] Tin nhắn trống.")
            return
        print(f"[→] Gửi: \"{message[:60]}{'...' if len(message) > 60 else ''}\"")
        reply = await chat_once(
            message=message,
            api_key=args.api_key,
            system_prompt=args.system_prompt,
        )
        print(f"[Gemini] {reply}")
    else:
        await chat_interactive(
            api_key=args.api_key,
            system_prompt=args.system_prompt,
        )


if __name__ == "__main__":
    asyncio.run(_main())


# ─── Usage examples ───────────────────────────────────────────────────────────
#
# Interactive chat (nhiều lượt):
#   python gemini_live_text.py
#
# One-shot (một lượt):
#   python gemini_live_text.py --one-shot --message "Thủ đô Việt Nam là gì?"
#
# Với system prompt tùy chỉnh:
#   python gemini_live_text.py --system-prompt "Bạn là chuyên gia lập trình Python"
#
# Bật thinking:
#   python gemini_live_text.py --thinking -1
#
# Dùng như module:
#   import asyncio
#   from gemini_live_text import chat_once, chat_interactive
#
#   # One-shot
#   reply = asyncio.run(chat_once("Xin chào!"))
#   print(reply)
