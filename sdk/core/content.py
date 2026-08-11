"""Convert OpenAI-format messages into Gemini Live API parts.

Text-only requests collapse to a single `Part(text=...)` holding exactly the
string the previous text-only implementation built, so existing flows keep
their behaviour. Images become `inline_data` parts sitting at the position
they appeared in the conversation.
"""

import base64
import binascii
import re
from typing import List, Optional, Union

from google.genai import types

from sdk.core.exceptions import InvalidRequestError
from sdk.core.models import Message

SUPPORTED_IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
}

# Gemini caps a whole request — text, system instruction and inline bytes — at
# 20MB. Stay under it with room for the prompt and base64 transport overhead.
MAX_TOTAL_IMAGE_BYTES = 15 * 1024 * 1024

_DATA_URL_RE = re.compile(r"data:(?P<mime>[^;,]+);base64,(?P<payload>.*)", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")

# A token is either a text fragment or an already-built image part. Adjacent
# text fragments merge into one part at the end.
_Token = Union[str, types.Part]


# ─── Data URLs ────────────────────────────────────────────────────────────────


def parse_data_url(url: str, where: str = "image") -> tuple[str, bytes]:
    """Decode a base64 data URL into (mime_type, bytes)."""
    match = _DATA_URL_RE.fullmatch(url.strip())
    if match is None:
        raise InvalidRequestError(
            400,
            f"{where}: only base64 data URLs are supported "
            f"(data:<mime>;base64,<payload>); remote URLs are not fetched",
        )

    mime_type = match.group("mime").strip().lower()
    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_MIME_TYPES))
        raise InvalidRequestError(
            400,
            f"{where}: unsupported image type '{mime_type}'; expected one of {supported}",
        )

    payload = _WHITESPACE_RE.sub("", match.group("payload"))
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequestError(400, f"{where}: malformed base64 image payload") from exc

    if not data:
        raise InvalidRequestError(400, f"{where}: image payload is empty")

    return mime_type, data


def sniff_image_mime(data: bytes) -> Optional[str]:
    """Identify an image by magic bytes.

    Ollama sends bare base64 with no mime type, so the format has to be read
    off the bytes themselves.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"hevx"):
            return "image/heic"
        if brand in (b"heif", b"mif1", b"msf1"):
            return "image/heif"
    return None


class _ImageBudget:
    """Track total decoded image bytes across a request."""

    def __init__(self, limit: int = MAX_TOTAL_IMAGE_BYTES):
        self._limit = limit
        self._used = 0

    def add(self, size: int, where: str) -> None:
        self._used += size
        if self._used > self._limit:
            raise InvalidRequestError(
                400,
                f"{where}: total image payload is {self._used} bytes, "
                f"over the {self._limit // (1024 * 1024)}MB limit",
            )


# ─── Content parts ────────────────────────────────────────────────────────────


def _image_part(raw: dict, where: str, budget: _ImageBudget) -> types.Part:
    image_url = raw.get("image_url")
    if not isinstance(image_url, dict):
        raise InvalidRequestError(400, f"{where}.image_url must be an object")

    url = image_url.get("url")
    if not isinstance(url, str) or not url:
        raise InvalidRequestError(400, f"{where}.image_url.url must be a non-empty string")

    mime_type, data = parse_data_url(url, where)
    budget.add(len(data), where)
    return types.Part(inline_data=types.Blob(data=data, mime_type=mime_type))


def _content_tokens(
    content: Optional[Union[str, List[dict]]],
    index: int,
    role: str,
    budget: _ImageBudget,
) -> list[_Token]:
    """Split one message's content into text fragments and image parts."""
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return [str(content)]

    tokens: list[_Token] = []
    previous_was_text = False

    for part_index, raw in enumerate(content):
        where = f"messages[{index}].content[{part_index}]"
        if not isinstance(raw, dict):
            raise InvalidRequestError(400, f"{where} must be an object")

        part_type = raw.get("type")
        if part_type == "text":
            if previous_was_text:
                tokens.append("\n")
            tokens.append(str(raw.get("text", "")))
            previous_was_text = True
        elif part_type == "image_url":
            if role != "user":
                raise InvalidRequestError(
                    400,
                    f"{where}: images are only supported in user messages, got role '{role}'",
                )
            tokens.append(_image_part(raw, where, budget))
            previous_was_text = False
        else:
            raise InvalidRequestError(
                400, f"{where}: unsupported content part type {part_type!r}"
            )

    return tokens


def _message_tokens(msg: Message, index: int, budget: _ImageBudget) -> list[_Token]:
    """Tokens for one non-system message, including its role prefix."""
    if msg.role == "user":
        return ["User: ", *_content_tokens(msg.content, index, msg.role, budget)]

    if msg.role == "assistant":
        if msg.tool_calls:
            calls_str = "; ".join(
                f"{tc.function.name}({tc.function.arguments})" for tc in msg.tool_calls
            )
            return [f"Assistant called: {calls_str}"]
        if msg.content:
            return ["Assistant: ", *_content_tokens(msg.content, index, msg.role, budget)]
        return []

    if msg.role == "tool":
        return [
            f"Tool result [{msg.tool_call_id}]: ",
            *_content_tokens(msg.content, index, msg.role, budget),
        ]

    return []


def _merge(tokens: list[_Token]) -> list[types.Part]:
    """Collapse adjacent text fragments into single parts."""
    parts: list[types.Part] = []
    buffer: list[str] = []

    for token in tokens:
        if isinstance(token, str):
            buffer.append(token)
            continue
        if buffer:
            parts.append(types.Part(text="".join(buffer)))
            buffer = []
        parts.append(token)

    if buffer or not parts:
        parts.append(types.Part(text="".join(buffer)))

    return parts


def build_parts(messages: List[Message]) -> tuple[Optional[str], list[types.Part]]:
    """Convert OpenAI messages → (system_prompt, parts).

    `system_prompt` is None when the request carries no system message, leaving
    the caller to apply its own default.
    """
    budget = _ImageBudget()
    system_parts: list[str] = []
    tokens: list[_Token] = []

    for index, msg in enumerate(messages):
        if msg.role == "system":
            system_tokens = _content_tokens(msg.content, index, msg.role, budget)
            system_parts.append("".join(str(token) for token in system_tokens))
            continue

        message_tokens = _message_tokens(msg, index, budget)
        if not message_tokens:
            continue
        if tokens:
            tokens.append("\n")
        tokens.extend(message_tokens)

    system_prompt = "\n".join(system_parts) if system_parts else None
    return system_prompt, _merge(tokens)


# ─── Introspection helpers ────────────────────────────────────────────────────


def parts_text(parts: List[types.Part]) -> str:
    """All text carried by `parts`, concatenated."""
    return "".join(part.text for part in parts if part.text)


def estimate_tokens(parts: List[types.Part]) -> int:
    """Rough token count for client-side throttling.

    Text reuses the existing len//4 heuristic. Images cost at least 258 tokens
    (Gemini's price for one 768x768 tile) and scale with payload size, so large
    images do not slip past the token bucket uncounted.
    """
    total = 0
    for part in parts:
        if part.text:
            total += len(part.text) // 4
        elif part.inline_data and part.inline_data.data:
            total += max(258, len(part.inline_data.data) // 750)
    return max(1, total)
