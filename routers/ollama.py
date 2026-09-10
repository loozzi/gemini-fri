"""Ollama-compatible endpoints, backed by Gemini Live.

Errors are shaped and returned here rather than by the app-wide handlers,
which speak OpenAI's error format and must keep doing so for /openai/v1.
"""

import json
import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse, StreamingResponse

from sdk.core.exceptions import APIError, AuthError
from sdk.core.model_registry import MODELS, DEFAULT_MODEL, find as find_model
from sdk.core.ollama_models import OllamaChatRequest, OllamaShowRequest
from sdk.providers.gemini_live import DEFAULT_API_KEY
from sdk.resources.chat import ChatCompletions
from sdk.resources.ollama.chat import (
    ollama_tag,
    to_chat_request,
    to_ollama_response,
    to_ollama_stream,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ollama"])

# Reported to clients that gate features on the Ollama version. This server is
# not an Ollama install — the value only says "new enough".
OLLAMA_VERSION = "0.11.4"

_STARTED_AT = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

_DETAILS = {
    "parent_model": "",
    "format": "gguf",
    "family": "gemini",
    "families": ["gemini"],
    "parameter_size": "",
    "quantization_level": "",
}


def _capabilities(model) -> list[str]:
    """What Open WebUI and friends gate their UI on."""
    caps = ["completion"]
    if model.supports_vision:
        caps.append("vision")
    if model.supports_tools:
        caps.append("tools")
    return caps


def _resolve_api_key(authorization: str | None) -> str:
    """Header first, then the environment.

    Most Ollama clients cannot set headers, so an env fallback is what makes
    this surface usable at all.
    """
    if authorization and authorization.startswith("Bearer "):
        key = authorization.removeprefix("Bearer ").strip()
        if key:
            return key
    if DEFAULT_API_KEY and DEFAULT_API_KEY != "YOUR_API_KEY":
        return DEFAULT_API_KEY
    raise AuthError(
        401,
        "no Gemini API key: send 'Authorization: Bearer <key>' or set GEMINI_API_KEY",
    )


def _error(exc: Exception) -> JSONResponse:
    if isinstance(exc, APIError):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.message})
    logger.exception("Unhandled error on the Ollama surface")
    return JSONResponse(status_code=500, content={"error": str(exc)})


@router.post("/chat")
async def chat(
    request: OllamaChatRequest,
    authorization: str | None = Header(default=None),
):
    started = time.perf_counter()
    try:
        api_key = _resolve_api_key(authorization)
        openai_request = to_chat_request(request)
        result = await ChatCompletions(api_key=api_key).create(openai_request)
    except Exception as exc:
        return _error(exc)

    if not openai_request.stream:
        return to_ollama_response(result, request.model, started)

    async def ndjson():
        try:
            async for chunk in to_ollama_stream(result, request.model, started):
                yield json.dumps(chunk, ensure_ascii=False) + "\n"
        except Exception as exc:
            logger.warning("Ollama stream failed: %s", exc)
            message = exc.message if isinstance(exc, APIError) else str(exc)
            yield json.dumps({"error": message}, ensure_ascii=False) + "\n"

    return StreamingResponse(ndjson(), media_type="application/x-ndjson")


@router.get("/tags")
async def tags():
    return {
        "models": [
            {
                "name": ollama_tag(model.id),
                "model": ollama_tag(model.id),
                "modified_at": _STARTED_AT,
                # No model file exists here, so these stay empty rather than
                # carrying invented values.
                "size": 0,
                "digest": "",
                "details": _DETAILS,
            }
            for model in MODELS
        ]
    }


@router.post("/show")
async def show(request: OllamaShowRequest):
    # Unknown names resolve to the default, matching what /api/chat would
    # actually run for that request.
    model = find_model(request.target()) or DEFAULT_MODEL
    return {
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": _DETAILS,
        "model_info": {
            "general.architecture": "gemini",
            "general.basename": model.id,
        },
        # Open WebUI reads "vision" to decide whether to offer image upload.
        "capabilities": _capabilities(model),
    }


@router.get("/version")
async def version():
    return {"version": OLLAMA_VERSION}
