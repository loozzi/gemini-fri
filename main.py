import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sdk.core.exceptions import APIError, AuthError, RateLimitError, ServerError
from sdk.core.models import ChatCompletionRequest
from sdk.resources.chat import ChatCompletions

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SERVER_API_KEY = os.environ.get("SERVER_API_KEY", "")  # optional: protect this server

app = FastAPI(
    title="OpenAI-Compatible Gemini API",
    description="FastAPI server wrapping Gemini Live API with OpenAI-compatible interface",
    version="0.1.0",
)


def _resolve_api_key(authorization: str | None) -> str:
    """Return the Gemini API key to use for this request."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        # if a server key is configured, token must match it; we still use GEMINI_API_KEY
        if SERVER_API_KEY and token != SERVER_API_KEY:
            raise AuthError(401, "Invalid API key")
    elif SERVER_API_KEY:
        raise AuthError(401, "Authorization header required")
    return GEMINI_API_KEY


@app.exception_handler(AuthError)
async def auth_error_handler(request: Request, exc: AuthError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.message, "type": "invalid_request_error", "code": "invalid_api_key"}},
    )


@app.exception_handler(RateLimitError)
async def rate_limit_handler(request: Request, exc: RateLimitError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.message, "type": "requests", "code": "rate_limit_exceeded"}},
    )


@app.exception_handler(ServerError)
async def server_error_handler(request: Request, exc: ServerError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.message, "type": "server_error", "code": "internal_error"}},
    )


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.message, "type": "api_error"}},
    )


@app.post("/openai/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
):
    api_key = _resolve_api_key(authorization)

    if not api_key:
        raise AuthError(401, "GEMINI_API_KEY not configured on server")

    completions = ChatCompletions(api_key=api_key)

    if request.stream:
        async def event_stream():
            async for chunk in await completions.create(request):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    response = await completions.create(request)
    return response


@app.get("/health")
async def health():
    return {"status": "ok"}
