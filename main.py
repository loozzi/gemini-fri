import json
import logging
import time

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sdk.core.exceptions import APIError, AuthError, RateLimitError, ServerError
from sdk.core.models import ChatCompletionRequest
from sdk.resources.chat import ChatCompletions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("api")

app = FastAPI(
    title="OpenAI-Compatible Gemini API",
    description="FastAPI server wrapping Gemini Live API with OpenAI-compatible interface",
    version="0.1.0",
)


def _resolve_api_key(authorization: str | None) -> str:
    """Extract the Gemini API key from the Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthError(401, "Authorization header required: Bearer <your-gemini-api-key>")
    key = authorization.removeprefix("Bearer ").strip()
    if not key:
        raise AuthError(401, "API key must not be empty")
    return key


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    logger.info("→ %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error("← ERROR %.1fms — %s: %s", elapsed_ms, type(exc).__name__, exc)
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("← %d %.1fms", response.status_code, elapsed_ms)
    return response


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
