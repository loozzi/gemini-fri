import json
import logging
import time

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from routers import ollama
from sdk.core.exceptions import APIError, AuthError, RateLimitError, ServerError
from sdk.core.model_registry import MODELS, find as find_model
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ollama.router)


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
        # Requests with tools or a schema do all their work inside create().
        # Awaiting it before the response starts lets a failure there become a
        # real error status instead of a stream that dies after its 200 header.
        chunks = await completions.create(request)

        async def event_stream():
            try:
                async for chunk in chunks:
                    yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as exc:
                # Headers are already sent, so the status cannot change. Without
                # an error event, clients record the turn as finished for an
                # "unknown" reason and the agent loop just stops.
                logger.warning("Stream failed mid-response: %s", exc)
                message = exc.message if isinstance(exc, APIError) else str(exc)
                error = {"error": {"message": message, "type": "server_error"}}
                yield f"data: {json.dumps(error)}\n\n"
                return
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    response = await completions.create(request)
    return response


# Models are static, so the whole list shares one `created` stamp: process
# start. OpenAI clients only ever sort on it.
_MODELS_CREATED = int(time.time())


def _model_payload(model) -> dict:
    return {
        "id": model.id,
        "object": "model",
        "created": _MODELS_CREATED,
        "owned_by": "google",
        # Not part of OpenAI's schema, but harmless to clients and the only way
        # a caller can tell what a model actually supports here.
        "description": model.description,
        "capabilities": {
            "tools": model.supports_tools,
            "vision": model.supports_vision,
        },
    }


@app.get("/openai/v1/models")
async def list_models():
    return {"object": "list", "data": [_model_payload(m) for m in MODELS]}


@app.get("/openai/v1/models/{model_id:path}")
async def retrieve_model(model_id: str):
    model = find_model(model_id)
    if model is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"The model '{model_id}' does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )
    return _model_payload(model)


@app.get("/health")
async def health():
    return {"status": "ok"}
