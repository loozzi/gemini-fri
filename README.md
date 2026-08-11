# gemini-fri

Transforms the **Google Gemini Live API** into an **OpenAI-compatible** and **Ollama-compatible** REST API — drop it in wherever you use OpenAI or Ollama, no code changes needed.

> [!NOTE]
> This project is intended for **research and educational purposes only**. Please use responsibly and refrain from any commercial use.

> [!WARNING]
> This project is **not affiliated with Google**. Use of the Gemini API is subject to [Google's Terms of Service](https://ai.google.dev/gemini-api/terms). The author assumes no responsibility for API quota charges, account actions, or data loss.

---

## Why gemini-fri?

**Problem:** You want an OpenAI-compatible endpoint backed by Gemini, without rewriting your client code.

**Solution:** A local FastAPI server that:

- Exposes `POST /openai/v1/chat/completions` — identical to the OpenAI spec
- Also speaks the **Ollama protocol** (`/api/chat`, `/api/tags`, …) for Open WebUI, `ChatOllama`, Continue
- Works with the OpenAI Python/JS SDK, LangChain, and any OpenAI-compatible tool
- Supports both **streaming** (SSE) and **non-streaming** responses
- Converts multi-turn message history to Gemini Live API format automatically
- Supports **function/tool calling** with automatic OpenAI → Gemini format conversion
- Accepts **images** in the OpenAI `image_url` content format
- Retries failed requests automatically (up to 3 attempts, exponential backoff)

**Use cases:**

- Prototype with Gemini using OpenAI-compatible tooling
- Learn how Live API streaming works under the hood
- Build local AI apps without vendor lock-in

---

## Quick Start

**Step 1 — Get your Gemini API key**

Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and create a key.

**Step 2 — Clone & install**

```bash
git clone <repo-url>
cd gemini-fri

uv sync
source .venv/bin/activate
```

> Requires **Python 3.13+** and [`uv`](https://docs.astral.sh/uv/).

**Step 3 — Run**

```bash
uvicorn main:app --reload
```

Done! Your server is live at `http://localhost:8000`. No `.env` needed — pass your Gemini API key with each request.

**Test it:**

```bash
curl -X POST http://localhost:8000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_GEMINI_API_KEY" \
  -d '{"model": "gemini", "messages": [{"role": "user", "content": "Hello!"}]}'
```

---

## Features

- **OpenAI-compatible** — works as a drop-in replacement for the OpenAI API
- **Streaming support** — real-time SSE chunks via Gemini Live API
- **Image input (vision)** — OpenAI `image_url` content parts with base64 `data:` URLs, sent to Gemini as `inline_data`
- **Structured output** — `response_format` with `json_schema` or `json_object`; works with `with_structured_output()` and `client.beta.chat.completions.parse()`
- **Ollama-compatible surface** — `/api/chat` (NDJSON streaming), `/api/tags`, `/api/show`, `/api/version`; images, tools and `format` all supported
- **Function/tool calling** — full OpenAI tool-call format; automatically converted to Gemini `FunctionDeclaration`
- **Auto-retry** — up to 3 attempts with exponential backoff (2s → 30s); auth errors are not retried
- **Optional auth** — protect your server with a bearer token (`SERVER_API_KEY`)
- **Multi-turn context** — system + user + assistant + tool message history handled automatically
- **Interactive docs** — Swagger UI at `/docs`

---

## Configuration

The server has no required environment variables. Your Gemini API key is passed per-request via the `Authorization` header — the server is stateless and never stores keys.

```
Authorization: Bearer YOUR_GEMINI_API_KEY
```

> For the standalone CLI (`gemini_live_text.py`), you can still set `GEMINI_API_KEY` in a `.env` file or pass `--api-key` on the command line.

---

## Usage Examples

### OpenAI Python SDK

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="YOUR_GEMINI_API_KEY",  # passed as Authorization: Bearer header
)

# Non-streaming
response = await client.chat.completions.create(
    model="gemini",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# Streaming
stream = await client.chat.completions.create(
    model="gemini",
    messages=[{"role": "user", "content": "Tell me a short story"}],
    stream=True,
)
async for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Image Input (Vision)

Send images as OpenAI `image_url` content parts. The URL must be a base64 `data:` URL — remote `http(s)` URLs are **not** fetched by the server.

```python
import base64

with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = await client.chat.completions.create(
    model="gemini",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Có gì trong ảnh này?"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }},
        ],
    }],
)
print(response.choices[0].message.content)
```

Images from earlier turns are re-sent with every request, so follow-up questions about a previous image work. Text and images keep the order you sent them in.

| Constraint | Value |
|---|---|
| Supported types | `image/png`, `image/jpeg`, `image/webp`, `image/heic`, `image/heif` |
| Source | base64 `data:` URLs only |
| Total image size | 15 MB per request (Gemini caps the whole request at 20 MB) |
| Allowed roles | `user` messages only |
| `detail` field | accepted and ignored |

Anything violating these returns `400` before a Gemini session is opened, so a malformed request costs no quota.

### Function/Tool Calling

```python
# same client with api_key="YOUR_GEMINI_API_KEY"
response = await client.chat.completions.create(
    model="gemini",
    messages=[{"role": "user", "content": "What's the weather in Hanoi?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    }],
)
# response.choices[0].finish_reason == "tool_calls"
tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name, tool_call.function.arguments)
```

### Structured Output

`response_format` is supported with both `json_schema` and `json_object`. The response `content` is a JSON string matching your schema.

```python
from pydantic import BaseModel

class EmailContent(BaseModel):
    sender: str | None = None
    recipients: list[str] = []
    body: str

# OpenAI SDK
completion = await client.beta.chat.completions.parse(
    model="gemini",
    messages=[{"role": "user", "content": "Trích xuất thông tin từ email này..."}],
    response_format=EmailContent,
)
print(completion.choices[0].message.parsed)
```

```python
# LangChain — the default method="json_schema" works
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="http://localhost:8000/openai/v1",
                 api_key="YOUR_GEMINI_API_KEY", model="gemini")
result = llm.with_structured_output(EmailContent).invoke([message])
```

> **How this works:** the Gemini Live API has no native structured output — it rejects `response_schema` with `1007: response_schema not supported in generation config`, and this model refuses `response_modalities=["TEXT"]`. So the server declares your schema as a function and instructs the model to answer by calling it, then returns the call arguments as `message.content`. If the model replies with prose instead, the server tries to parse that text as JSON before giving up. Reliable in practice, but not a hard guarantee the way OpenAI's strict mode is.

### cURL — non-streaming

```bash
curl -X POST http://localhost:8000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_GEMINI_API_KEY" \
  -d '{
    "model": "gemini",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of Vietnam?"}
    ]
  }'
```

### cURL — streaming (SSE)

```bash
curl -X POST http://localhost:8000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_GEMINI_API_KEY" \
  -d '{
    "model": "gemini",
    "messages": [{"role": "user", "content": "Count from 1 to 5"}],
    "stream": true
  }'
```

### Ollama-compatible endpoints

Point any Ollama client at this server instead of `http://localhost:11434`.

```python
from ollama import Client

client = Client(host="http://localhost:8000")
MODEL = "gemini-3.1-flash-live-preview:latest"   # as returned by /api/tags

# streaming (Ollama's default)
for chunk in client.chat(model=MODEL, messages=[{"role": "user", "content": "Xin chào!"}]):
    print(chunk.message.content, end="", flush=True)

# with an image — bare base64, no data: prefix
import base64
client.chat(model=MODEL, stream=False, messages=[{
    "role": "user",
    "content": "Ảnh này là gì?",
    "images": [base64.b64encode(open("photo.jpg", "rb").read()).decode()],
}])

# structured output via `format`
client.chat(model=MODEL, stream=False, format={
    "type": "object",
    "properties": {"city": {"type": "string"}},
}, messages=[{"role": "user", "content": "Thủ đô Việt Nam."}])
```

```bash
curl http://localhost:8000/api/chat -d '{
  "model": "gemini-3.1-flash-live-preview:latest",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

**Authentication.** Ollama has no auth, and most Ollama clients cannot set headers. So this surface takes the key from `Authorization: Bearer <key>` if present, otherwise from the `GEMINI_API_KEY` environment variable:

```bash
GEMINI_API_KEY=your_key uvicorn main:app
```

> This is the one place the server is not fully stateless. `/openai/v1` still requires the header and never falls back to the environment.

**Mapping.** `options.temperature` / `top_p` / `num_predict` → generation config; `format` → structured output; `images` → Gemini `inline_data` (mime type detected from the bytes, since Ollama sends bare base64); `tools` and `tool_calls` are translated both ways, including Ollama's object-valued `arguments`. `keep_alive`, `think` and `raw` are accepted and ignored. Model-management endpoints (`/api/pull`, `/api/create`, …) are not implemented — there is no local model to manage.

`/api/tags` reports `size: 0` and `digest: ""` because no model file exists here; those fields are left empty rather than filled with invented values.

### Built-in CLI (`gemini_live_text.py`)

```bash
# Interactive multi-turn chat
python gemini_live_text.py

# One-shot message
python gemini_live_text.py --one-shot --message "What is AI?"

# Custom system prompt
python gemini_live_text.py --system-prompt "You are a Python expert"
```

---

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API docs powered by Swagger UI.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/openai/v1/chat/completions` | OpenAI-compatible chat completions |
| `POST` | `/api/chat` | Ollama-compatible chat (NDJSON streaming) |
| `GET` | `/api/tags` | Ollama-compatible model list |
| `POST` | `/api/show` | Ollama-compatible model info |
| `GET` | `/api/version` | Ollama version (compatibility constant) |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |

### Supported Request Fields

| Field | Type | Description |
|---|---|---|
| `model` | string | Any string (passed through; actual model is `gemini-3.1-flash-live-preview`) |
| `messages` | array | OpenAI message format (`system`, `user`, `assistant`, `tool`); `content` may be a string or a list of `text` / `image_url` parts |
| `stream` | bool | Enable SSE streaming (default: `false`) |
| `tools` | array | OpenAI function definitions (converted to Gemini `FunctionDeclaration`) |
| `tool_choice` | string/object | Accepted but not forwarded to Gemini |
| `response_format` | object | `{"type": "json_schema", ...}`, `{"type": "json_object"}` or `{"type": "text"}`; implemented via function calling |
| `temperature` | float | Forwarded to Gemini `GenerationConfig` (default: model default) |
| `max_tokens` | int | Forwarded as `max_output_tokens` |
| `top_p` | float | Forwarded to Gemini `GenerationConfig` (default: model default) |

### Error Codes

| HTTP Status | Exception | Cause |
|---|---|---|
| `400` | `InvalidRequestError` | Malformed content part (bad data URL, unsupported image type, oversized payload, image outside a `user` message) or malformed `response_format` |
| `401` | `AuthError` | Missing or invalid API key |
| `429` | `RateLimitError` | Gemini quota exceeded |
| `5xx` | `ServerError` | Gemini upstream error |

---

## Project Structure

```
gemini-fri/
├── gemini_live_text.py          # Standalone CLI demo (text only, independent of sdk/)
├── main.py                      # FastAPI app, OpenAI route, exception handlers
├── requirements.txt
├── pyproject.toml
├── routers/
│   └── ollama.py                # Ollama-compatible endpoints
└── sdk/
    ├── __init__.py              # OpenAICompatClient
    ├── types.py                 # MessageRole, FinishReason enums
    ├── core/
    │   ├── models.py            # OpenAI pydantic schemas (request / response / tool call)
    │   ├── ollama_models.py     # Ollama pydantic schemas
    │   ├── content.py           # OpenAI content → Gemini parts (text + images), mime sniffing
    │   └── exceptions.py        # SDKError, APIError, InvalidRequestError, AuthError, RateLimitError, ServerError
    ├── providers/
    │   └── gemini_live.py       # Gemini Live API client (chat_once, chat_stream, chat_once_ex)
    └── resources/
        ├── chat/
        │   └── completions.py   # ChatCompletions.create(), retry, tools, response_format
        └── ollama/
            └── chat.py          # Ollama ⇄ OpenAI translation
```

### How it works

```
Client (OpenAI SDK / curl)          Client (Ollama SDK / Open WebUI)
        │                                    │
        ▼                                    ▼
FastAPI  main.py                     routers/ollama.py
        │  auth check, routing               │  auth (header → env), NDJSON, Ollama errors
        │                                    ▼
        │                            sdk/resources/ollama/chat.py
        │                                    │  rewrites into ChatCompletionRequest
        ├────────────────────────────────────┘
        ▼
sdk/core/content.py
        │  converts OpenAI messages → (system_prompt, list[Part])
        │  validates + decodes base64 images → inline_data parts
        ▼
sdk/resources/chat/completions.py
        │  converts OpenAI tools → Gemini FunctionDeclaration
        │  wraps calls with tenacity retry (3 attempts, exp backoff)
        ▼
sdk/providers/gemini_live.py
        │  chat_once / chat_stream / chat_once_ex
        │  sends one turn via send_client_content
        │  uses AUDIO modality + output_audio_transcription
        ▼
Google Gemini Live API  (model: gemini-3.1-flash-live-preview)
```

> **Why `send_client_content`:** the Live API also accepts `send_realtime_input`, but that path gives no ordering guarantee, and images carry enough preprocessing cost that a following text message can reach the model first. `send_client_content` sends text and images as one ordered turn.

> **Note on audio transcription:** The Gemini Live API session is opened with `response_modalities=["AUDIO"]` and `output_audio_transcription` enabled. Text is extracted from the transcription, not from a text-mode response. This is a quirk of the Live API's current interface.

---

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License.

---

## Disclaimer

> **This project is created solely for learning and educational purposes.**
> It is not intended for production use, commercial deployment, or any mission-critical application.
>
> - This is an **unofficial** wrapper and is **not affiliated with, endorsed by, or supported by Google**.
> - The author(s) take **no responsibility** for any issues arising from the use of this software, including but not limited to API quota charges, data loss, or service interruptions.
> - Always review [Google's Gemini API Terms of Service](https://ai.google.dev/gemini-api/terms) before use.
> - **Never commit your API keys to version control.**

---

Made with love for learning
