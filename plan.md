# 🛠️ Build OpenAI-Compatible SDK (Python)

## Mục tiêu

Xây dựng một Python SDK client dùng để gọi tới bất kỳ backend nào hỗ trợ OpenAI-compatible API (route `/openai/v1/chat/completions`). SDK phải hỗ trợ cả **non-stream** và **streaming** response, schema chuẩn theo spec của OpenAI.

---

## Yêu cầu kỹ thuật

### Stack

- Python 3.11+
- `httpx` (async HTTP client)
- `pydantic` v2 (schema validation)
- `asyncio` (async/await)

### Tính năng bắt buộc

- [x] Gọi `POST /openai/v1/chat/completions`
- [x] Hỗ trợ `stream=False` → trả về `ChatCompletionResponse`
- [x] Hỗ trợ `stream=True` → trả về `AsyncIterator[dict]` (SSE parsing)
- [x] Header `Authorization: Bearer <api_key>`
- [x] Timeout configurable
- [x] Raise exception rõ ràng khi HTTP error (4xx/5xx)

### Tính năng optional (nice to have)

- [ ] Retry với exponential backoff
- [ ] Logging middleware (log request/response)
- [ ] Support `tools` / `tool_choice` (function calling)

---

## Cấu trúc thư mục

```
sdk/
├── core/
│   ├── __init__.py
│   ├── client.py          # BaseClient với _post() và _stream()
│   ├── models.py          # Pydantic schemas: Request, Response, Choice, Usage
│   └── exceptions.py      # APIError, AuthError, RateLimitError
├── resources/
│   └── chat/
│       ├── __init__.py
│       └── completions.py  # ChatCompletions.create()
├── __init__.py             # Export OpenAICompatClient
└── types.py               # Shared types (MessageRole, FinishReason...)
```

---

## Chi tiết từng file

### `core/models.py`

Định nghĩa các Pydantic models sau:

```python
# Request
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[dict]]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None
    user: Optional[str] = None

# Response
class ChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class Choice(BaseModel):
    index: int
    message: Optional[Message] = None
    delta: Optional[ChoiceDelta] = None
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
```

---

### `core/client.py`

```python
class BaseClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 60.0):
        ...

    def _headers(self) -> dict:
        # Trả về {"Content-Type": "application/json", "Authorization": "Bearer <key>"}
        ...

    async def _post(self, path: str, payload: dict) -> dict:
        # httpx.AsyncClient POST, raise_for_status(), return resp.json()
        ...

    async def _stream(self, path: str, payload: dict) -> AsyncIterator[str]:
        # httpx stream POST
        # Parse từng line SSE: line bắt đầu bằng "data: "
        # Skip "[DONE]"
        # yield raw JSON string
        ...
```

---

### `core/exceptions.py`

```python
class SDKError(Exception): ...
class APIError(SDKError): 
    def __init__(self, status_code: int, message: str): ...
class AuthError(APIError): ...       # 401
class RateLimitError(APIError): ...  # 429
class ServerError(APIError): ...     # 5xx
```

Map HTTP status code → exception tương ứng trong `_post()` và `_stream()`.

---

### `resources/chat/completions.py`

```python
class ChatCompletions:
    def __init__(self, client: BaseClient): ...

    async def create(self, **kwargs) -> Union[ChatCompletionResponse, AsyncIterator[dict]]:
        # Build ChatCompletionRequest từ kwargs
        # Nếu stream=True → gọi _stream() → yield parsed JSON chunk
        # Nếu stream=False → gọi _post() → return ChatCompletionResponse
        ...
```

---

### `__init__.py` — Entrypoint

```python
class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str = None, timeout: float = 60.0):
        self._http = BaseClient(base_url=base_url, api_key=api_key, timeout=timeout)
        self.chat = _ChatNamespace(self._http)

class _ChatNamespace:
    def __init__(self, client: BaseClient):
        self.completions = ChatCompletions(client)
```

---

## Ví dụ usage (dùng để test)

```python
# examples/basic.py
import asyncio
from sdk import OpenAICompatClient

client = OpenAICompatClient(
    base_url="https://your-api.com",
    api_key="sk-xxx"
)

async def test_non_stream():
    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(resp.choices[0].message.content)

async def test_stream():
    async for chunk in await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True,
    ):
        delta = chunk["choices"][0]["delta"].get("content", "")
        print(delta, end="", flush=True)

asyncio.run(test_non_stream())
asyncio.run(test_stream())
```

---

## Yêu cầu bổ sung

1. Tất cả methods đều phải là `async`
2. Không dùng `requests` (sync), chỉ dùng `httpx`
3. Pydantic model dùng `model_dump(exclude_none=True)` khi serialize payload
4. SSE parser phải handle được các edge case:
   - Line trống → skip
   - `data: [DONE]` → stop iteration
   - JSON parse error → raise `SDKError`
5. Viết `requirements.txt`:
   ```
   httpx>=0.27.0
   pydantic>=2.0.0
   ```

---

## Định nghĩa "done"

- [ ] Chạy được `examples/basic.py` với cả 2 mode stream/non-stream
- [ ] Nếu sai API key → raise `AuthError`
- [ ] Nếu server 500 → raise `ServerError`
- [ ] Type hints đầy đủ toàn bộ codebase