# Mặt API tương thích Ollama

Ngày: 2026-08-11

## Mục tiêu

Phơi thêm các endpoint theo giao thức Ollama bên cạnh `/openai/v1` sẵn có, vẫn dùng
Gemini Live làm backend. Để các công cụ chỉ nói được giao thức Ollama — Open WebUI,
LangChain `ChatOllama`, Continue — dùng được Gemini mà không phải đổi code.

Đây là **mặt API**, không phải provider: server không gọi tới Ollama nào cả.

## Phạm vi

Đã chốt với người dùng:

- Endpoint: `/api/chat`, `/api/tags`, `/api/show`, `/api/version`. Bỏ `/api/generate`
  vì các client hiện đại đều dùng `/api/chat`.
- API key: `Authorization: Bearer` trước, không có thì lấy `GEMINI_API_KEY` từ env,
  thiếu cả hai thì 401.
- Có dịch tool calling cả hai chiều.
- Route OpenAI giữ nguyên trong `main.py`; chỉ thêm `routers/ollama.py`.

Ngoài phạm vi: `/api/embeddings` (Live API không có embedding), `/api/pull`,
`/api/push`, `/api/create`, `/api/delete`, `/api/copy`, `/api/ps` — đều là quản lý
model cục bộ, không có ý nghĩa ở đây.

## Kiến trúc

Mặt Ollama **không nói chuyện thẳng với Gemini**. Nó dịch request sang
`ChatCompletionRequest`, gọi `ChatCompletions.create()`, rồi dịch kết quả ngược lại.

```
Ollama client
      │  POST /api/chat
      ▼
routers/ollama.py                 auth, hình dạng lỗi kiểu Ollama, NDJSON
      ▼
sdk/resources/ollama/chat.py      dịch Ollama ⇄ OpenAI
      ▼
sdk/resources/chat/completions.py ĐÃ CÓ — ảnh, tools, response_format, retry, token bucket
      ▼
sdk/providers/gemini_live.py      ĐÃ CÓ
      ▼
Gemini Live API
```

Nhờ vậy toàn bộ phần ảnh, tool calling, `response_format`, retry và token bucket
được dùng lại nguyên vẹn. Không có đường code song song phải bảo trì gấp đôi, và
mọi cải tiến ở lõi tự động có hiệu lực cho cả hai mặt.

## Dịch request

`POST /api/chat`:

| Ollama | → OpenAI |
|---|---|
| `model` | `model` (truyền qua, backend luôn dùng hằng `MODEL`) |
| `messages[].role` / `.content` | như cũ |
| `messages[].images[]` | base64 thô → sniff mime → data URL → content part `image_url` |
| `messages[].tool_calls[].function.arguments` (object) | `arguments` (chuỗi JSON) |
| `messages[].tool_name` | `name` |
| `tools[]` | `tools[]` (cùng hình dạng) |
| `format: "json"` | `response_format: {"type": "json_object"}` |
| `format: {…schema…}` | `response_format: {"type": "json_schema", "json_schema": {"name": "response", "schema": …}}` |
| `options.temperature` | `temperature` |
| `options.top_p` | `top_p` |
| `options.num_predict` | `max_tokens` |
| `stream` | `stream` — **mặc định `true`**, ngược với OpenAI |
| `keep_alive`, `think`, `raw`, `context` | bỏ qua |

### Sniff mime cho ảnh

Ollama gửi base64 **trần**, không kèm mime type, trong khi `sdk/core/content.py` đòi
data URL đầy đủ. Thêm `sniff_image_mime(data: bytes) -> Optional[str]` vào
`content.py`, nhận dạng bằng magic bytes:

| Định dạng | Magic bytes |
|---|---|
| `image/png` | `89 50 4E 47 0D 0A 1A 0A` |
| `image/jpeg` | `FF D8 FF` |
| `image/webp` | `RIFF` ở offset 0 và `WEBP` ở offset 8 |
| `image/heic` / `image/heif` | `ftyp` ở offset 4, brand `heic`/`heix`/`heif`/`mif1` |

Không nhận ra được → 400 với thông báo rõ ràng, cùng cách xử lý như data URL hỏng.

## Dịch response

NDJSON (`application/x-ndjson`), mỗi dòng một object JSON, không có tiền tố `data:`,
không có `[DONE]`.

Chunk giữa chừng:

```json
{"model":"…","created_at":"…","message":{"role":"assistant","content":"The"},"done":false}
```

Chunk cuối:

```json
{"model":"…","created_at":"…","message":{"role":"assistant","content":""},
 "done":true,"done_reason":"stop",
 "total_duration":…,"load_duration":0,
 "prompt_eval_count":…,"prompt_eval_duration":…,
 "eval_count":…,"eval_duration":…}
```

Các trường thời lượng tính bằng **nanosecond**. `total_duration` đo bằng đồng hồ
thực; `load_duration` là 0 vì không có model nào được nạp. `prompt_eval_count` và
`eval_count` lấy từ `usage` của response OpenAI — vốn đã là ước lượng theo số từ.

Không streaming: đúng object cuối đó, kèm `message.content` đầy đủ, trả một lần.

`tool_calls` đổi ngược: `arguments` từ chuỗi JSON về object, và bỏ trường `id`
(Ollama không có).

## Endpoint discovery

`GET /api/tags` — quảng cáo một model duy nhất, tên theo model thật ở backend:

```json
{"models":[{"name":"gemini-3.1-flash-live-preview:latest",
            "model":"gemini-3.1-flash-live-preview:latest",
            "modified_at":"…","size":0,"digest":"",
            "details":{"parent_model":"","format":"gguf","family":"gemini",
                       "families":["gemini"],"parameter_size":"","quantization_level":""}}]}
```

`size: 0` và `digest: ""` là cố ý: server không có file model thật, nên không bịa ra
hash và dung lượng giả. Nếu gặp client nào bắt buộc phải có, sẽ chỉnh sau.

`POST /api/show` — trả `details` như trên cộng
`capabilities: ["completion", "vision", "tools"]`. Open WebUI đọc `vision` để quyết
định có bật nút upload ảnh hay không. `modelfile`, `template`, `parameters` để rỗng.

`GET /api/version` — một hằng số tương thích, không phải phiên bản Ollama thật đang
chạy ở đâu đó. Đặt trong `routers/ollama.py` để dễ sửa khi có client gate theo
version.

## Auth và lỗi

Key theo thứ tự: header `Authorization: Bearer` → env `GEMINI_API_KEY` → 401.

Điều này nới lỏng tính chất "stateless, never stores keys" mà README quảng cáo —
nhưng phần lớn client Ollama không cho cấu hình header, nên bắt buộc header sẽ khiến
tính năng vô dụng với đúng những công cụ nó nhắm tới. README cần nói rõ đánh đổi này.

Lỗi trả theo hình dạng của Ollama:

```json
{"error": "thông báo"}
```

Xử lý ngay trong router bằng `try/except`, không đụng vào các exception handler chung
đang phục vụ mặt OpenAI — chúng trả hình dạng lỗi của OpenAI và phải giữ nguyên.

Ánh xạ status: `AuthError` → 401, `InvalidRequestError` → 400, `RateLimitError` → 429,
còn lại → 500.

## Kiểm chứng

Không có test tự động (theo lựa chọn trước đó của người dùng). Kiểm chứng bằng script
tạm trong scratchpad, không thêm file vào repo:

1. Dịch request cả hai chiều với provider giả: ảnh, tools, `format`, `options`,
   mặc định `stream`.
2. Hình dạng NDJSON đúng từng trường, kể cả chunk cuối và các trường nanosecond.
3. Bốn endpoint qua `TestClient`.
4. End-to-end với API key thật: chat thường, chat kèm ảnh, và `format` schema.

Đường OpenAI sẵn có phải vẫn xanh — ba bộ kiểm chứng cũ chạy lại sau khi sửa.
