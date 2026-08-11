# Hỗ trợ ảnh đầu vào (vision) cho endpoint OpenAI-compatible

Ngày: 2026-08-11

## Bối cảnh

`POST /openai/v1/chat/completions` hiện chỉ xử lý được text. Trường `Message.content`
đã khai báo là `Optional[Union[str, List[dict]]]`, nhưng khi client gửi content dạng
list (định dạng multimodal của OpenAI), `_build_gemini_prompt()` trong
`sdk/resources/chat/completions.py` gọi `str(msg.content)` — tức là dump nguyên cái
Python dict, kể cả chuỗi base64, vào prompt text. Model không hề nhìn thấy ảnh.

Sau đó `gemini_live_text.py` gửi prompt bằng `session.send_realtime_input(text=...)`,
một đường truyền chỉ nhận text.

## Live API có xử lý được ảnh không?

Có. Xác minh trực tiếp trên `google-genai 2.8.0` đã cài trong `.venv`:

| Cách gửi | Chữ ký | Đặc tính |
|---|---|---|
| `send_realtime_input(media=...)` | `PIL.Image` hoặc `types.Blob(data=..., mime_type="image/jpeg")` | Tối ưu độ trễ, **không đảm bảo thứ tự** |
| `send_client_content(turns=Content(parts=[...]))` | `Part(inline_data=Blob(...))` + `Part(text=...)` | Turn-based, **đúng thứ tự** |

Docstring của `send_client_content` trong SDK nói rõ nó vượt trội khi gửi "objects that
have significant preprocessing time (typically images)", và cảnh báo không nên trộn hai
kiểu gửi trong cùng một cuộc hội thoại.

Hai điểm đã xác minh bằng cách chạy code thật:

1. Converter `_Part_to_mldev` map `inline_data` → `inlineData` cho đường Gemini API
   (mldev, không phải Vertex), nên `send_client_content` đẩy được ảnh lên.
2. `send_client_content` serialize `Blob.data` (bytes) thành chuỗi base64 URL-safe.
   Đặc tả proto3 JSON chấp nhận cả base64 chuẩn lẫn URL-safe, có hay không có padding,
   nên dữ liệu nhị phân đi qua nguyên vẹn.

Giới hạn phía Gemini:

- MIME types nhận được: `image/png`, `image/jpeg`, `image/webp`, `image/heic`, `image/heif`
- Inline data: tổng request (text + system instruction + bytes) phải dưới 20 MB

## Phạm vi

Đã chốt với người dùng:

- **Chỉ nhận base64 `data:` URL.** Không tải ảnh từ URL http(s) từ xa — tránh độ trễ
  mạng và rủi ro SSRF.
- **Gửi tất cả ảnh trong toàn bộ history**, không chỉ ảnh ở lượt cuối. Mỗi request mở
  một Live session mới nên không có state phía server để tận dụng.
- **Chỉ sửa code trong `sdk/`.** File `gemini_live_text.py` giữ nguyên, tiếp tục chạy
  độc lập như CLI demo. `main.py` và `pyproject.toml` cũng không đổi. Ràng buộc này áp
  cho code; `README.md` là tài liệu nên vẫn được cập nhật.
- **Ảnh không hợp lệ → trả lỗi 400 ngay**, không âm thầm bỏ qua.
- **Không viết test tự động.** Repo chưa có test suite và người dùng chọn không thêm.

Nằm ngoài phạm vi: audio input, video input, Files API cho ảnh lớn, trường `detail`
của OpenAI (low/high/auto), và việc tái cấu trúc đường tool calling.

## Hướng tiếp cận

Đã cân nhắc ba hướng:

**A (chọn) — `send_client_content` với danh sách Part theo đúng thứ tự, gói trong một
user turn.** Đổi hàm dựng prompt từ trả về `str` sang trả về `list[Part]`.

**B — Dựng `list[Content]` nhiều lượt đúng chuẩn** (role `user`/`model`, parts
`function_call`/`function_response`). Đúng ngữ nghĩa nhất nhưng phải viết lại cả đường
tool calling vốn vừa được sửa ở commit `8945a58`. Rủi ro hồi quy cao, không phục vụ
yêu cầu hiện tại.

**C — `send_realtime_input(media=Blob)` cho từng ảnh rồi `send_realtime_input(text=...)`.**
Loại, vì SDK không đảm bảo thứ tự và ảnh có thời gian tiền xử lý đáng kể, nên text dễ
tới model trước ảnh.

Lý do chọn A: khi request không có ảnh, nó sinh ra kết quả giống hệt hành vi hiện tại,
nên tool calling và multi-turn text không thể hồi quy; đồng thời ảnh nằm đúng vị trí nó
xuất hiện trong hội thoại.

## Kiến trúc

### Bố cục file

| File | Trạng thái |
|---|---|
| `sdk/core/content.py` | mới — chuyển OpenAI content sang Gemini Parts, kèm validate |
| `sdk/providers/__init__.py` | mới |
| `sdk/providers/gemini_live.py` | mới — Live client riêng của sdk, nhận `list[Part]` |
| `sdk/core/exceptions.py` | sửa — thêm `InvalidRequestError` |
| `sdk/resources/chat/completions.py` | sửa — dùng provider mới, bỏ import từ `gemini_live_text` |
| `sdk/core/models.py` | không đổi |
| `main.py` | không đổi |
| `pyproject.toml` | không đổi |
| `gemini_live_text.py` | không đổi |
| `README.md` | sửa — tài liệu |

Hiện `completions.py` import `chat_once`, `chat_once_ex`, `chat_stream`,
`DEFAULT_API_KEY`, `DEFAULT_SYSTEM_PROMPT` từ `gemini_live_text.py` ở root. Vì không
được sửa file đó, `sdk/` phải tự sở hữu Live client của mình. Hệ quả chấp nhận được:
các hằng `MODEL`, `DEFAULT_SYSTEM_PROMPT`, `DEFAULT_API_KEY` tồn tại hai bản — một
trong `sdk/providers/gemini_live.py`, một trong file CLI demo.

### `sdk/core/content.py`

Hằng số:

```python
SUPPORTED_IMAGE_MIME_TYPES = {
    "image/png", "image/jpeg", "image/webp", "image/heic", "image/heif",
}
MAX_TOTAL_IMAGE_BYTES = 15 * 1024 * 1024
```

Ngưỡng 15 MB thấp hơn giới hạn 20 MB của Gemini để chừa chỗ cho text, system
instruction và overhead của base64 trên đường truyền.

Hai hàm public:

```python
def parse_data_url(url: str) -> tuple[str, bytes]
def build_parts(messages: List[Message]) -> tuple[str, list[types.Part]]
```

`parse_data_url` khớp chuỗi với `data:<mime>;base64,<payload>`, decode base64, và
raise `InvalidRequestError` khi định dạng sai, base64 hỏng, hoặc MIME không nằm trong
tập hỗ trợ.

`build_parts` thay thế `_build_gemini_prompt`, trả về `(system_prompt, parts)`.

Thuật toán, chia hai giai đoạn để giữ nguyên ngữ nghĩa nối chuỗi hiện tại:

1. Duyệt messages, sinh một danh sách token phẳng. Mỗi token là `("text", str)` hoặc
   `("image", mime, data)`. Giữa hai message liên tiếp chèn một token `("text", "\n")`,
   tương ứng với `"\n".join(conversation_parts)` hôm nay.
2. Gộp các token text liền kề thành một `types.Part(text=...)` duy nhất; mỗi token
   image thành `types.Part(inline_data=types.Blob(data=..., mime_type=...))`.

Tính chất then chốt: **request không chứa ảnh sẽ sinh ra đúng một `Part(text=...)`, nội
dung giống hệt chuỗi mà code hiện tại tạo ra.** Đây là điều làm cho thay đổi này an
toàn với các luồng đang chạy.

Quy tắc theo role — giữ nguyên toàn bộ tiền tố và cách nối chuỗi hiện có
(`"User: "`, `"Assistant: "`, `"Assistant called: "`, `"Tool result [id]: "`):

- `system` — gom vào `system_prompt`. Content dạng list thì chỉ lấy các phần tử `text`.
- `user` — content `str` sinh một token text `"User: {content}"`. Content dạng list thì
  duyệt từng phần tử theo thứ tự:
  - `{"type": "text", "text": ...}` → token text
  - `{"type": "image_url", "image_url": {"url": ...}}` → gọi `parse_data_url`, sinh
    token image. Trường `detail` nếu có thì bỏ qua.
  - type khác → `InvalidRequestError`

  Tiền tố `"User: "` phát ra thành một token text riêng trước khi duyệt các phần tử,
  bất kể phần tử đầu tiên là text hay ảnh.
- `assistant`, `tool` — giữ nguyên logic hiện tại.
- Content không phải `str` cũng không phải `list` (ví dụ `None` ở role `user`) — rơi về
  `str(content)` đúng như code hiện tại, kể cả khi kết quả là chuỗi `"None"`. Đây là
  hành vi có sẵn, không sửa trong lần này.
- Phần tử `image_url` xuất hiện ở role khác `user` → `InvalidRequestError`.

Tổng số byte ảnh đã decode được cộng dồn trong suốt `build_parts`; vượt
`MAX_TOTAL_IMAGE_BYTES` thì raise `InvalidRequestError`.

Nếu không có message nào sinh ra token, `parts` là một `Part(text="")` duy nhất, khớp
với hành vi chuỗi rỗng hiện tại.

### `sdk/providers/gemini_live.py`

Ba hàm gương với `gemini_live_text.py`: `chat_once`, `chat_stream`, `chat_once_ex`.
Khác đúng hai điểm:

- Tham số đầu là `parts: list[types.Part]` thay cho `message: str`.
- Gửi bằng:

  ```python
  await session.send_client_content(
      turns=types.Content(role="user", parts=parts),
      turn_complete=True,
  )
  ```

  thay cho `session.send_realtime_input(text=message)`.

Mọi thứ còn lại giữ y hệt bản gốc: `http_options=types.HttpOptions(api_version="v1alpha")`,
`response_modalities=["AUDIO"]`, `output_audio_transcription=types.AudioTranscriptionConfig()`,
`system_instruction`, `generation_config` dựng qua `_make_generation_config`, `tools`,
vòng lặp `session.receive()`, cách đọc `server_content.output_transcription`, điều kiện
thoát `turn_complete`, và cách gom `response.tool_call.function_calls` trong
`chat_once_ex`.

Module cũng định nghĩa `MODEL`, `DEFAULT_SYSTEM_PROMPT`, `DEFAULT_API_KEY` để
`completions.py` không còn phụ thuộc vào `gemini_live_text.py`.

### `sdk/core/exceptions.py`

```python
class InvalidRequestError(APIError):
    def __init__(self, message: str):
        super().__init__(400, message)
```

Starlette tra exception handler bằng cách duyệt `type(exc).__mro__`, nên handler
`@app.exception_handler(APIError)` sẵn có trong `main.py` sẽ bắt được lớp con này và
trả về status 400 — không cần sửa `main.py`.

Đánh đổi đã được người dùng chấp nhận: trường `"type"` trong JSON lỗi sẽ là
`"api_error"` thay vì `"invalid_request_error"` như đặc tả OpenAI. Status code và
message vẫn đúng.

### `sdk/resources/chat/completions.py`

- Import từ `sdk.providers.gemini_live` thay vì `gemini_live_text`.
- `create()` gọi `build_parts(request.messages)` thay cho `_build_gemini_prompt(...)`,
  rồi truyền `parts` xuống `_complete()` và `_stream()`.
- `_complete()` và `_stream()` đổi tham số `user_message: str` thành `parts: list[Part]`.
- Xoá `_build_gemini_prompt` (đã được `build_parts` thay thế).
- `_convert_tools` giữ nguyên.

Ước lượng token cho `TokenBucket`: `_estimate_tokens` hiện tính `len(text) // 4`. Mở
rộng để nhận `parts` — cộng `len(part.text) // 4` cho mỗi text part, và
`max(258, len(data) // 750)` cho mỗi image part. Con số 258 là chi phí token của một
tile 768×768 theo tài liệu Gemini. Đây là ước lượng thô, cùng mức độ thô như heuristic
text sẵn có, nhưng bỏ qua hẳn ảnh sẽ khiến token bucket 65k TPM đếm hụt nghiêm trọng.

Trường `usage` trong response hiện đếm bằng `len(text.split())`. Với parts, dùng tổng
số từ của các text part; ảnh không cộng vào. Đây vốn đã là con số xấp xỉ và không nằm
trong phạm vi cải thiện lần này.

## Luồng dữ liệu

```
POST /openai/v1/chat/completions
  → ChatCompletionRequest (messages: content str hoặc list[dict])
  → build_parts()            ── lỗi validate ──→ InvalidRequestError → HTTP 400
  → (system_prompt, list[Part])
  → TokenBucket.consume(ước lượng token của parts)
  → providers.gemini_live.chat_once / chat_once_ex / chat_stream
  → session.send_client_content(Content(role="user", parts=parts))
  → đọc output_transcription cho tới turn_complete
  → ChatCompletionResponse hoặc các SSE chunk
```

## Xử lý lỗi

Mọi lỗi validate xảy ra **trước khi** mở Live session, nên request hỏng không tốn quota.

| Tình huống | Kết quả |
|---|---|
| `image_url.url` không phải `data:` URL | 400 |
| Thiếu `;base64,` hoặc sai cấu trúc data URL | 400 |
| Payload base64 hỏng | 400 |
| MIME ngoài tập 5 loại Gemini hỗ trợ | 400 |
| Tổng byte ảnh vượt 15 MB | 400 |
| Content part có `type` lạ | 400 |
| `image_url` ở role khác `user` | 400 |

Message lỗi nêu rõ vị trí (chỉ số message, chỉ số part) và nguyên nhân.

Logic retry và dịch exception (`_translate_exc`, `_RETRY_CONFIG`) không đổi.
`InvalidRequestError` kế thừa `APIError` chứ không kế thừa `AuthError`, nên về mặt kỹ
thuật nó nằm trong tập được retry — nhưng vì nó được raise trước khi vào khối
`AsyncRetrying`, điều đó không có hiệu lực thực tế.

## Kiểm chứng

Không có test tự động (người dùng đã chọn). Việc kiểm chứng dựa vào:

1. **Không hồi quy đường text**: đọc lại `build_parts` và xác nhận với input không có
   ảnh, chuỗi text sinh ra khớp từng ký tự với `_build_gemini_prompt` cũ. Chạy thử một
   request text-only và một request tool-calling trên server đang chạy.
2. **Smoke test thủ công có ảnh**: người dùng chạy với API key thật và một file PNG nhỏ
   encode base64, kiểm tra model mô tả đúng nội dung ảnh.

Điểm chưa xác minh được: model `gemini-3.1-flash-live-preview` có thực sự nhận ảnh hay
không. Môi trường phát triển không có API key nên chỉ smoke test ở bước 2 mới trả lời
được. Nếu model này từ chối ảnh, phương án là đổi sang một model Live khác có hỗ trợ
vision — thay đổi chỉ nằm ở hằng `MODEL` trong `sdk/providers/gemini_live.py`.

## Tài liệu

Cập nhật `README.md`: thêm ảnh vào danh sách tính năng và một ví dụ `curl` gửi ảnh
base64, kèm ghi chú về các MIME được hỗ trợ và giới hạn 15 MB.
