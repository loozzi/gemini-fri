# Structured output (`response_format`) trên Gemini Live API

Ngày: 2026-08-11

## Bối cảnh

Client gửi `response_format` — qua `client.beta.chat.completions.parse()` hoặc
`llm.with_structured_output(Model)` của LangChain — nhận về lỗi:

```
pydantic_core.ValidationError: 1 validation error for EmailContent
  Invalid JSON: expected value at line 1 column 1
  [type=json_invalid, input_value='Dưới đây là thông...-tuyen-dung từ ảnh:', input_type=str]
```

Nguyên nhân gốc: `ChatCompletionRequest` không khai báo `response_format`, và
pydantic mặc định `extra="ignore"`, nên trường này bị **vứt lặng lẽ**. Server trả
200 như không có gì xảy ra, schema không bao giờ tới Gemini, model trả văn xuôi,
rồi OpenAI SDK cố parse văn xuôi thành JSON.

Đây là lỗi thuộc loại tệ nhất: thất bại im lặng ở ranh giới request.

LangChain 1.4.2 mặc định `method="json_schema"` cho `with_structured_output`, nên
mọi người dùng LangChain đều đâm vào lỗi này ngay lần đầu.

## Gemini Live API hỗ trợ tới đâu

Thăm dò trực tiếp với API key thật trên `gemini-3.1-flash-live-preview`:

| Thử | Kết quả |
|---|---|
| `response_modalities=["TEXT"]` | **Từ chối** — `1007: The requested combination of response modalities (TEXT) is not supported by the model` |
| `response_schema` trong `GenerationConfig` (TEXT) | **Từ chối** — `1007: response_schema not supported in generation config` |
| `response_schema` trong `GenerationConfig` (AUDIO) | **Từ chối** — cùng lỗi |
| `response_mime_type="application/json"` (TEXT) | **Từ chối** — vì TEXT không được hỗ trợ |
| AUDIO + `output_audio_transcription` (config hiện tại) | Chạy, trả văn xuôi |

Hai kết luận quan trọng:

1. **Modality AUDIO không phải lựa chọn tuỳ tiện của dự án** — model này không cho
   dùng TEXT. Đừng tốn công "sửa" chỗ đó.
2. **Live API không có structured output native.** `LiveConnectConfig` không có
   trường `response_schema`; converter `_LiveConnectConfig_to_mldev` có copy nguyên
   khối `generation_config` nên tham số *được gửi đi*, nhưng server Gemini từ chối
   thẳng.

Hai đường còn lại, cả hai đều đã thử và đều chạy:

| Đường | Kết quả thăm dò |
|---|---|
| Nhét schema vào system prompt, đòi JSON thuần | Transcript ra `{"city": "Hà Nội", "population": 8435700}` — parse được. Transcription **giữ nguyên** dấu `{`, `"`, `:` |
| Khai schema thành `FunctionDeclaration`, bảo model gọi tool | `tool_calls: [{'name': 'EmailContent', 'args': {'population': 8400000, 'city': 'Hà Nội'}}]` |

Chọn đường tool calling: args về dưới dạng dữ liệu có cấu trúc sẵn, không phụ thuộc
vào việc model có chịu nhịn không viết lời dẫn hay không, và tái dùng đường tool
calling đã chạy ổn từ commit `8945a58`.

Lưu ý: `LiveConnectConfig` **không có** `tool_config`, nên không ép được
`function_calling_config.mode="ANY"`. Phải lái model bằng system instruction.

## Thiết kế

### Luồng

```
response_format = {"type": "json_schema", "json_schema": {"name": N, "schema": S}}
        │
        ├─ S → FunctionDeclaration(name=N, parameters=S)  → thêm vào tools
        └─ system_prompt += "Answer only by calling the function N..."
        ▼
Gemini trả tool_call(name=N, args={...})
        ▼
message.content = json.dumps(args)   ·   finish_reason = "stop"
```

Client thấy một completion văn bản bình thường mà nội dung là JSON — đúng như OpenAI
structured output, nên `.parsed` và `with_structured_output` hoạt động.

### Thứ tự ưu tiên khi dựng response

1. Có tool call trùng tên schema → dùng args của nó (đường chính)
2. Có tool call khác (client cũng gửi `tools` thật) → trả `tool_calls` như bình thường
3. Không có tool call nào → thử parse text thành JSON, chấp nhận cả markdown fence
4. Vẫn không được → trả nguyên text kèm log warning

Bước 3 tồn tại vì thăm dò cho thấy model đôi khi bỏ qua tool mà trả thẳng JSON.
Bước 4 chọn trả text thay vì ném 500: client vẫn còn cơ hội xử lý, và log nói rõ
chuyện gì đã xảy ra.

### `json_object`

`{"type": "json_object"}` không kèm schema nên không dựng được tool. Dùng đường
prompt: thêm chỉ thị đòi JSON thuần, rồi parse text (bóc fence nếu có).

### Streaming

Structured output cần trọn câu trả lời mới validate được, nên `stream=true` kèm
`response_format` sẽ chạy đường non-streaming rồi phát lại kết quả thành ba chunk
(role → content → finish). Thà trả một chunk to còn hơn lặng lẽ bỏ qua
`response_format` như trước.

### Chuyển đổi JSON Schema

Converter cũ (`convert_schema` lồng trong `_convert_tools`) được nâng lên mức module
thành `_convert_schema` / `_object_schema`, dùng chung cho cả `tools` và
`response_format`, và xử lý thêm ba dạng mà Pydantic sinh ra:

- `$ref` / `$defs` → inline (giới hạn độ sâu 16 để không lặp vô hạn với schema đệ quy)
- `anyOf: [T, null]` (tức `Optional[T]`) → lấy nhánh không phải null; Gemini không có union
- `type: ["integer", "null"]` → lấy phần tử không phải null

Trước đây `Optional[T]` rơi vào nhánh mặc định và thành `STRING`.

### Validation

`response_format` sai cấu trúc → `InvalidRequestError` (400), ném trước khi mở Live
session: `type` lạ, thiếu `json_schema`, thiếu `name`, thiếu `schema`.

## Kiểm chứng

37 kiểm tra offline với provider giả (chuyển đổi schema, dựng tool, hình dạng
response, ba mức fallback, chế độ `json_object`, streaming, bốn ca 400), cộng ba bộ
kiểm chứng có sẵn của tính năng ảnh vẫn xanh.

End-to-end với API key thật và ảnh chụp email thật: `method="json_schema"` — chính
cái trước đây ném `ValidationError` — nay trích xuất đúng `sender`, `recipients`,
`body`, `links`. `method="function_calling"` cũng vẫn chạy.

## Giới hạn đã biết

Không có bảo đảm cứng như strict mode của OpenAI. Model được *hướng dẫn* gọi tool
chứ không bị *ép* — Live API không cho set `tool_config`. Ba mức fallback giảm rủi
ro nhưng không xoá được nó.
