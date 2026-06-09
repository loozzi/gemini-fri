from enum import Enum


class MessageRole(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class FinishReason(str, Enum):
    stop = "stop"
    length = "length"
    tool_calls = "tool_calls"
    content_filter = "content_filter"
