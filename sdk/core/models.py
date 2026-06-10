from typing import List, Literal, Optional, Union
from pydantic import BaseModel


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON-encoded string


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[dict]]] = None  # None when tool_calls present
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


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


class ChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChoiceDeltaToolCall(BaseModel):
    index: int
    id: Optional[str] = None
    type: Optional[str] = "function"
    function: Optional[ToolCallFunction] = None


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
