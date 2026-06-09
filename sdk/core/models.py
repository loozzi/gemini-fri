from typing import List, Literal, Optional, Union
from pydantic import BaseModel


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
