from typing import Literal

from pydantic import BaseModel, Field


# One earlier message of the conversation, sent by Laravel
class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


# Chat request: the new message and the recent conversation
class ChatRequest(BaseModel):
    message: str
    history: list[ChatHistoryMessage] = Field(default_factory=list)
