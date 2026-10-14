"""Offline stand-ins: a chat model that plays back a script, and message builders."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """Returns or raises the next scripted item on each call, and records the messages it was sent."""

    script: list[Any]
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        if not self.script:
            raise AssertionError("scripted model ran out of responses")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])


def ai(
    content: str = "",
    *,
    tool_calls: list[dict] | None = None,
    model: str = "fake-model",
    tokens: tuple[int, int] = (100, 50),
) -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=tool_calls or [],
        response_metadata={"model_name": model},
        usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": sum(tokens)},
    )


def tool_call(name: str, args: dict, call_id: str | None = None) -> dict:
    return {"name": name, "args": args, "id": call_id or f"call_{name}", "type": "tool_call"}
