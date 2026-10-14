"""LangChain and LangGraph behaviour evenkeel relies on. A failure here after an upgrade means the
middleware must change before it supports that version."""

import operator
from typing import Annotated

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from typing_extensions import NotRequired

from evenkeel._compat import (
    AgentMiddleware,
    AgentState,
    Command,
    ExtendedModelResponse,
    GraphBubbleUp,
    hook_config,
    interrupt,
)
from tests.fakes import ScriptedChatModel, ai, tool_call


@tool
def boom(x: str) -> str:
    """Always fails."""
    raise RuntimeError("boom")


def start(text: str = "hi") -> dict:
    return {"messages": [HumanMessage(text)]}


def test_interrupt_is_an_exception_the_middleware_must_let_through():
    assert issubclass(GraphInterrupt, GraphBubbleUp)
    assert issubclass(GraphBubbleUp, Exception)


def test_override_messages_reaches_the_model():
    class AddNote(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            return handler(request.override(messages=[*request.messages, HumanMessage("note")]))

    model = ScriptedChatModel(script=[ai("ok")])
    create_agent(model, middleware=[AddNote()]).invoke(start())
    assert model.calls[0][-1].content == "note"


def test_interrupt_in_wrap_model_call_pauses_and_resumes_with_the_value():
    class AskFirst(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            answer = interrupt("go?")
            return handler(request.override(messages=[*request.messages, HumanMessage(str(answer))]))

    model = ScriptedChatModel(script=[ai("ok")])
    agent = create_agent(model, middleware=[AskFirst()], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "c3"}}
    first = agent.invoke(start(), config)
    assert first["__interrupt__"][0].value == "go?"
    assert model.calls == []
    agent.invoke(Command(resume="yes"), config)
    assert model.calls[0][-1].content == "yes"


def test_before_model_can_end_the_run():
    class Stop(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        def before_model(self, state, runtime):
            return {"messages": [AIMessage("stopped")], "jump_to": "end"}

    model = ScriptedChatModel(script=[ai("never")])
    result = create_agent(model, middleware=[Stop()]).invoke(start())
    assert model.calls == []
    assert result["messages"][-1].content == "stopped"


def test_tool_exceptions_reach_wrap_tool_call():
    seen: list[str] = []

    class Watch(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            try:
                return handler(request)
            except Exception as error:
                seen.append(type(error).__name__)
                call = request.tool_call
                return ToolMessage("handled", tool_call_id=call["id"], name=call["name"], status="error")

    model = ScriptedChatModel(script=[ai(tool_calls=[tool_call("boom", {"x": "1"})]), ai("done")])
    create_agent(model, tools=[boom], middleware=[Watch()]).invoke(start())
    assert seen == ["RuntimeError"]


def test_unknown_tool_becomes_an_error_tool_message():
    model = ScriptedChatModel(script=[ai(tool_calls=[tool_call("serch_web", {"q": "x"})]), ai("done")])
    result = create_agent(model, tools=[boom]).invoke(start())
    errors = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert errors and errors[0].status == "error" and "serch_web" in errors[0].content


class NotesState(AgentState):
    notes: NotRequired[Annotated[list[str], operator.add]]


def test_wrap_hooks_can_append_to_custom_state():
    class Note(AgentMiddleware):
        state_schema = NotesState

        def wrap_model_call(self, request, handler):
            note = Command(update={"notes": ["m"]})
            return ExtendedModelResponse(model_response=handler(request), command=note)

        def wrap_tool_call(self, request, handler):
            message = handler(request)
            return Command(update={"messages": [message], "notes": ["t"]})

    @tool
    def ok(x: str) -> str:
        """Succeeds."""
        return "fine"

    model = ScriptedChatModel(script=[ai(tool_calls=[tool_call("ok", {"x": "1"})]), ai("done")])
    agent = create_agent(model, tools=[ok], middleware=[Note()], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "c7"}}
    agent.invoke(start(), config)
    assert sorted(agent.get_state(config).values["notes"]) == ["m", "m", "t"]
