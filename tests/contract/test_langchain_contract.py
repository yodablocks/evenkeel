"""LangChain and LangGraph behaviour evenkeel relies on. A failure here after an upgrade means the
middleware must change before it supports that version."""

import operator
from typing import Annotated, NotRequired

import pytest
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from pydantic import BaseModel

from evenkeel._compat import (
    AgentMiddleware,
    AgentState,
    Command,
    ExtendedModelResponse,
    GraphBubbleUp,
    ModelResponse,
    hook_config,
    interrupt,
)
from evenkeel.budget import ModelPrice, PriceTable
from evenkeel.middleware import EvenKeelMiddleware
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


def test_a_model_request_carries_what_the_budget_check_reads():
    seen: dict = {}

    class Look(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            seen.update(
                state=request.state,
                settings=request.model_settings,
                system=request.system_message,
                tools=request.tools,
                model=request.model,
            )
            return handler(request)

    @tool
    def ok(x: str) -> str:
        """Succeeds."""
        return "fine"

    model = ScriptedChatModel(script=[ai("done")])
    create_agent(model, tools=[ok], system_prompt="be brief", middleware=[Look()]).invoke(start())
    assert "messages" in seen["state"]
    assert isinstance(seen["settings"], dict)
    assert seen["system"].content == "be brief"
    assert [t.name for t in seen["tools"]] == ["ok"]
    assert seen["model"] is model


def test_wrap_model_call_can_answer_without_calling_the_model_and_end_the_run():
    class Answer(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            return ModelResponse(result=[AIMessage("stopped")])

    model = ScriptedChatModel(script=[ai("never")])
    result = create_agent(model, middleware=[Answer()]).invoke(start())
    assert model.calls == []
    assert result["messages"][-1].content == "stopped"


def test_a_structured_output_error_reaches_wrap_model_call_with_the_rejected_reply():
    # evenkeel counts a rejected reply's cost from the error's ai_message, and retries it as bad_output.
    class Answer(BaseModel):
        amount: int

    prices = PriceTable({"priced-model": ModelPrice(input_per_mtok=1.0, output_per_mtok=1.0)})
    rejected = ai(
        tool_calls=[tool_call("Answer", {"amount": "lots"}, "c1")], model="priced-model", tokens=(100, 50)
    )
    accepted = ai(
        tool_calls=[tool_call("Answer", {"amount": 5}, "c2")], model="priced-model", tokens=(100, 50)
    )
    model = ScriptedChatModel(script=[rejected, accepted])
    agent = create_agent(
        model,
        response_format=ToolStrategy(Answer, handle_errors=False),
        middleware=[EvenKeelMiddleware(prices=prices)],
        checkpointer=InMemorySaver(),
    )
    result = agent.invoke(start(), {"configurable": {"thread_id": "structured"}})

    assert len(model.calls) == 2
    assert result["structured_response"] == Answer(amount=5)
    assert result["evenkeel_spent_usd"] == pytest.approx(2 * 0.00015), "the rejected reply is counted"
    assert [(r["error_type"], r["kind"]) for r in result["evenkeel_attempts"]] == [
        ("StructuredOutputValidationError", "bad_output")
    ]
