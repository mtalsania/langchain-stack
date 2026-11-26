r"""The finished assistant: a LangGraph agent with RAG, tools, memory and a
human approval step. Modules 06-12 build up to exactly this file.

    START -> agent --(no tool calls)--------------------> END
               |  \--(only safe tools)--> tools -> agent
               \----(sensitive tool)----> human_review --approve--> tools
                                                      \--reject---> agent

`graph` at the bottom is what langgraph.json points Studio / `langgraph dev` at.
"""
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from common.llm import get_llm
from common.tools import ALL_TOOLS, SENSITIVE_TOOLS

SYSTEM_PROMPT = """You are the customer assistant for Nimbus Bikes.
- For any question about Nimbus Bikes, call search_docs first and answer only from what it returns.
- Use calculator for every calculation, even simple ones.
- If the documents do not contain the answer, say you don't know.
- Cite the source file in brackets, e.g. [products.md].
- Only call create_support_ticket when the customer explicitly asks for a ticket."""

SENSITIVE_NAMES = {t.name for t in SENSITIVE_TOOLS}


class State(MessagesState):
    """Conversation messages (with the add_messages reducer) - that's all we need."""


def agent(state: State) -> dict:
    llm = get_llm().bind_tools(ALL_TOOLS)
    response = llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
    return {"messages": [response]}


def route_after_agent(state: State) -> Literal["tools", "human_review", "__end__"]:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return END
    if any(call["name"] in SENSITIVE_NAMES for call in last.tool_calls):
        return "human_review"
    return "tools"


def human_review(state: State) -> Command[Literal["tools", "agent"]]:
    """Pause the graph and wait for a person. interrupt() saves state via the
    checkpointer and returns whatever the caller passes in Command(resume=...)."""
    last: AIMessage = state["messages"][-1]
    decision = interrupt(
        {
            "question": "Approve these tool calls?",
            "tool_calls": [{"name": c["name"], "args": c["args"]} for c in last.tool_calls],
        }
    )
    if decision.get("action") == "approve":
        return Command(goto="tools")
    if decision.get("action") == "edit":
        # Replace the arguments of the sensitive call(s) with the reviewer's version.
        new_calls = [
            {**c, "args": decision["args"]} if c["name"] in SENSITIVE_NAMES else c
            for c in last.tool_calls
        ]
        edited = AIMessage(content=last.content, tool_calls=new_calls, id=last.id)
        return Command(goto="tools", update={"messages": [edited]})
    # Rejected: every tool call still needs a ToolMessage answer, or the model API errors.
    reason = decision.get("feedback", "The human reviewer rejected this action.")
    rejections = [
        ToolMessage(content=f"Not executed. {reason}", tool_call_id=c["id"])
        for c in last.tool_calls
    ]
    return Command(goto="agent", update={"messages": rejections})


def build_graph(checkpointer=None):
    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_node("human_review", human_review)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


# Used by `langgraph dev` / Studio. The server supplies its own checkpointer.
graph = build_graph()
