"""The tool-calling loop, shared by the main agent and its subagents.

    START → agent ──(tool calls?)──► tools ──┐
              ▲                              │
              └──────────────────────────────┘
              └──(no tool calls)──► END

Built by hand rather than with a prebuilt agent so the state can grow: the
concept graph from ``requirement.md`` will become fields on ``AgentState``,
updated by tools, rather than something reconstructed from chat history.

Several tool calls in one model turn run concurrently in the ``tools`` node,
which is how the agent fans out parallel subagents.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolNode, tools_condition


class AgentState(MessagesState):
    """Conversation state. Extend here (e.g. a concept graph) as the design grows."""


def patch_dangling_tool_calls(messages: Sequence[BaseMessage]) -> tuple[list[BaseMessage], bool]:
    """Answer tool calls that never got a result.

    A run cancelled between the model's tool call and the tool's result (the
    UI's Stop button, a page reload, Ctrl-C, a server restart) leaves an
    unanswered call in the checkpoint, and providers reject every later
    request on that thread. Each such call gets a synthetic result saying it
    was cancelled, so the model knows and can retry if it still matters.
    """
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    patched: list[BaseMessage] = []
    repaired = False
    for message in messages:
        patched.append(message)
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            if call["id"] not in answered:
                patched.append(
                    ToolMessage(
                        content="Cancelled: this tool call was interrupted before it "
                                "returned a result.",
                        tool_call_id=call["id"],
                        name=call["name"],
                        status="error",
                    )
                )
                repaired = True
    return patched, repaired


def load_model(spec: str | BaseChatModel) -> BaseChatModel:
    return spec if isinstance(spec, BaseChatModel) else init_chat_model(spec)


def build_agent_graph(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str,
    *,
    checkpointer=None,
    name: str | None = None,
):
    """Compile a tool-calling agent: one model node, one tool node, one loop."""
    llm = load_model(model).bind_tools(list(tools))
    system = SystemMessage(system_prompt)

    async def agent(state: AgentState) -> dict:
        messages, repaired = patch_dangling_tool_calls(state["messages"])
        response = await llm.ainvoke([system, *messages])
        if not repaired:
            return {"messages": [response]}
        # The synthetic results must sit right after their tool call, which an
        # append can't do, so rewrite the history in place.
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages, response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    # Tool failures go back to the model as error messages so it can recover,
    # instead of aborting the conversation.
    builder.add_node("tools", ToolNode(list(tools), handle_tool_errors=True))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer, name=name)
