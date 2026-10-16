"""One terminal session: the agent, the current thread, and running a turn.

Shared by the interactive REPL and headless ``optimist -p``. A turn streams
until the agent finishes or pauses for approval; approvals are answered (by
the mode, or by asking) and the run resumes, until the agent is done.
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.types import Command

from ..agent import MUTATING_TOOLS, build_assistant
from ..config import get_settings
from ..subagents import default_subagents
from ..threads import ThreadStore
from .approvals import Approvals, AskFn
from .render import Renderer
from .usage import Usage, UsageTracker

RECURSION_LIMIT = 500


@dataclass
class TurnResult:
    text: str = ""
    """The main agent's last message."""
    usage: Usage = field(default_factory=Usage)
    interrupted: bool = False
    seconds: float = 0.0


@dataclass
class Session:
    saver: object
    threads: ThreadStore
    renderer: Renderer
    approvals: Approvals
    model: str = ""
    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    usage: UsageTracker = field(default_factory=UsageTracker)
    graph: object = None

    def __post_init__(self) -> None:
        self.model = self.model or get_settings().model
        self.build()

    @property
    def subagent_model(self) -> str:
        settings = get_settings()
        # Subagents follow /model unless they have a model of their own.
        return self.model if settings.subagent_model == settings.model else settings.subagent_model

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}, "recursion_limit": RECURSION_LIMIT}

    def build(self) -> None:
        subagents = default_subagents()
        for spec in subagents:
            spec["model"] = self.subagent_model
        # Every mutating tool pauses; Approvals decides which pauses need a person.
        self.graph = build_assistant(
            model=self.model,
            subagents=subagents,
            checkpointer=self.saver,
            approval_tools=MUTATING_TOOLS,
        )

    def new_thread(self, thread_id: str | None = None) -> None:
        self.thread_id = thread_id or str(uuid.uuid4())
        self.usage.context_tokens = 0

    async def message_count(self) -> int:
        state = await self.graph.aget_state(self.config)
        return len((state.values or {}).get("messages", []))

    async def run(
        self, message: str, *, ask: AskFn | None, watch_keys: bool, title: str | None = None
    ) -> TurnResult:
        """Send a user message and run until the agent is done or interrupted."""
        await self.threads.touch(self.thread_id, title or message)
        result = TurnResult()
        started = time.monotonic()
        graph_input = {"messages": [{"role": "user", "content": message}]}
        try:
            while graph_input is not None:
                self.renderer.start()
                task = asyncio.create_task(self._stream(graph_input, result))
                try:
                    interrupted = await _wait(task, watch_keys)
                finally:
                    self.renderer.stop()
                if interrupted:
                    result.interrupted = True
                    self.renderer.notice("  ⎿  Interrupted. What should the agent do instead?")
                    break
                state = await self.graph.aget_state(self.config)
                graph_input = None
                if state.interrupts:
                    resume = await self.approvals.decide(state.interrupts, ask)
                    graph_input = Command(resume=resume)
        finally:
            result.seconds = time.monotonic() - started
            await self.threads.touch(self.thread_id, title or message)
        return result

    async def _stream(self, graph_input, result: TurnResult) -> None:
        renderer = self.renderer
        async for ns, mode, chunk in self.graph.astream(
            graph_input,
            config=self.config,
            stream_mode=["messages", "updates", "custom"],
            subgraphs=True,
        ):
            if mode == "messages":
                message, meta = chunk
                if (
                    not ns
                    and isinstance(message, AIMessageChunk)
                    and meta.get("langgraph_node") == "model"
                    and message.text
                ):
                    renderer.token(message.text)
            elif mode == "custom":
                if isinstance(chunk, dict):
                    renderer.progress(chunk)
            elif mode == "updates":
                for node, update in chunk.items():
                    self._update(node, update, ns, result)

    def _update(self, node: str, update, ns: tuple, result: TurnResult) -> None:
        if not isinstance(update, dict):
            return
        messages = update.get("messages")
        if not isinstance(messages, list):
            return
        for message in messages:
            if node == "model" and isinstance(message, AIMessage):
                model = self.subagent_model if ns else self.model
                result.usage.add(self.usage.record(message, model, main=not ns))
                if not ns:
                    result.text = message.text
                self.renderer.ai_message(message, ns)
            elif isinstance(message, ToolMessage) and (
                node == "tools" or node.startswith("HumanInTheLoopMiddleware")
            ):
                # The approval node only adds results for rejected calls.
                self.renderer.tool_result(message, ns)


async def _wait(task: asyncio.Task, watch_keys: bool) -> bool:
    """Wait for the run; Esc or Ctrl-C cancels it. Returns True if cancelled."""
    if not (watch_keys and sys.stdin.isatty()):
        await task
        return False

    from prompt_toolkit.input import create_input
    from prompt_toolkit.keys import Keys

    stop_keys = {Keys.Escape, Keys.ControlC}
    loop = asyncio.get_running_loop()
    inp = create_input()

    def check(keys) -> None:
        if any(k.key in stop_keys for k in keys) and not task.done():
            task.cancel()

    def on_input() -> None:
        check(inp.read_keys())
        # A lone Esc is held back in case it starts an escape sequence.
        loop.call_later(0.05, lambda: check(inp.flush_keys()))

    # Raw mode: keys arrive one by one and Ctrl-C is a key, not SIGINT.
    with inp.raw_mode(), inp.attach(on_input):
        await asyncio.wait({task})
    if task.cancelled():
        return True
    task.result()  # re-raise a failure
    return False
