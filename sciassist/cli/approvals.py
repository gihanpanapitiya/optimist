"""Approval modes, as in Claude Code and Codex.

The terminal's agent pauses before every tool in ``MUTATING_TOOLS``; the mode
decides which pauses are answered automatically and which ask the user:

    ask     ask before every file change and shell command
    edits   file edits and writes go through; shell commands and deletions ask
    auto    everything goes through

Shift+Tab cycles the mode. Answering "always" at a prompt stops asking about
that tool for the rest of the session.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

MODES = ("ask", "edits", "auto")
MODE_HELP = {
    "ask": "ask before every change",
    "edits": "auto-approve file edits, ask for shell and deletes",
    "auto": "approve everything",
}
_AUTO_APPROVED = {
    "ask": frozenset(),
    "edits": frozenset({"write_file", "edit_file"}),
    "auto": None,  # everything
}

Decision = dict
AskFn = Callable[[dict], Awaitable[Decision | None]]
"""Shows one action and returns the user's decision, or None for "always"."""


@dataclass
class Approvals:
    mode: str = "edits"
    always: set[str] = field(default_factory=set)

    def cycle(self) -> str:
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        return self.mode

    def auto_approves(self, tool: str) -> bool:
        allowed = _AUTO_APPROVED[self.mode]
        return allowed is None or tool in allowed or tool in self.always

    async def decide(self, interrupts, ask: AskFn | None) -> dict:
        """Resume payload for pending interrupts: interrupt id -> decisions.

        ``ask`` is None when nobody can answer (headless); actions that would
        need asking are then rejected with a reason the agent can act on.
        """
        resume = {}
        for interrupt in interrupts:
            decisions = []
            for action in (interrupt.value or {}).get("action_requests", []):
                decisions.append(await self._decide_one(action, ask))
            resume[interrupt.id] = {"decisions": decisions}
        return resume

    async def _decide_one(self, action: dict, ask: AskFn | None) -> Decision:
        name = action["name"]
        if self.auto_approves(name):
            return {"type": "approve"}
        if ask is None:
            return {
                "type": "reject",
                "message": f"Not run: `{name}` needs approval, and this is a non-interactive "
                f"session in '{self.mode}' mode. Say what you would run instead.",
            }
        decision = await ask(action)
        if decision is None:  # "yes, and don't ask again"
            self.always.add(name)
            return {"type": "approve"}
        return decision
