"""Shared fixtures: a scripted chat model and an isolated workspace and database."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class ScriptedModel(GenericFakeChatModel):
    """Replies with the given messages in order; ignores tool binding."""

    disable_streaming: bool = True

    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def ai(text: str = "", *calls: dict, tokens: tuple[int, int] = (100, 10)) -> AIMessage:
    return AIMessage(
        text,
        tool_calls=list(calls),
        usage_metadata={
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "total_tokens": sum(tokens),
        },
    )


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Point the workspace and chat database at a temp dir for every test."""
    from optimist.config import get_settings

    workspace = tmp_path / "workspace"
    monkeypatch.setenv("OPTIMIST_WORKSPACE", str(workspace))
    monkeypatch.setenv("OPTIMIST_DB", str(tmp_path / "data" / "chats.sqlite"))
    monkeypatch.setenv("OPTIMIST_MODEL", "openai:gpt-5.4")
    monkeypatch.delenv("OPTIMIST_SUBAGENT_MODEL", raising=False)
    monkeypatch.delenv("OPTIMIST_AUTO_APPROVE", raising=False)
    get_settings.cache_clear()
    yield get_settings().workspace
    get_settings.cache_clear()


@pytest.fixture
def script(monkeypatch):
    """Make every agent the CLI builds use scripted models.

    Call the fixture with the main agent's replies (and optionally the
    subagents' replies).
    """
    import optimist.cli.session as session_mod

    real_build = session_mod.build_assistant

    def install(main: list[AIMessage], sub: list[AIMessage] | None = None):
        main_iter, sub_iter = iter(main), iter(sub or [])

        def build(*, model, subagents, **kwargs):
            for spec in subagents:
                spec["model"] = ScriptedModel(messages=sub_iter)
            return real_build(
                model=ScriptedModel(messages=main_iter), subagents=subagents, **kwargs
            )

        monkeypatch.setattr(session_mod, "build_assistant", build)

    return install
