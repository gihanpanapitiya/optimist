# sciassist

An AI science assistant: a tool-calling LangGraph agent with tools and
subagents. Design notes: [`../requirement.md`](../requirement.md),
[`../capability_taxonomy.md`](../capability_taxonomy.md).

```bash
uv venv .venv && uv pip install -e '.[openai,anthropic,ui,dev]'
cp .env.example .env          # API keys are also read from ../literature/.env
.venv/bin/sciassist chat      # terminal chat
.venv/bin/sciassist ui        # web UI at http://127.0.0.1:8000
langgraph dev                 # LangGraph Studio, via langgraph.json
```

## Web UI

Chat on the left; the workspace on the right, with the file tree on top and a
viewer below (rendered markdown, highlighted code/JSON, images, PDFs). Tool
calls show as expandable cards with live progress. Workspace paths in chat
and `[doc_0014]` citations in reports open in the viewer, and the open file
reloads as deep research rewrites it. Panes are resizable.

`sciassist/server.py` is the FastAPI backend (SSE chat stream, history, file
API confined to the workspace); `sciassist/static/` is a no-build frontend.
`/?thread=<id>#file=<path>` deep-links a conversation and a file.

Chats are checkpointed to SQLite (`data/chats.sqlite`, or `SCIASSIST_DB`), so
they survive restarts. The sidebar lists them by last activity, with search,
rename and delete; a `threads` table in the same file holds titles and times
(`sciassist/threads.py`). Switching chats doesn't stop a run: it continues in
the background, marked with a pulsing dot. The web UI and the terminal share
the database: `sciassist chat --thread <id>` resumes any conversation.

## Architecture

```
main agent (sciassist/agent.py)
├── search_literature          quick multi-source search (seconds)
├── deep_literature_research   full litresearch graph (minutes) → workspace/literature/<slug>/
├── list_files / read_file     read reports and evidence from the workspace
└── task(description, subagent_type)   spawn a subagent; several calls in one turn run in parallel
    └── general-purpose        search_literature, read_file, list_files
```

All agents share one loop (`sciassist/graph.py`): `agent → tools → agent`
until the model stops calling tools. `AgentState` is where the concept
graph is meant to go.

Deep research is a **tool**, not a subagent: litresearch already has its own
planner and critic, so another LLM loop around it would add cost but no
judgement. It returns a summary and file paths, and the agent reads what it
needs.

| Module | Role |
| --- | --- |
| `graph.py` | Generic tool-calling loop, `AgentState` |
| `agent.py` | Main agent assembly; `make_graph` for `langgraph.json` |
| `subagents.py` | `SubAgent` specs (deepagents-shaped) and the `task` tool |
| `tools/literature.py` | Wrappers over `litresearch` |
| `tools/files.py` | Workspace-confined file reading |
| `prompts.py` | System prompts |
| `server.py`, `static/` | Web UI |
| `threads.py` | Conversation index for the chat list |
| `config.py` | `SCIASSIST_MODEL`, `SCIASSIST_SUBAGENT_MODEL`, `SCIASSIST_WORKSPACE`, `SCIASSIST_DB` |

Deep-research budgets and models stay in `../literature/litresearch.yaml`.
