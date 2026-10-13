# sciassist

An AI science assistant that can also write and run code: a deepagents
agent with literature tools, file and shell tools, and subagents. Design
notes: [`../requirement.md`](../requirement.md),
[`../capability_taxonomy.md`](../capability_taxonomy.md).

```bash
uv venv .venv && uv pip install -e '.[openai,anthropic,ui,dev]'
cp .env.example .env          # API keys are also read from ../literature/.env
.venv/bin/sciassist           # terminal session
.venv/bin/sciassist ui        # web UI at http://127.0.0.1:8000
langgraph dev                 # LangGraph Studio, via langgraph.json
.venv/bin/pytest              # tests (scripted model, no API calls)
```

## Terminal

Modelled on Claude Code and Codex.

```bash
sciassist                               # interactive session
sciassist --resume last                 # continue the latest conversation
sciassist -m anthropic:claude-opus-5-5  # pick the model
sciassist --mode auto                   # approval mode: ask | edits | auto
sciassist -p "fit a line to @data.csv"  # one turn, answer on stdout
cat log.txt | sciassist -p "why did this fail?" --json
```

Text streams as the model writes it, with a spinner showing what's running.
Tool calls print as `● tool(args)` with a short result under them. File edits
and writes show as diffs, and subagent activity is indented under the `task`
call that started it. Each turn ends with its token count and cost; prices
come from `genai-prices`.

| Key | |
| --- | --- |
| Enter | send |
| Esc Enter, Ctrl-J, `\` Enter | new line |
| Shift+Tab | cycle approval mode |
| Esc, Ctrl-C | interrupt the agent; a running shell command is killed |
| `@path` | attach a workspace file (Tab completes) |

**Approval modes.** `ask` confirms every file change and shell command.
`edits` (the default) lets edits and writes through and asks for shell
commands and deletions. `auto` asks for nothing. At a prompt, answer `y`,
`a` (yes, and stop asking about this tool for the session), `n`, or type what
the agent should do instead. Headless runs can't ask, so anything that would
need a person is rejected, with a reason the agent sees.

**Commands.** `/help`, `/new`, `/resume`, `/model`, `/mode`, `/compact`
(summarize older messages to free context), `/cost`, `/init` (write
`/AGENTS.md`), `/verbose`, `/exit`.

**Project instructions.** `AGENTS.md` at the workspace root is loaded into the
system prompt on every turn, in the terminal and the web UI. `/init` drafts
one.

The terminal and web UI share the chat database, so a conversation started in
one can be resumed in the other.

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

The agent is built with [deepagents](../deepagents/libs/deepagents)
(`create_deep_agent`). deepagents supplies the loop and the harness; this
package adds the literature tools, the prompts, the subagents and the UI.

```
main agent (sciassist/agent.py)
├── search_literature          quick multi-source search (seconds)
├── deep_literature_research   full litresearch graph (minutes) → /literature/<slug>/
├── ls / read_file / write_file / edit_file / delete / glob / grep    (deepagents)
├── execute                    shell command in the workspace          (deepagents)
└── task(description, subagent_type)   spawn a subagent; several calls in one turn run in parallel
    ├── general-purpose        search_literature + file tools + execute
    └── coder                  file tools + execute: write, run and debug code
```

From deepagents the agent also gets summarization of long threads, offloading
of large tool results to files, and repair of tool calls left dangling by a
cancelled run.

**Workspace.** The workspace is the agent's filesystem root: `/literature/x.md`
means `workspace/literature/x.md`, and file tools reject paths outside it.
Shell commands run with the workspace as working directory and a 300 s
timeout. They get only `PATH`, `HOME`, `LANG` and a few similar variables, so
API keys stay out of the shell. Running in the workspace isn't a sandbox,
though: a command can still reach the rest of the machine as your user.

**Approval.** In the web UI, `execute` and `delete` pause the run for
approval. The terminal pauses on every file change too, and its approval
mode decides which pauses need asking (see [Terminal](#terminal)). A
rejection can carry a reason, which the agent sees.
`SCIASSIST_AUTO_APPROVE=1` turns approval off in the web UI and starts the
terminal in `auto` mode.

Deep research is a **tool**, not a subagent: litresearch already has its own
planner and critic, so another LLM loop around it would add cost but no
judgement. It returns a summary and file paths, and the agent reads what it
needs.

| Module | Role |
| --- | --- |
| `agent.py` | `create_deep_agent` assembly, workspace backend, approval rules; `make_graph` for `langgraph.json` |
| `subagents.py` | Subagent specs (`general-purpose`, `coder`) |
| `tools/literature.py` | Wrappers over `litresearch` |
| `tools/files.py` | Workspace path mapping for the UI and tool output |
| `prompts.py` | System prompts |
| `server.py`, `static/` | Web UI |
| `threads.py` | Conversation index for the chat list |
| `shell.py` | Workspace shell that kills commands when a run is cancelled |
| `cli/` | Terminal: `repl.py` (input, commands), `session.py` (running turns), `render.py`, `approvals.py`, `usage.py` |
| `config.py` | `SCIASSIST_MODEL`, `SCIASSIST_SUBAGENT_MODEL`, `SCIASSIST_WORKSPACE`, `SCIASSIST_DB`, `SCIASSIST_AUTO_APPROVE` |

Deep-research budgets and models stay in `../literature/litresearch.yaml`.
