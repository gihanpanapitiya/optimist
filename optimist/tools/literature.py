"""Literature tools, backed by the ``optimistresearch`` package.

Two speeds on purpose. ``search_literature`` is one round of retrieval and
returns in seconds, which is what iterative brainstorming needs: check an idea,
think, check again. ``deep_literature_research`` runs the whole optimistresearch
graph (plan, search, snowball, read, synthesize, critique) and takes minutes.

Deep research is a *tool* rather than a subagent because it is already a
self-contained agent with its own planner and critic. Wrapping it in another
LLM loop would add cost and a second planner without adding judgement. The
tool returns a summary plus file paths, so the main agent's context only holds
what it chooses to read.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from pydantic import Field

from ..config import get_settings
from .files import to_agent_path

logger = logging.getLogger(__name__)

# Deep research runs are expensive, and optimistresearch closes its shared HTTP
# client when a run ends -- so two concurrent runs would also step on each
# other. One at a time.
_deep_research_lock = asyncio.Lock()

REPORT_PREVIEW_CHARS = 8000


def _emit(runtime: ToolRuntime | None, tool_name: str, message: str) -> None:
    """Send a progress event to stream_mode="custom", tagged with the tool call.

    The tool_call_id lets a UI attach progress to the right call when several
    run in parallel.
    """
    if runtime is None:
        return
    try:
        runtime.stream_writer(
            {"tool": tool_name, "message": message, "tool_call_id": runtime.tool_call_id}
        )
    except Exception:  # noqa: BLE001 - progress must never break a tool
        pass


@tool
async def search_literature(
    query: Annotated[
        str, Field(description="Keyword-style query, e.g. 'SEM image segmentation U-Net'")
    ],
    max_results: Annotated[int, Field(description="Results per source", ge=1, le=25)] = 8,
    year_from: Annotated[int | None, Field(description="Earliest publication year")] = None,
    year_to: Annotated[int | None, Field(description="Latest publication year")] = None,
    sources: Annotated[
        list[str] | None,
        Field(description="Subset of arxiv, openalex, crossref, pubmed, semantic_scholar, web. "
                          "Default: all configured sources."),
    ] = None,
    runtime: ToolRuntime = None,
) -> str:
    """Quick literature search across scholarly sources (seconds, not minutes).

    Returns deduplicated papers with authors, year, venue, identifiers, citation
    counts and abstracts. Use for quick checks and targeted lookups; use
    deep_literature_research for a full, cited survey.
    """
    from optimistresearch.sources.registry import active_sources
    from optimistresearch.state import QueryPlan
    from optimistresearch.store.canonicalize import Canonicalizer

    available = active_sources()
    chosen = {
        name: src for name, src in available.items() if sources is None or name in sources
    }
    if not chosen:
        return f"None of {sources} are available. Configured sources: {sorted(available)}"

    _emit(runtime, "search_literature", f"searching {', '.join(chosen)}: {query}")

    async def one(name: str, source) -> list:
        plan = QueryPlan(
            intent_id="adhoc",
            subquestion_id="",
            source=name,
            query=query,
            year_from=year_from,
            year_to=year_to,
            max_results=max_results,
        )
        try:
            return await source.search(plan)
        except Exception as exc:  # noqa: BLE001 - one dead source shouldn't sink the search
            logger.warning("search on %s failed: %s", name, exc)
            return []

    results = await asyncio.gather(*(one(n, s) for n, s in chosen.items()))
    hits = [hit for batch in results for hit in batch]
    if not hits:
        return f"No results for {query!r} from {', '.join(chosen)}."

    # Hits arrive source by source; interleave them by rank so the first
    # documents minted are each source's best, not all of one source's.
    hits.sort(key=lambda h: h.provenance.rank)
    docs = list(Canonicalizer({}).ingest(hits).documents.values())
    return _format_documents(docs, query)


def _format_documents(docs: list, query: str) -> str:
    lines = [f"{len(docs)} papers for {query!r}:\n"]
    for i, doc in enumerate(docs, 1):
        authors = ", ".join(doc.authors[:3]) + (" et al." if len(doc.authors) > 3 else "")
        meta = " | ".join(
            part
            for part in (
                str(doc.year) if doc.year else "",
                doc.venue or "",
                f"doi:{doc.doi}" if doc.doi else "",
                f"arXiv:{doc.arxiv_id}" if doc.arxiv_id else "",
                f"cited by {doc.citation_count}" if doc.citation_count is not None else "",
                f"via {', '.join(doc.sources)}",
            )
            if part
        )
        lines.append(f"[{i}] {doc.title}\n    {authors}\n    {meta}")
        if doc.url and not doc.doi:
            lines.append(f"    {doc.url}")
        if doc.abstract:
            abstract = " ".join(doc.abstract.split())
            lines.append(f"    {abstract[:600]}{'…' if len(abstract) > 600 else ''}")
        lines.append("")
    return "\n".join(lines)


@tool
async def deep_literature_research(
    question: Annotated[str, Field(description="The research question to investigate")],
    scope_notes: Annotated[
        str,
        Field(description="Constraints for the planner: domain, years, methods to include "
                          "or exclude, what the answer will be used for"),
    ] = "",
    runtime: ToolRuntime = None,
) -> str:
    """Run an autonomous, multi-iteration literature review (takes several minutes).

    Plans subquestions, searches multiple sources, follows citations, reads the
    best papers to extract evidence, synthesizes, critiques its own coverage,
    and writes a cited report. Returns run statistics, the start of the report,
    and paths to the full report and evidence files (read them with read_file).
    """
    from optimistresearch.runner import run_research, summarize_run

    out_dir = str(get_settings().literature_dir)
    seen_log = 0

    def on_progress(_kind: str, snapshot: dict) -> None:
        nonlocal seen_log
        log = snapshot.get("log") or []
        if len(log) < seen_log:  # the log reducer can clear between iterations
            seen_log = 0
        for line in log[seen_log:]:
            _emit(runtime, "deep_literature_research", line)
        seen_log = len(log)

    if _deep_research_lock.locked():
        _emit(runtime, "deep_literature_research", "waiting for another run to finish")

    async with _deep_research_lock:
        _emit(runtime, "deep_literature_research", f"starting: {question}")
        state = await run_research(
            question, scope_notes=scope_notes, output_dir=out_dir, on_progress=on_progress
        )

    summary = summarize_run(state, out_dir)
    report = state.get("final_report") or ""
    preview = report[:REPORT_PREVIEW_CHARS]
    truncated = len(report) > REPORT_PREVIEW_CHARS

    return "\n".join(
        [
            "Deep research complete.",
            f"- documents retrieved: {summary['documents_retrieved']}, "
            f"read: {summary['documents_read']}, evidence items: {summary['evidence_items']}, "
            f"contradictions: {summary['contradictions']}",
            f"- iterations: {summary['iterations']}, mean subquestion coverage: "
            f"{summary['mean_coverage']:.2f}, stopped because: {summary['terminated_because']}",
            f"- corpus: {to_agent_path(summary['corpus_root'])}",
            f"- full report: {to_agent_path(summary['report_path'])}",
            "- also in the corpus: research_plan.md, evidence/claims.md, "
            "evidence/contradictions.md, subtopics/, titles.md",
            "",
            "## Report" + (" (truncated; read the full report for the rest)" if truncated else ""),
            preview or "(no report was written; check the corpus files)",
        ]
    )
