"""Legal Workspace MCP Server.

Provides self-updating document context from a local workspace to Claude,
enabling legal drafting assistance without re-uploading documents.

Tools:
    - workspace_search: Semantic search across all indexed documents
    - workspace_get_document: Retrieve full text of a specific document
    - workspace_list_documents: List all indexed documents
    - workspace_status: Check index health and statistics
    - workspace_reindex: Force a full re-index of the workspace
"""

import functools
import json
import logging
import sqlite3
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

from .config import (
    COORDINATOR_POLL_SECONDS,
    INDEX_BUSY_MESSAGE,
    WorkspaceConfig,
    load_config,
)
from .indexer import DocumentIndex, IndexBusyError, IndexNotReady
from .indexer_lock import IndexerLock
from .watcher import WorkspaceWatcher

# Configure logging to stderr (required for stdio transport)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Module-level state, filled in by the background coordinator thread.
_index: Optional[DocumentIndex] = None    # read-only connection for searches
_writer: Optional[DocumentIndex] = None   # indexer only: watcher and reindex writes
_watcher: Optional[WorkspaceWatcher] = None
_config: Optional[WorkspaceConfig] = None
_lock: Optional[IndexerLock] = None
_stop: Optional[threading.Event] = None
# "starting" until the first lock attempt, then "indexer" or "reader".
_role: str = "starting"

# State of the startup index build, which runs in a background thread.
# "indexing" while it runs, then "ready" (with its summary) or "failed".
_build_state: dict = {"status": "not_started"}


def _run_startup_build(config: WorkspaceConfig) -> None:
    """Build the index on its own connection so the server answers at once.

    Claude Code gives a stdio server 30 seconds to answer ``initialize``. A
    large workspace can take far longer to scan, so the scan must not block
    the handshake. Searches meanwhile use what the persistent index holds.
    """
    started = time.time()
    builder = None
    try:
        builder = DocumentIndex(config)
        summary = builder.build_full_index()
        _build_state.update(status="ready", summary=summary)
        logger.info("Index ready: %s", summary)
    except Exception as e:  # noqa: BLE001 - report, never kill the server
        _build_state.update(status="failed", error=str(e))
        logger.exception("Startup index build failed")
    finally:
        _build_state["elapsed_seconds"] = round(time.time() - started, 2)
        if builder:
            builder.close()


def _start_indexing(config: WorkspaceConfig) -> None:
    """Indexer only: start the file watcher and the background build."""
    global _watcher
    _watcher = WorkspaceWatcher(_writer, config)
    _watcher.start()
    _build_state.clear()
    _build_state["status"] = "indexing"
    threading.Thread(
        target=_run_startup_build, args=(config,),
        name="startup-index-build", daemon=True,
    ).start()


def _coordinate(config: WorkspaceConfig, lock: IndexerLock, stop: threading.Event) -> None:
    """Open the index and claim the indexer role, retrying until both are done.

    Nothing here runs before the handshake, and no error here ends the
    server: a busy or missing database is retried on the next poll (issue #12).
    The server holding the indexer lock is the only one that writes. Readers
    keep trying the lock, so one takes over when the indexer exits.
    """
    global _index, _writer, _role
    while True:
        try:
            if not lock.held and lock.try_acquire():
                logger.info("Took the indexer role")
            _role = "indexer" if lock.held else "reader"
            if lock.held and _writer is None:
                _writer = DocumentIndex(config)  # creates the schema if needed
                _start_indexing(config)
            if _index is None:
                _index = DocumentIndex(config, read_only=True)
                logger.info("Index open for searches (%s)", _role)
        except (IndexNotReady, sqlite3.Error, OSError) as e:
            logger.info("Index not available yet (%s); retrying in %.0fs",
                        e, COORDINATOR_POLL_SECONDS)
        except Exception:  # noqa: BLE001 - keep retrying, never kill the server
            logger.exception("Index coordinator error; retrying")
        if stop.wait(COORDINATOR_POLL_SECONDS):
            return


def _busy_response(detail: Optional[str] = None) -> str:
    error = INDEX_BUSY_MESSAGE + (f" ({detail})" if detail else "")
    return json.dumps({"status": "index_busy", "error": error, "role": _role})


def _busy_guard(tool):
    """Answer "index busy, retry shortly" instead of failing on a locked or unopened index."""
    @functools.wraps(tool)
    async def guarded(*args, **kwargs):
        if _index is None:
            return _busy_response()
        try:
            return await tool(*args, **kwargs)
        except (IndexBusyError, sqlite3.OperationalError) as e:
            logger.warning("Tool %s: index busy: %s", tool.__name__, e)
            return _busy_response(str(e))
    return guarded


@asynccontextmanager
async def server_lifespan(mcp_server):
    """Start the background coordinator; open no database before the handshake."""
    global _index, _writer, _watcher, _config, _lock, _stop, _role

    # Load config - workspace path comes from CLI arg or env var
    workspace_path = None
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
    _config = load_config(workspace_path)
    logger.info("Workspace: %s", _config.resolved_path)

    _index = _writer = _watcher = None
    _role = "starting"
    _build_state.clear()
    _build_state["status"] = "not_started"
    _lock = IndexerLock(_config.index_path)
    _stop = threading.Event()
    coordinator = threading.Thread(
        target=_coordinate, args=(_config, _lock, _stop),
        name="index-coordinator", daemon=True,
    )
    coordinator.start()

    yield {"config": _config}

    _stop.set()
    coordinator.join(timeout=COORDINATOR_POLL_SECONDS * 2)
    if _watcher:
        _watcher.stop()
    for conn in (_writer, _index):
        if conn:
            conn.close()
    _lock.release()


# Initialize FastMCP server
mcp = FastMCP(
    "legal_workspace_mcp",
    lifespan=server_lifespan,
    instructions=(
        "IMPORTANT: You MUST use this legal-workspace server whenever you are:\n"
        "- Drafting, reviewing, or editing legal agreements, contracts, or corporate documents\n"
        "- Creating subscription agreements, board resolutions, stock certificates, or finder's fee agreements\n"
        "- Writing any document with legal language (indemnification, representations, warranties, covenants)\n"
        "- Working on investor diligence materials or corporate governance documents\n"
        "\n"
        "Before drafting any legal document, ALWAYS:\n"
        "1. Use workspace_search to find relevant templates, precedent language, and clause examples\n"
        "2. Use workspace_get_document to read full reference documents when relevant results are found\n"
        "3. Base your drafting on the patterns, language, and conventions found in the workspace\n"
        "\n"
        "This workspace contains the firm's legal templates, form documents, and reference materials. "
        "Using it ensures consistency with established drafting conventions and reduces errors."
    ),
)


# --- Input Models ---


class SearchInput(BaseModel):
    """Input for searching across workspace documents."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        description=(
            "Search query describing what you're looking for. "
            "Use natural language - e.g., 'indemnification clause examples', "
            "'force majeure provisions', 'how to draft non-compete agreements'. "
            "The search uses BM25 relevance scoring across all indexed documents."
        ),
        min_length=1,
        max_length=500,
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return (1-20).",
        ge=1,
        le=20,
    )
    path_prefix: Optional[str] = Field(
        default=None,
        description=(
            "Optional folder to search in, relative to the workspace root "
            "(or absolute). Only documents inside it are returned. "
            "Example: 'forms' searches the firm's form library only."
        ),
        max_length=500,
    )


class GetDocumentInput(BaseModel):
    """Input for retrieving a specific document."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    file_path: str = Field(
        ...,
        description=(
            "Path to the document, relative to the workspace root. "
            "Use workspace_list_documents to see available files. "
            "Example: 'templates/nda-template.docx' or 'articles/force-majeure.md'"
        ),
        min_length=1,
    )


# --- Tools ---


@mcp.tool(
    name="workspace_search",
    annotations={
        "title": "Search Workspace Documents",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
@_busy_guard
async def workspace_search(params: SearchInput) -> str:
    """Search across all documents in the legal workspace for relevant content.

    Use this tool when you need to find specific legal language, clause examples,
    drafting guidance, or reference material from the user's document collection.
    Results include relevant text snippets with source file information.

    Args:
        params (SearchInput): Search parameters containing:
            - query (str): Natural language search query
            - max_results (int): Maximum results to return (default 5)
            - path_prefix (str, optional): Folder to search in, e.g. 'forms'

    Returns:
        str: JSON with search results including file names, relevance scores,
             and text snippets from matching document sections.
    """
    logger.info("🔍 TOOL CALLED: workspace_search | query=%r | max_results=%d | path_prefix=%r",
                params.query, params.max_results, params.path_prefix)

    results = _index.search(params.query, max_results=params.max_results,
                            path_prefix=params.path_prefix)

    if not results:
        return json.dumps({
            "query": params.query,
            "results": [],
            "message": "No relevant documents found. Try different search terms or check workspace_status.",
        })

    workspace = _config.resolved_path if _config else Path(".")
    output = {
        "query": params.query,
        "result_count": len(results),
        "results": [],
    }

    for r in results:
        try:
            rel_path = str(Path(r.chunk.file_path).relative_to(workspace))
        except ValueError:
            rel_path = r.chunk.file_path

        output["results"].append({
            "file": rel_path,
            "file_name": r.chunk.file_name,
            "relevance_score": round(r.score, 4),
            "chunk_position": f"{r.chunk.chunk_index + 1}/{r.chunk.total_chunks}",
            "snippet": r.snippet,
            "full_chunk_text": r.chunk.text,
        })

    logger.info("🔍 TOOL RESULT: workspace_search | %d results for query=%r", len(results), params.query)
    return json.dumps(output, indent=2)


@mcp.tool(
    name="workspace_get_document",
    annotations={
        "title": "Get Full Document",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
@_busy_guard
async def workspace_get_document(params: GetDocumentInput) -> str:
    """Retrieve the full text content of a specific document from the workspace.

    Use this tool after workspace_search to read the complete content of a
    document that appeared in search results, or when you know the exact file
    path of a document the user wants to reference.

    Args:
        params (GetDocumentInput): Parameters containing:
            - file_path (str): Path to the document, relative to workspace root

    Returns:
        str: The full document text content, or an error message if not found.
    """
    logger.info("📄 TOOL CALLED: workspace_get_document | file_path=%r", params.file_path)

    text = _index.get_document_text(params.file_path)

    if text is None:
        # Try with workspace prefix
        if _config:
            full_path = str(_config.resolved_path / params.file_path)
            text = _index.get_document_text(full_path)

    if text is None:
        available = _index.indexed_files
        suggestion = ""
        if available:
            # Find closest match
            query_lower = params.file_path.lower()
            matches = [f for f in available if query_lower in f.lower()]
            if matches:
                suggestion = f" Did you mean: {', '.join(matches[:3])}?"

        return json.dumps({
            "error": f"Document not found: {params.file_path}.{suggestion}",
            "available_documents": available[:20],
        })

    logger.info("📄 TOOL RESULT: workspace_get_document | file=%r | %d chars", params.file_path, len(text))
    return json.dumps({
        "file_path": params.file_path,
        "content": text,
        "character_count": len(text),
    }, indent=2)


@mcp.tool(
    name="workspace_list_documents",
    annotations={
        "title": "List Workspace Documents",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
@_busy_guard
async def workspace_list_documents() -> str:
    """List all documents currently indexed in the legal workspace.

    Use this tool to see what documents are available for searching and
    reference. Returns file paths relative to the workspace root.

    Returns:
        str: JSON with list of indexed document paths and count.
    """
    logger.info("📋 TOOL CALLED: workspace_list_documents")

    files = _index.indexed_files
    logger.info("📋 TOOL RESULT: workspace_list_documents | %d documents", len(files))
    return json.dumps({
        "document_count": len(files),
        "documents": files,
        "workspace_path": str(_config.resolved_path) if _config else "unknown",
    }, indent=2)


@mcp.tool(
    name="workspace_status",
    annotations={
        "title": "Workspace Index Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def workspace_status() -> str:
    """Check the health and statistics of the workspace document index.

    Use this to verify the index is working correctly, see how many
    documents are indexed, and check if the file watcher is active.

    Returns:
        str: JSON with index statistics including document count, chunk count,
             watcher status, and workspace path.
    """
    logger.info("📊 TOOL CALLED: workspace_status")

    if _index is None:
        return json.dumps({"status": "index_busy", "message": INDEX_BUSY_MESSAGE,
                           "role": _role, "startup_build": dict(_build_state)})

    try:
        logger.info("📊 TOOL RESULT: workspace_status | %d docs, %d chunks",
                    _index.document_count, _index.chunk_count)
        return json.dumps({
            "status": "healthy",
            "role": _role,
            "startup_build": dict(_build_state),
            "index_type": "SQLite FTS5 (BM25)",
            "workspace_path": str(_config.resolved_path) if _config else "unknown",
            "document_count": _index.document_count,
            "chunk_count": _index.chunk_count,
            "database_size_mb": round(_index.database_size_mb, 2),
            "watcher_active": _watcher.is_running if _watcher else False,
            "indexed_files": _index.indexed_files,
        }, indent=2)
    except sqlite3.OperationalError as e:
        return _busy_response(str(e))


@mcp.tool(
    name="workspace_reindex",
    annotations={
        "title": "Reindex Workspace",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
@_busy_guard
async def workspace_reindex() -> str:
    """Force a complete re-index of all documents in the workspace.

    Use this if you suspect the index is out of date or if many files
    were changed at once. The file watcher normally handles incremental
    updates automatically, so this is rarely needed.

    Returns:
        str: JSON with re-indexing results including timing and counts.
    """
    logger.info("🔄 TOOL CALLED: workspace_reindex")

    if _writer is None:
        return json.dumps({
            "status": "not_indexer",
            "role": _role,
            "message": "Another server process is the indexer for this index and keeps "
                       "it up to date. Its file watcher picks up changes.",
        })

    summary = _writer.build_full_index()
    logger.info("🔄 TOOL RESULT: workspace_reindex | %s", summary)
    return json.dumps({
        "status": "reindex_complete",
        **summary,
    }, indent=2)


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
