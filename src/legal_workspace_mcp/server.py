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

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

from .config import WorkspaceConfig, load_config
from .indexer import DocumentIndex
from .watcher import WorkspaceWatcher

# Configure logging to stderr (required for stdio transport)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Module-level state (initialized in lifespan)
_index: Optional[DocumentIndex] = None
_watcher: Optional[WorkspaceWatcher] = None
_config: Optional[WorkspaceConfig] = None


@asynccontextmanager
async def server_lifespan(mcp_server):
    """Initialize index and file watcher on server startup."""
    global _index, _watcher, _config

    # Load config - workspace path comes from CLI arg or env var
    workspace_path = None
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
    _config = load_config(workspace_path)

    logger.info("Workspace: %s", _config.resolved_path)

    # Create index (opens or creates SQLite database)
    _index = DocumentIndex(_config)

    # Build index synchronously — SQLite is fast enough and the DB
    # persists across restarts, so only changed files are re-indexed
    summary = _index.build_full_index()
    logger.info("Index ready: %s", summary)

    # Start file watcher for live updates
    _watcher = WorkspaceWatcher(_index, _config)
    _watcher.start()

    yield {"index": _index, "watcher": _watcher, "config": _config}

    # Cleanup
    if _watcher:
        _watcher.stop()
    if _index:
        _index.close()


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
async def workspace_search(params: SearchInput) -> str:
    """Search across all documents in the legal workspace for relevant content.

    Use this tool when you need to find specific legal language, clause examples,
    drafting guidance, or reference material from the user's document collection.
    Results include relevant text snippets with source file information.

    Args:
        params (SearchInput): Search parameters containing:
            - query (str): Natural language search query
            - max_results (int): Maximum results to return (default 5)

    Returns:
        str: JSON with search results including file names, relevance scores,
             and text snippets from matching document sections.
    """
    logger.info("🔍 TOOL CALLED: workspace_search | query=%r | max_results=%d", params.query, params.max_results)

    if _index is None:
        return json.dumps({"error": "Index not initialized. Server may still be starting."})

    results = _index.search(params.query, max_results=params.max_results)

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

    if _index is None:
        return json.dumps({"error": "Index not initialized."})

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
async def workspace_list_documents() -> str:
    """List all documents currently indexed in the legal workspace.

    Use this tool to see what documents are available for searching and
    reference. Returns file paths relative to the workspace root.

    Returns:
        str: JSON with list of indexed document paths and count.
    """
    logger.info("📋 TOOL CALLED: workspace_list_documents")

    if _index is None:
        return json.dumps({"error": "Index not initialized."})

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
        return json.dumps({"status": "initializing", "message": "Index not yet ready."})

    logger.info("📊 TOOL RESULT: workspace_status | %d docs, %d chunks", _index.document_count, _index.chunk_count)
    return json.dumps({
        "status": "healthy",
        "index_type": "SQLite FTS5 (BM25)",
        "workspace_path": str(_config.resolved_path) if _config else "unknown",
        "document_count": _index.document_count,
        "chunk_count": _index.chunk_count,
        "database_size_mb": round(_index.database_size_mb, 2),
        "watcher_active": _watcher.is_running if _watcher else False,
        "indexed_files": _index.indexed_files,
    }, indent=2)


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
async def workspace_reindex() -> str:
    """Force a complete re-index of all documents in the workspace.

    Use this if you suspect the index is out of date or if many files
    were changed at once. The file watcher normally handles incremental
    updates automatically, so this is rarely needed.

    Returns:
        str: JSON with re-indexing results including timing and counts.
    """
    logger.info("🔄 TOOL CALLED: workspace_reindex")

    if _index is None:
        return json.dumps({"error": "Index not initialized."})

    summary = _index.build_full_index()
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
