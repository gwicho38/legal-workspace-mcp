# Legal Workspace MCP Server

An MCP (Model Context Protocol) server that gives Claude **self-updating access** to a local directory of legal documents — templates, articles, opinions, and drafting guides — so it can reference them automatically when answering legal drafting questions.

## How It Works

1. **You point it at a directory** containing your legal documents (PDFs, DOCX, Markdown, HTML, TXT, etc.).
2. **It indexes everything** using TF-IDF for fast, relevant search.
3. **It watches for changes** — add, edit, or delete files and the index updates automatically.
4. **Claude uses it as context** — when you ask a legal drafting question, Claude can search your documents and pull in the most relevant passages without you uploading anything.

## Quick Start

### Prerequisites

- Python 3.10+
- [Claude Desktop](https://claude.ai/download) or [Claude Code](https://docs.anthropic.com/en/docs/claude-code)

### Install

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/legal-workspace-mcp.git
cd legal-workspace-mcp

# Install with pip
pip install -e .
```

### Configure Claude Desktop

Add this to your Claude Desktop config file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "legal-workspace": {
      "command": "legal-workspace-mcp",
      "args": ["/path/to/your/legal/documents"]
    }
  }
}
```

Replace `/path/to/your/legal/documents` with the actual path to your legal workspace directory.

### Configure Claude Code

```bash
claude mcp add legal-workspace -- legal-workspace-mcp /path/to/your/legal/documents
```

### Alternative: Environment Variable

Instead of passing the path as an argument, you can set:

```bash
export LEGAL_WORKSPACE_PATH="/path/to/your/legal/documents"
```

## Available Tools

Once configured, Claude gains access to these tools:

| Tool | Description |
|------|-------------|
| `workspace_search` | Search all documents for relevant content by topic or keyword |
| `workspace_get_document` | Retrieve the full text of a specific document |
| `workspace_list_documents` | List all indexed documents in the workspace |
| `workspace_status` | Check index health, document count, and watcher status |
| `workspace_reindex` | Force a complete re-index (rarely needed) |

## Example Conversations

Once set up, you can ask Claude things like:

- *"Draft an indemnification clause for a software license agreement, using the style from my templates."*
- *"What do my reference articles say about force majeure provisions post-COVID?"*
- *"Help me revise this non-compete — check my drafting guides for best practices."*
- *"Compare the NDA templates I have and suggest the best one for a mutual NDA."*

Claude will automatically search your workspace documents and incorporate relevant context.

## Supported File Types

| Format | Extensions |
|--------|------------|
| Plain text | `.txt`, `.md`, `.markdown` |
| PDF | `.pdf` |
| Word | `.docx` |
| HTML | `.html`, `.htm` |
| Rich Text | `.rtf` |
| Data | `.json`, `.csv` |

## Configuration

You can place a `.legal-workspace-mcp.json` file in your workspace directory for fine-tuning:

```json
{
  "chunk_size": 1500,
  "chunk_overlap": 200,
  "max_results": 10,
  "excluded_dirs": [".git", "__pycache__", "node_modules", "archive"],
  "excluded_patterns": [".*"]
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `chunk_size` | 1500 | Characters per chunk for indexing |
| `chunk_overlap` | 200 | Overlap between chunks to preserve context |
| `max_results` | 10 | Default max search results |
| `excluded_dirs` | `[".git", ...]` | Directories to skip |
| `excluded_patterns` | `[".*"]` | File patterns to skip |

## Self-Updating Behavior

The server uses [watchdog](https://github.com/gorakhargosh/watchdog) to monitor your workspace directory. When you:

- **Add a file** → It's indexed within ~2 seconds
- **Edit a file** → The index updates with the new content
- **Delete a file** → It's removed from the index

Changes are debounced (2-second quiet period) to avoid thrashing during active editing.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Claude (Desktop / Code / API)              │
│                                             │
│  "Draft a non-compete clause..."            │
│         │                                   │
│         ▼                                   │
│  ┌──────────────┐                           │
│  │ MCP Protocol │  (stdio transport)        │
│  └──────┬───────┘                           │
└─────────┼───────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────┐
│  Legal Workspace MCP Server                 │
│                                             │
│  ┌────────────┐  ┌──────────────────────┐   │
│  │  FastMCP   │  │  Document Index      │   │
│  │  (tools)   │──│  (TF-IDF + chunks)   │   │
│  └────────────┘  └──────────┬───────────┘   │
│                             │               │
│  ┌────────────┐  ┌──────────▼───────────┐   │
│  │  Watchdog  │──│  Text Extractors     │   │
│  │  (fs watch)│  │  (PDF,DOCX,MD,HTML)  │   │
│  └────────────┘  └──────────────────────┘   │
│                             │               │
└─────────────────────────────┼───────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Your Legal Docs  │
                    │  ~/Legal/         │
                    │  ├── templates/   │
                    │  ├── articles/    │
                    │  └── opinions/    │
                    └───────────────────┘
```

## Development

```bash
# Clone and install in dev mode
git clone https://github.com/YOUR_USERNAME/legal-workspace-mcp.git
cd legal-workspace-mcp
pip install -e .

# Run directly
legal-workspace-mcp /path/to/test/documents

# Test with MCP Inspector
npx @modelcontextprotocol/inspector legal-workspace-mcp /path/to/test/documents
```

## License

MIT
