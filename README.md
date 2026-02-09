# Legal Workspace MCP Server

An MCP (Model Context Protocol) server that gives Claude **self-updating access** to a local directory of documents — templates, articles, opinions, drafting guides, and more — so it can reference them automatically when answering questions.

Built for legal professionals, but works with **any document collection**.

## How It Works

1. **You point it at a directory** containing your documents (PDFs, DOCX, Markdown, HTML, TXT, etc.).
2. **It indexes everything** using TF-IDF for fast, relevant search.
3. **It watches for changes** — add, edit, or delete files and the index updates automatically.
4. **Claude uses it as context** — when you ask a question, Claude can search your documents and pull in the most relevant passages without you uploading anything.

## Install

> **Requirement**: You must specify a target directory. The server watches **one directory at a time** (including all subdirectories). To change directories, re-register the server with a different path.

### Option A: Install from PyPI (recommended)

```bash
pip install legal-workspace-mcp
```

### Option B: Install from GitHub

```bash
pip install git+https://github.com/gwicho38/legal-workspace-mcp.git
```

### Option C: Run without installing (uvx)

```bash
uvx legal-workspace-mcp /path/to/your/documents
```

### Option D: Clone and install locally

```bash
git clone https://github.com/gwicho38/legal-workspace-mcp.git
cd legal-workspace-mcp
pip install -e .
```

## Setup

### Claude Code (CLI)

```bash
claude mcp add legal-workspace -- legal-workspace-mcp /path/to/your/documents
```

That's it. The server starts automatically with every `claude` session.

**To change the target directory**, remove and re-add:

```bash
claude mcp remove legal-workspace
claude mcp add legal-workspace -- legal-workspace-mcp /new/path/to/documents
```

### Claude Desktop

Add this to your Claude Desktop config file:

| OS | Config path |
|----|-------------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

```json
{
  "mcpServers": {
    "legal-workspace": {
      "command": "legal-workspace-mcp",
      "args": ["/path/to/your/documents"]
    }
  }
}
```

### Alternative: Environment Variable

Instead of passing the path as an argument:

```bash
export LEGAL_WORKSPACE_PATH="/path/to/your/documents"
```

## Directory Management

The server watches **exactly one directory** (recursively) per instance. Here's how directory configuration works:

### Specifying the directory (required)

The target directory is resolved at startup with this priority:

| Priority | Method | Example |
|----------|--------|---------|
| 1 (highest) | CLI argument | `legal-workspace-mcp /path/to/docs` |
| 2 | Environment variable | `LEGAL_WORKSPACE_PATH=/path/to/docs` |
| 3 | Config file | `~/.legal-workspace-mcp.json` |

If no directory is configured, the server exits with an error — it will not silently default to a directory.

### What gets indexed

- All files matching supported extensions (see below) in the target directory and all subdirectories
- Symlinks are resolved automatically
- Hidden files (starting with `.`) are skipped
- Certain directories are excluded by default: `.git`, `__pycache__`, `node_modules`, `.venv`, `venv`

### Changing the target directory

Since the server is started as a subprocess by Claude, you change the directory by updating your MCP registration:

```bash
# Claude Code
claude mcp remove legal-workspace
claude mcp add legal-workspace -- legal-workspace-mcp /new/path

# Claude Desktop
# Edit the config JSON file and change the path in "args"
```

The old index is persisted inside the watched directory (`.legal_workspace_index.json`). When you point at a new directory, a fresh index is built automatically.

### Live updates

The server uses [watchdog](https://github.com/gorakhargosh/watchdog) to monitor the directory. Changes are processed automatically:

| Action | Behavior |
|--------|----------|
| Add a file | Indexed within ~2 seconds |
| Edit a file | Re-indexed with new content |
| Delete a file | Removed from index |
| Rename a file | Old entry removed, new entry indexed |

Changes are debounced (2-second quiet period) to avoid thrashing during active editing.

### Optional fine-tuning

Place a `.legal-workspace-mcp.json` file inside the target directory:

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

## Available Tools

Once configured, Claude gains access to these tools:

| Tool | Description |
|------|-------------|
| `workspace_search` | Search all documents for relevant content by topic or keyword |
| `workspace_get_document` | Retrieve the full text of a specific document |
| `workspace_list_documents` | List all indexed documents in the workspace |
| `workspace_status` | Check index health, document count, and watcher status |
| `workspace_reindex` | Force a complete re-index (rarely needed) |

## Supported File Types

| Format | Extensions |
|--------|------------|
| Plain text | `.txt`, `.md`, `.markdown` |
| PDF | `.pdf` |
| Word | `.docx` |
| HTML | `.html`, `.htm` |
| Rich Text | `.rtf` |
| Data | `.json`, `.csv` |

## Example Usage

Once set up, ask Claude things like:

- *"Draft an indemnification clause for a software license agreement, using the style from my templates."*
- *"What do my reference articles say about force majeure provisions post-COVID?"*
- *"Help me revise this non-compete — check my drafting guides for best practices."*
- *"Compare the NDA templates I have and suggest the best one for a mutual NDA."*

Claude will automatically search your workspace documents and incorporate relevant context.

## Architecture

```
Claude (Desktop / Code / API)
    │
    ▼  stdio transport
Legal Workspace MCP Server
    ├── FastMCP (tool definitions)
    ├── Document Index (TF-IDF + chunks)
    ├── Watchdog (filesystem monitoring)
    └── Text Extractors (PDF, DOCX, MD, HTML)
            │
            ▼
    Your Document Directory
    ├── templates/
    ├── articles/
    └── opinions/
```

## Development

```bash
git clone https://github.com/gwicho38/legal-workspace-mcp.git
cd legal-workspace-mcp
pip install -e .

# Run directly
legal-workspace-mcp /path/to/test/documents

# Test with MCP Inspector
npx @modelcontextprotocol/inspector legal-workspace-mcp /path/to/test/documents
```

## License

MIT
