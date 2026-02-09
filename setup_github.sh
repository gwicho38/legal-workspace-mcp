#!/bin/bash
# Setup script: Create a GitHub repo and push this project
# Run this from the legal-workspace-mcp directory

set -e

# 1. Initialize git
git init
git add -A
git commit -m "Initial commit: Legal Workspace MCP Server

- Self-updating document indexer with TF-IDF search
- File watcher for automatic index updates (watchdog)
- Support for PDF, DOCX, MD, TXT, HTML, CSV, JSON
- 5 MCP tools: search, get_document, list_documents, status, reindex
- Configurable chunking, exclusions, and result limits
- Persisted index for fast startup"

# 2. Create GitHub repo (requires gh CLI: brew install gh)
#    Uncomment the line you want:

# Public repo:
gh repo create legal-workspace-mcp --public --source=. --remote=origin --push

# Private repo:
# gh repo create legal-workspace-mcp --private --source=. --remote=origin --push

echo ""
echo "✅ Repository created and pushed!"
echo ""
echo "Next steps:"
echo "  1. Install: pip install -e ."
echo "  2. Configure Claude Desktop (see README.md)"
echo "  3. Point it at your legal documents directory"
