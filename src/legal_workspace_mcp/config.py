"""Configuration for the Legal Workspace MCP Server."""

import os
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# Supported file extensions for document extraction
SUPPORTED_EXTENSIONS: set[str] = {
    ".txt", ".md", ".markdown",
    ".pdf",
    ".docx",
    ".html", ".htm",
    ".rtf",
}

# Default chunk size for splitting documents (in characters)
DEFAULT_CHUNK_SIZE: int = 1500
DEFAULT_CHUNK_OVERLAP: int = 200

# Maximum number of search results to return
DEFAULT_MAX_RESULTS: int = 10

# Debounce delay for file watcher (seconds)
WATCHER_DEBOUNCE_SECONDS: float = 2.0

# Default excluded directories
DEFAULT_EXCLUDED_DIRS: list[str] = [
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".legal_workspace_index", "backend_OLD",
]

# Default excluded file patterns
DEFAULT_EXCLUDED_PATTERNS: list[str] = [
    ".*",  # hidden files
]

# Index persistence filename
INDEX_FILENAME: str = ".legal_workspace_index.json"

# Config file name
CONFIG_FILENAME: str = ".legal-workspace-mcp.json"


@dataclass
class WorkspaceConfig:
    """Configuration for a watched workspace."""

    workspace_path: str
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    max_results: int = DEFAULT_MAX_RESULTS
    excluded_dirs: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDED_DIRS))
    excluded_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDED_PATTERNS))
    file_extensions: set[str] = field(default_factory=lambda: SUPPORTED_EXTENSIONS.copy())

    @property
    def resolved_path(self) -> Path:
        """Return the resolved absolute path of the workspace."""
        return Path(self.workspace_path).expanduser().resolve()

    @property
    def index_path(self) -> Path:
        """Path to the persisted index file."""
        return self.resolved_path / INDEX_FILENAME

    def to_dict(self) -> dict:
        """Serialize config to a dictionary."""
        return {
            "workspace_path": self.workspace_path,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_results": self.max_results,
            "excluded_dirs": self.excluded_dirs,
            "excluded_patterns": self.excluded_patterns,
            "file_extensions": sorted(self.file_extensions),
        }


def load_config(workspace_path: Optional[str] = None) -> WorkspaceConfig:
    """Load configuration, with priority: CLI arg > env var > config file > default.

    Args:
        workspace_path: Explicit workspace path (highest priority).

    Returns:
        WorkspaceConfig with resolved settings.
    """
    # Priority 1: Explicit argument
    if workspace_path:
        path = workspace_path
    # Priority 2: Environment variable
    elif os.environ.get("LEGAL_WORKSPACE_PATH"):
        path = os.environ["LEGAL_WORKSPACE_PATH"]
    # Priority 3: Config file in home directory
    else:
        config_file = Path.home() / CONFIG_FILENAME
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
                path = data.get("workspace_path", "")
            except (json.JSONDecodeError, KeyError):
                path = ""
        else:
            path = ""

    if not path:
        print(
            "Error: No workspace path configured.\n"
            "Set it via:\n"
            "  1. CLI argument: legal-workspace-mcp /path/to/workspace\n"
            "  2. Environment variable: LEGAL_WORKSPACE_PATH=/path/to/workspace\n"
            "  3. Config file: ~/.legal-workspace-mcp.json\n",
            file=sys.stderr,
        )
        sys.exit(1)

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        print(f"Error: Workspace path does not exist or is not a directory: {resolved}", file=sys.stderr)
        sys.exit(1)

    # Load additional settings from config file if present
    config_file = resolved / CONFIG_FILENAME
    extra = {}
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                extra = json.load(f)
        except json.JSONDecodeError:
            pass

    return WorkspaceConfig(
        workspace_path=str(resolved),
        chunk_size=extra.get("chunk_size", DEFAULT_CHUNK_SIZE),
        chunk_overlap=extra.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
        max_results=extra.get("max_results", DEFAULT_MAX_RESULTS),
        excluded_dirs=extra.get("excluded_dirs", DEFAULT_EXCLUDED_DIRS),
        excluded_patterns=extra.get("excluded_patterns", DEFAULT_EXCLUDED_PATTERNS),
    )
