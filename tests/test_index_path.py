"""The index database must be able to live outside the workspace.

The workspace sits on an external SSD that detaches without warning. SQLite
memory-maps the WAL "-shm" file next to the database; when the disk goes, the
next read faults with SIGBUS and kills the server. An override lets the index
live on the internal disk while the documents stay where they are.
"""

import pytest

from legal_workspace_mcp.config import (
    INDEX_FILENAME,
    INDEX_PATH_ENV_VAR,
    WorkspaceConfig,
    load_config,
)
from legal_workspace_mcp.indexer import DocumentIndex


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(INDEX_PATH_ENV_VAR, raising=False)


def test_default_index_path_is_inside_workspace(workspace):
    config = load_config(str(workspace))
    assert config.index_path == workspace.resolve() / INDEX_FILENAME


def test_env_var_overrides_index_path(tmp_path, workspace, monkeypatch):
    target = tmp_path / "internal" / "index.db"
    monkeypatch.setenv(INDEX_PATH_ENV_VAR, str(target))
    config = load_config(str(workspace))
    assert config.index_path == target.resolve()


def test_override_expands_user_home(workspace, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(INDEX_PATH_ENV_VAR, "~/idx/index.db")
    config = load_config(str(workspace))
    assert config.index_path == (tmp_path / "idx" / "index.db").resolve()


def test_empty_env_var_keeps_default(workspace, monkeypatch):
    monkeypatch.setenv(INDEX_PATH_ENV_VAR, "")
    config = load_config(str(workspace))
    assert config.index_path == workspace.resolve() / INDEX_FILENAME


def test_indexer_writes_to_override_and_not_workspace(tmp_path, workspace):
    (workspace / "doc.md").write_text("indemnification clause")
    target = tmp_path / "internal" / "nested" / "index.db"
    config = WorkspaceConfig(workspace_path=str(workspace), index_path_override=str(target))

    index = DocumentIndex(config)
    try:
        index.build_full_index()
        assert index.search("indemnification")
    finally:
        index.close()

    assert target.exists()
    assert not (workspace / INDEX_FILENAME).exists()
