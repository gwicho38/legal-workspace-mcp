"""File discovery must follow symlinked directories.

The workspace root is a hub of symlinks into each firm's cloud-synced folder
(~/legal/holon/holon_onedrive -> the Holon Law Partners OneDrive). os.walk does not
descend into symlinked directories by default, so an entire firm's documents were
silently missing from the index while status reported "healthy".
"""



import pytest

from legal_workspace_mcp.config import WorkspaceConfig
from legal_workspace_mcp.indexer import DocumentIndex


@pytest.fixture
def indexer(tmp_path):
    (tmp_path / "workspace").mkdir(exist_ok=True)
    return DocumentIndex(WorkspaceConfig(workspace_path=str(tmp_path / "workspace")))


def test_discovers_files_through_a_symlinked_directory(tmp_path, indexer):
    workspace = tmp_path / "workspace"
    (workspace / "direct").mkdir(parents=True, exist_ok=True)
    (workspace / "direct" / "here.md").write_text("in the workspace")

    outside = tmp_path / "onedrive" / "Matters"
    outside.mkdir(parents=True)
    (outside / "matter-intake.md").write_text("behind a symlink")
    (workspace / "firm_onedrive").symlink_to(outside, target_is_directory=True)

    found = {p.name for p in indexer._discover_files(workspace)}
    assert "here.md" in found
    assert "matter-intake.md" in found, "symlinked firm folder was skipped"


def test_nested_symlinked_directories_are_followed(tmp_path, indexer):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    outside = tmp_path / "cloud"
    (outside / "client" / "matter").mkdir(parents=True)
    (outside / "client" / "matter" / "deep.md").write_text("two levels down")
    (workspace / "link").symlink_to(outside, target_is_directory=True)

    assert "deep.md" in {p.name for p in indexer._discover_files(workspace)}


def test_symlink_loop_does_not_hang_or_duplicate(tmp_path, indexer):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "doc.md").write_text("only file")
    (workspace / "loop").symlink_to(workspace, target_is_directory=True)

    found = indexer._discover_files(workspace)
    assert [p.name for p in found] == ["doc.md"]


def test_excluded_directories_are_still_skipped_behind_a_symlink(tmp_path, indexer):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    outside = tmp_path / "cloud"
    (outside / "node_modules").mkdir(parents=True)
    (outside / "node_modules" / "junk.md").write_text("noise")
    (outside / "keep.md").write_text("signal")
    (workspace / "link").symlink_to(outside, target_is_directory=True)

    found = {p.name for p in indexer._discover_files(workspace)}
    assert "keep.md" in found
    assert "junk.md" not in found
