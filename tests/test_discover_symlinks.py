"""Discovery must traverse symlinked directories.

The workspace root is a hub: `~/legal` holds little of its own, and points at the
real document trees through symlinks (`Matters` -> a synced OneDrive mount, and so
on). `os.walk` does not follow symlinks by default, so a hub workspace indexed
zero documents while 1,396 sat one level behind the link.

These tests pin the traversal behaviour, and pin the two things that make
following links safe: the same real file is never indexed twice, and a symlink
cycle terminates.
"""

import os

import pytest

from legal_workspace_mcp.config import WorkspaceConfig
from legal_workspace_mcp.indexer import DocumentIndex


def _index_for(path):
    cfg = WorkspaceConfig(workspace_path=str(path))
    return DocumentIndex(cfg)


def _write(path, text="Consulting Agreement. Governing law: Florida."):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_discovers_files_behind_a_symlinked_directory(tmp_path):
    """The hub case: everything real lives behind a symlink."""
    real = tmp_path / "real_matters"
    _write(real / "client_a" / "agreement.md")
    _write(real / "client_b" / "engagement_letter.md")

    hub = tmp_path / "hub"
    hub.mkdir()
    (hub / "Matters").symlink_to(real, target_is_directory=True)

    found = _index_for(hub)._discover_files(hub)
    names = sorted(p.name for p in found)

    assert names == ["agreement.md", "engagement_letter.md"], (
        f"expected both files behind the symlink, found {names}"
    )


def test_discovers_files_behind_nested_symlinks(tmp_path):
    """A hub of hubs — several firms, each its own linked root."""
    holon = _write(tmp_path / "holon" / "matter" / "nda.md").parent.parent
    jzlaw = _write(tmp_path / "jzlaw" / "onehouse" / "apa.md").parent.parent

    hub = tmp_path / "hub"
    hub.mkdir()
    (hub / "Holon").symlink_to(holon, target_is_directory=True)
    (hub / "JZLaw").symlink_to(jzlaw, target_is_directory=True)

    found = _index_for(hub)._discover_files(hub)
    assert sorted(p.name for p in found) == ["apa.md", "nda.md"]


def test_same_file_reached_by_two_paths_is_returned_once(tmp_path):
    """`current/` links back into an already-walked tree. Do not index twice."""
    real = tmp_path / "real"
    _write(real / "matter" / "deed.md")

    hub = tmp_path / "hub"
    hub.mkdir()
    (hub / "Matters").symlink_to(real, target_is_directory=True)
    # a second route to the very same file
    (hub / "current").symlink_to(real / "matter", target_is_directory=True)

    found = _index_for(hub)._discover_files(hub)
    resolved = [os.path.realpath(p) for p in found]

    assert len(found) == 1, f"deed.md indexed {len(found)} times via two paths"
    assert len(set(resolved)) == 1


def test_symlink_cycle_terminates(tmp_path):
    """A link pointing at its own ancestor must not hang the indexer."""
    root = tmp_path / "root"
    _write(root / "sub" / "memo.md")
    (root / "sub" / "loop").symlink_to(root, target_is_directory=True)

    found = _index_for(root)._discover_files(root)

    assert [p.name for p in found] == ["memo.md"]


def test_broken_symlink_is_ignored(tmp_path):
    """A dangling link — an unmounted drive — must not raise."""
    root = tmp_path / "root"
    _write(root / "memo.md")
    (root / "gone").symlink_to(tmp_path / "does_not_exist", target_is_directory=True)

    found = _index_for(root)._discover_files(root)
    assert [p.name for p in found] == ["memo.md"]
