"""Search inside one folder, and a failed write never keeps the index locked.

1. Skills draft from the firm's form library (a "forms" folder in the
   workspace). Across hundreds of matter files, BM25 puts a form below matter
   copies of it, so workspace_search takes a path_prefix that keeps results
   inside one folder.

2. Every session runs its own server on one shared index file. If an insert
   fails after the implicit transaction began, the connection must roll back.
   Otherwise it keeps the write lock for the life of the process, and every
   other server (and every new server's start-up) waits on it.
"""

import asyncio
import json
import sqlite3

import pytest

from legal_workspace_mcp import server as server_mod
from legal_workspace_mcp.config import WorkspaceConfig
from legal_workspace_mcp.indexer import DocumentIndex


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    (ws / "forms" / "M&A Forms").mkdir(parents=True)
    (ws / "holon" / "matter").mkdir(parents=True)
    (ws / "forms" / "M&A Forms" / "spa.md").write_text("stock purchase agreement indemnification escrow")
    for i in range(5):
        (ws / "holon" / "matter" / f"spa_{i}.md").write_text(
            "stock purchase agreement indemnification escrow indemnification escrow")
    return ws


@pytest.fixture
def index(tmp_path, workspace):
    cfg = WorkspaceConfig(workspace_path=str(workspace),
                          index_path_override=str(tmp_path / "idx" / "index.db"))
    idx = DocumentIndex(cfg)
    idx.build_full_index()
    yield idx
    idx.close()


def test_search_path_prefix_keeps_results_in_folder(index, workspace):
    hits = index.search("indemnification escrow", max_results=3, path_prefix="forms")
    assert hits, "the form must be found"
    assert all(h.chunk.file_path.startswith(str(workspace / "forms") + "/") for h in hits)


def test_path_prefix_absolute_and_trailing_slash(index, workspace):
    a = index.search("escrow", path_prefix=str(workspace / "forms") + "/")
    b = index.search("escrow", path_prefix="forms")
    assert [h.chunk.file_path for h in a] == [h.chunk.file_path for h in b]


def test_path_prefix_is_not_a_name_prefix(index, workspace):
    (workspace / "forms_old").mkdir()
    (workspace / "forms_old" / "x.md").write_text("escrow")
    index.build_full_index()
    hits = index.search("escrow", max_results=20, path_prefix="forms")
    assert not any("forms_old" in h.chunk.file_path for h in hits)


def test_path_prefix_wildcards_are_literal(index, workspace):
    # "%" and "_" are LIKE wildcards; a folder name with them must match literally.
    assert index.search("escrow", path_prefix="f%") == []
    assert index.search("escrow", path_prefix="M&A Forms") == []


def test_no_prefix_searches_everything(index):
    hits = index.search("indemnification escrow", max_results=10)
    assert any("/holon/" in h.chunk.file_path for h in hits)


def test_failed_insert_releases_the_lock(index, workspace, monkeypatch):
    doc = workspace / "holon" / "matter" / "spa_0.md"
    doc.write_text("changed text so the hash differs")
    real = index._chunk_text

    class Bad:  # fails after the first chunk is inserted, inside the transaction
        def __getattr__(self, name):
            raise RuntimeError("insert failed")

    monkeypatch.setattr(index, "_chunk_text", lambda text, path: real(text, path) + [Bad()])
    with pytest.raises(RuntimeError):
        index.update_file(doc)

    assert not index._conn.in_transaction
    other = sqlite3.connect(str(index.config.index_path), timeout=0.5)
    try:
        other.execute("CREATE TABLE IF NOT EXISTS probe(x)")
        other.commit()  # raises "database is locked" if the lock was kept
    finally:
        other.close()
    # The old rows survive a failed update: delete and insert are one transaction.
    assert index.search("stock purchase", max_results=20, path_prefix="holon/matter")


def test_tool_accepts_path_prefix(index, monkeypatch):
    monkeypatch.setattr(server_mod, "_index", index)
    monkeypatch.setattr(server_mod, "_config", index.config)
    out = asyncio.run(server_mod.workspace_search(
        server_mod.SearchInput(query="escrow", path_prefix="forms")))
    data = json.loads(out)
    paths = [r["file"] for r in data["results"]]
    assert paths and all(p.startswith("forms/") for p in paths)
