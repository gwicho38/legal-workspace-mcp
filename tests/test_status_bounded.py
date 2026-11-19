"""workspace_status stays fast and small on a large index (issue #14).

On the shared index (34,133 files, 942,414 chunks) workspace_status took 86 s
and returned 8 MB: COUNT(*) over the FTS5 chunks table scans every row, it was
read twice, and the payload listed every indexed path. Claude Code drops a
stdio server whose reply passes 16 MB, so the whole server went down.

Rules tested here:
1. chunk_count never scans the chunks table; it sums file_meta.total_chunks.
2. workspace_status returns counts, not the file list.
3. workspace_list_documents pages its result.
"""

import asyncio
import json

import pytest

from legal_workspace_mcp import server as server_mod
from legal_workspace_mcp.config import WorkspaceConfig
from legal_workspace_mcp.indexer import DocumentIndex

N_FILES = 650


@pytest.fixture
def index(tmp_path):
    ws = tmp_path / "workspace"
    (ws / "forms").mkdir(parents=True)
    (ws / "matters").mkdir()
    for i in range(N_FILES):
        folder = "forms" if i < 50 else "matters"
        (ws / folder / f"doc_{i:04d}.md").write_text(f"agreement number {i} " * 5)
    cfg = WorkspaceConfig(workspace_path=str(ws),
                          index_path_override=str(tmp_path / "idx" / "index.db"))
    idx = DocumentIndex(cfg)
    idx.build_full_index()
    yield idx
    idx.close()


@pytest.fixture
def server(index, monkeypatch):
    monkeypatch.setattr(server_mod, "_index", index)
    monkeypatch.setattr(server_mod, "_config", index.config)
    monkeypatch.setattr(server_mod, "_watcher", None)
    return server_mod


def test_chunk_count_does_not_scan_chunks_table(index):
    seen = []
    index._conn.set_trace_callback(seen.append)
    try:
        n = index.chunk_count
    finally:
        index._conn.set_trace_callback(None)
    expected = index._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert n == expected
    assert not any("FROM chunks" in s for s in seen), seen


def test_status_has_counts_not_file_list(server):
    raw = asyncio.run(server.workspace_status())
    data = json.loads(raw)
    assert data["document_count"] == N_FILES
    assert data["chunk_count"] >= N_FILES
    assert "indexed_files" not in data
    assert len(raw) < 4096


def test_status_reads_chunk_count_once(server, index, monkeypatch):
    calls = []
    real = type(index).chunk_count.fget

    def counting(self):
        calls.append(1)
        return real(self)

    monkeypatch.setattr(type(index), "chunk_count", property(counting))
    asyncio.run(server.workspace_status())
    assert len(calls) == 1


def test_list_documents_pages(server):
    first = json.loads(asyncio.run(server.workspace_list_documents()))
    assert first["document_count"] == N_FILES
    assert len(first["documents"]) == server.LIST_PAGE_SIZE
    assert first["next_offset"] == server.LIST_PAGE_SIZE

    rest = json.loads(asyncio.run(server.workspace_list_documents(
        server.ListDocumentsInput(offset=first["next_offset"]))))
    assert len(rest["documents"]) == N_FILES - server.LIST_PAGE_SIZE
    assert rest["next_offset"] is None
    assert not set(first["documents"]) & set(rest["documents"])


def test_list_documents_path_prefix(server):
    data = json.loads(asyncio.run(server.workspace_list_documents(
        server.ListDocumentsInput(path_prefix="forms"))))
    assert data["document_count"] == 50
    assert all(p.startswith("forms/") for p in data["documents"])
