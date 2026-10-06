"""The server answers the MCP handshake before the index build finishes.

Claude Code gives a stdio server 30 seconds to answer ``initialize``. The
startup index build used to run inside the lifespan, before the server read
its first message. On a workspace of symlinks into OneDrive, the build hashed
every file by reading it, and each read of a cloud-only ("dataless") file made
macOS download it first. Startup took many minutes and every session saw
"connection timed out after 30000ms".

Two guards:
1. The lifespan yields at once; the build runs in a background thread.
2. The build does not read a dataless file, so it never forces a download.
"""

import asyncio
import os
import threading
import time
import types

import pytest

from legal_workspace_mcp import indexer as indexer_mod
from legal_workspace_mcp import server as server_mod
from legal_workspace_mcp.config import WorkspaceConfig
from legal_workspace_mcp.indexer import DocumentIndex, is_dataless


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    (path / "local.md").write_text("indemnification clause for the licensor")
    (path / "cloud.md").write_text("tranche milestone for the SAFE")
    return path


@pytest.fixture
def config(tmp_path, workspace):
    return WorkspaceConfig(workspace_path=str(workspace),
                           index_path_override=str(tmp_path / "idx" / "index.db"))


def test_is_dataless_reads_the_sf_dataless_flag():
    assert is_dataless(types.SimpleNamespace(st_flags=0x40000000)) is True
    assert is_dataless(types.SimpleNamespace(st_flags=0x40)) is False
    assert is_dataless(types.SimpleNamespace()) is False  # no st_flags (Linux)


def test_build_does_not_read_dataless_files(config, workspace, monkeypatch):
    cloud = str(workspace / "cloud.md")
    real_stat = os.stat

    def fake_is_dataless(st):
        return getattr(st, "_cloud", False)

    class _Stat:
        def __init__(self, st, cloud_flag):
            self._st, self._cloud = st, cloud_flag

        def __getattr__(self, name):
            return getattr(self._st, name)

    from pathlib import Path
    real_path_stat = Path.stat

    def fake_path_stat(self, *a, **kw):
        st = real_path_stat(self, *a, **kw)
        return _Stat(st, str(self) == cloud)

    read_paths = []
    real_extract = indexer_mod.extract_text

    def spy_extract(p):
        read_paths.append(str(p))
        return real_extract(p)

    real_read_bytes = Path.read_bytes

    def spy_read_bytes(self):
        read_paths.append(str(self))
        return real_read_bytes(self)

    monkeypatch.setattr(indexer_mod, "is_dataless", fake_is_dataless)
    monkeypatch.setattr(Path, "stat", fake_path_stat)
    monkeypatch.setattr(Path, "read_bytes", spy_read_bytes)
    monkeypatch.setattr(indexer_mod, "extract_text", spy_extract)

    index = DocumentIndex(config)
    try:
        summary = index.build_full_index()
        assert cloud not in read_paths
        assert str(workspace / "local.md") in read_paths
        assert summary["skipped_cloud_only"] == 1
        assert index.document_count == 1
    finally:
        index.close()


def test_lifespan_yields_before_index_build_finishes(config, workspace, monkeypatch):
    release = threading.Event()
    started = threading.Event()

    def slow_build(self):
        started.set()
        release.wait(10)
        return {"documents": 0, "chunks": 0, "elapsed_seconds": 0, "errors": []}

    monkeypatch.setattr(DocumentIndex, "build_full_index", slow_build)
    monkeypatch.setattr(server_mod, "load_config", lambda _p=None: config)
    monkeypatch.setattr(server_mod.sys, "argv", ["legal-workspace-mcp"])

    async def enter():
        cm = server_mod.server_lifespan(None)
        t0 = time.monotonic()
        await asyncio.wait_for(cm.__aenter__(), timeout=2)
        assert time.monotonic() - t0 < 2, "lifespan blocked on the index build"
        try:
            assert started.wait(2), "index build never started"
            status = server_mod._build_state["status"]
            assert status == "indexing"
        finally:
            release.set()
            await cm.__aexit__(None, None, None)

    asyncio.run(enter())
