"""Many servers share one index; a busy database never kills a server (issue #12).

Each Claude Code session starts its own stdio server on the same index file.
With about 14 copies running, a new copy died at startup with
``sqlite3.OperationalError: database is locked``: ``_init_db`` wrote to the
database on every start, inside the lifespan, so the handshake waited on a
write lock and the busy timeout expired. Every copy also ran its own full
index build and file watcher, so the copies contended for the lock all the time.

Guards:
(a) a server answers ``initialize`` and ``tools/list`` while another process
    holds the database locked, and its tool calls say "busy" until it can read;
(b) only one server per index path becomes the indexer, and another takes
    over when it exits;
(c) a search during a held lock returns results or the busy message;
(d) opening an index whose schema is current issues no write.
"""

import asyncio
import re
import sqlite3
import time

import pytest

from legal_workspace_mcp import indexer as indexer_mod
from legal_workspace_mcp import server as server_mod
from legal_workspace_mcp.config import WorkspaceConfig
from legal_workspace_mcp.indexer import DocumentIndex
from legal_workspace_mcp.server import SearchInput
from mcp_stdio import StdioServer

WRITE_SQL = re.compile(
    r"^\s*(CREATE|INSERT|REPLACE|UPDATE|DELETE|DROP|ALTER|"
    r"BEGIN\s+(IMMEDIATE|EXCLUSIVE)|PRAGMA\s+journal_mode\s*=)",
    re.IGNORECASE,
)


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    (path / "doc.md").write_text("indemnification clause for the licensor")
    return path


@pytest.fixture
def config(tmp_path, workspace):
    return WorkspaceConfig(workspace_path=str(workspace),
                           index_path_override=str(tmp_path / "idx" / "index.db"))


@pytest.fixture
def built_index(config):
    """An index on disk with the current schema and one document in it."""
    index = DocumentIndex(config)
    index.build_full_index()
    index.close()
    return config.index_path


def _lock(db_path, mode):
    """Hold a write lock on the index from this (test) process."""
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute(f"BEGIN {mode}")
    return conn


def _short_timeouts(monkeypatch, seconds=1.0):
    monkeypatch.setattr(indexer_mod, "BUSY_TIMEOUT_SECONDS", seconds)
    monkeypatch.setattr(indexer_mod, "READ_BUSY_TIMEOUT_SECONDS", seconds, raising=False)


def _is_busy(payload):
    return payload.get("status") == "index_busy" and "retry" in payload.get("error", "").lower()


# --- (d) no write when the schema is current ---------------------------------


def test_init_on_current_schema_issues_no_write(built_index, config, monkeypatch):
    statements = []
    real_connect = sqlite3.connect

    def tracing_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    holder = _lock(built_index, "IMMEDIATE")  # another writer is mid-transaction
    monkeypatch.setattr(indexer_mod.sqlite3, "connect", tracing_connect)
    _short_timeouts(monkeypatch, 2.0)
    try:
        started = time.monotonic()
        index = DocumentIndex(config)
        elapsed = time.monotonic() - started
        try:
            assert index.document_count == 1
        finally:
            index.close()
    finally:
        holder.rollback()
        holder.close()

    writes = [s for s in statements if WRITE_SQL.match(s)]
    assert not writes, f"init wrote to a current index: {writes}"
    assert elapsed < 1.5, f"init waited {elapsed:.1f}s on another writer's lock"


def test_init_creates_schema_on_a_new_index(config):
    index = DocumentIndex(config)
    try:
        assert index.document_count == 0
        version = index._conn.execute(
            "SELECT value FROM schema_info WHERE key='version'").fetchone()[0]
        assert version == str(indexer_mod.SCHEMA_VERSION)
    finally:
        index.close()


def test_read_only_index_refuses_writes(built_index, config):
    index = DocumentIndex(config, read_only=True)
    try:
        assert index.search("indemnification")
        with pytest.raises(sqlite3.OperationalError):
            index._conn.execute("DELETE FROM file_meta")
    finally:
        index.close()


# --- (c) searches during a held lock -----------------------------------------


@pytest.fixture
def server_index(built_index, config, monkeypatch):
    _short_timeouts(monkeypatch)
    index = DocumentIndex(config)
    monkeypatch.setattr(server_mod, "_index", index)
    monkeypatch.setattr(server_mod, "_config", config)
    yield index
    index.close()


def _search(query="indemnification"):
    import json
    return json.loads(asyncio.run(server_mod.workspace_search(SearchInput(query=query))))


def test_search_while_another_process_writes_returns_results(server_index, built_index):
    holder = _lock(built_index, "IMMEDIATE")
    try:
        payload = _search()
    finally:
        holder.rollback()
        holder.close()
    assert payload.get("result_count") == 1, payload


def test_search_while_database_is_exclusively_locked_says_busy(server_index, built_index):
    holder = _lock(built_index, "EXCLUSIVE")
    try:
        started = time.monotonic()
        payload = _search()
        elapsed = time.monotonic() - started
    finally:
        holder.rollback()
        holder.close()
    assert _is_busy(payload), payload
    assert elapsed < 5, elapsed
    assert _search().get("result_count") == 1  # works again once the lock is gone


def test_tool_call_before_index_opens_says_busy(monkeypatch, config):
    monkeypatch.setattr(server_mod, "_index", None)
    monkeypatch.setattr(server_mod, "_config", config)
    assert _is_busy(_search())


# --- (a) the handshake survives a locked database ----------------------------


def test_server_answers_handshake_while_database_is_locked(built_index, workspace):
    holder = _lock(built_index, "EXCLUSIVE")  # no reads, no writes
    server = StdioServer(workspace, built_index)
    try:
        resp, elapsed = server.initialize(timeout=8)
        assert "result" in resp, resp
        assert elapsed < 8

        tools, _ = server.request("tools/list", timeout=5)
        names = {t["name"] for t in tools["result"]["tools"]}
        assert "workspace_search" in names

        busy = server.call_tool("workspace_search", {"params": {"query": "indemnification"}})
        assert _is_busy(busy), busy

        holder.rollback()  # the other process finishes
        deadline = time.monotonic() + 30
        payload = busy
        while time.monotonic() < deadline:
            payload = server.call_tool("workspace_search",
                                       {"params": {"query": "indemnification"}})
            if payload.get("result_count"):
                break
            time.sleep(0.5)
        assert payload.get("result_count") == 1, payload
        assert server.proc.poll() is None, "server exited"
    finally:
        holder.close()
        server.close()


# --- (b) one indexer per index path ------------------------------------------


def _role(server):
    return server.call_tool("workspace_status").get("role")


def _wait_roles(servers, timeout=20):
    deadline = time.monotonic() + timeout
    roles = []
    while time.monotonic() < deadline:
        roles = [_role(s) for s in servers]
        if all(r in ("indexer", "reader") for r in roles):
            return roles
        time.sleep(0.5)
    return roles


def test_only_one_server_becomes_indexer_and_another_takes_over(built_index, workspace):
    servers = [StdioServer(workspace, built_index) for _ in range(2)]
    try:
        for s in servers:
            s.initialize(timeout=8)
        roles = _wait_roles(servers)
        assert sorted(roles) == ["indexer", "reader"], roles

        leader = servers[roles.index("indexer")]
        follower = servers[roles.index("reader")]
        leader.close()

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and _role(follower) != "indexer":
            time.sleep(0.5)
        assert _role(follower) == "indexer"
        payload = follower.call_tool("workspace_search",
                                     {"params": {"query": "indemnification"}})
        assert payload.get("result_count") == 1, payload
    finally:
        for s in servers:
            s.close()
