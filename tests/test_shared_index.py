"""Several server processes share one index database without crashing.

Each Claude session starts its own server, and every server opens the same
index file. In WAL mode SQLite memory-maps a shared "-shm" file. When one
process re-initializes that file while another still maps it, the next read
in the other process faults with SIGBUS (issue #7). The index must therefore
use a rollback journal, which has no shared mapping, and must wait for locks
instead of failing while another process writes.
"""

import sqlite3
import subprocess
import sys
import textwrap
import time

import pytest

from legal_workspace_mcp.config import WorkspaceConfig
from legal_workspace_mcp.indexer import DocumentIndex


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


def _journal_mode(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()


def test_index_has_no_shared_memory_file(config):
    index = DocumentIndex(config)
    try:
        index.build_full_index()
        assert index.search("indemnification")
        shm = config.index_path.with_name(config.index_path.name + "-shm")
        assert not shm.exists(), "WAL shared-memory file exists; it is the SIGBUS source"
    finally:
        index.close()
    assert _journal_mode(config.index_path) == "delete"


def test_existing_wal_index_is_converted(config):
    config.index_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(config.index_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t(x)")
    conn.commit()
    conn.close()
    assert _journal_mode(config.index_path) == "wal"

    index = DocumentIndex(config)
    try:
        assert _journal_mode(config.index_path) == "delete"
    finally:
        index.close()


WRITER = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    from legal_workspace_mcp.config import WorkspaceConfig
    from legal_workspace_mcp.indexer import DocumentIndex
    ws, db, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
    index = DocumentIndex(WorkspaceConfig(workspace_path=ws, index_path_override=db))
    end = time.time() + seconds
    n = 0
    while time.time() < end:
        p = Path(ws) / f"w{n % 20}.md"
        p.write_text(f"royalty schedule revision {n} " * 400)
        index.update_file(p)
        n += 1
    index.close()
    print(n)
""")


def test_reader_survives_concurrent_writer_process(config, workspace):
    index = DocumentIndex(config)
    index.build_full_index()
    writer = subprocess.Popen(
        [sys.executable, "-c", WRITER, str(workspace), str(config.index_path), "3"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    errors = []
    try:
        end = time.time() + 3
        while time.time() < end:
            try:
                index.search("indemnification")
                index.document_count
            except sqlite3.Error as e:  # "database is locked" without a busy timeout
                errors.append(repr(e))
    finally:
        out, err = writer.communicate(timeout=60)
        index.close()
    assert writer.returncode == 0, err
    assert int(out.strip()) > 0
    assert not errors, errors[:3]
