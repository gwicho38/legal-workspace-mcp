"""Document indexing and search engine using SQLite FTS5.

Handles chunking documents, building a searchable index with BM25 ranking,
and persisting the index to a SQLite database that survives server restarts.
"""

import hashlib
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import WorkspaceConfig, LEGACY_INDEX_FILENAME
from .extractors import extract_text

logger = logging.getLogger(__name__)

# SQLite schema version for future migrations
SCHEMA_VERSION = 1


@dataclass
class DocumentChunk:
    """A chunk of a document with metadata."""

    chunk_id: str
    file_path: str
    file_name: str
    chunk_index: int
    total_chunks: int
    text: str
    file_hash: str
    file_modified: float
    file_size: int


@dataclass
class SearchResult:
    """A search result with relevance score."""

    chunk: DocumentChunk
    score: float
    snippet: str


class DocumentIndex:
    """SQLite FTS5 document index with BM25 ranking.

    The index automatically handles:
    - Chunking documents into searchable pieces
    - BM25-ranked full-text search via SQLite FTS5
    - Persisting the index to a SQLite database
    - Incremental updates when files change
    - File size and chunk limits to bound memory usage
    """

    def __init__(self, config: WorkspaceConfig):
        self.config = config
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database and create tables."""
        db_path = self.config.index_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Performance pragmas
        self._conn.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA cache_size=-2000;
            PRAGMA temp_store=MEMORY;
        """)

        # Create tables
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS file_meta(
                file_path TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                file_modified REAL NOT NULL,
                file_size INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                indexed_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS schema_info(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

        # Create FTS5 table if it doesn't exist
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                file_path UNINDEXED,
                file_name,
                chunk_index UNINDEXED,
                total_chunks UNINDEXED,
                text,
                tokenize='porter unicode61'
            );
        """)

        # Store schema version
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_info(key, value) VALUES('version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.execute("PRAGMA optimize")
                self._conn.close()
            except Exception as e:
                logger.error("Error closing database: %s", e)
            finally:
                self._conn = None

    @property
    def document_count(self) -> int:
        """Number of unique documents in the index."""
        if not self._conn:
            return 0
        row = self._conn.execute("SELECT COUNT(*) FROM file_meta").fetchone()
        return row[0] if row else 0

    @property
    def chunk_count(self) -> int:
        """Total number of chunks in the index."""
        if not self._conn:
            return 0
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    @property
    def indexed_files(self) -> list[str]:
        """List of all indexed file paths (relative to workspace)."""
        if not self._conn:
            return []
        workspace = self.config.resolved_path
        rows = self._conn.execute(
            "SELECT file_path FROM file_meta ORDER BY file_path"
        ).fetchall()
        result = []
        for row in rows:
            try:
                result.append(str(Path(row[0]).relative_to(workspace)))
            except ValueError:
                result.append(row[0])
        return result

    @property
    def database_size_mb(self) -> float:
        """Size of the SQLite database file in MB."""
        db_path = self.config.index_path
        if db_path.exists():
            return db_path.stat().st_size / (1024 * 1024)
        return 0.0

    def build_full_index(self) -> dict:
        """Scan the workspace and build/rebuild the full index.

        Returns:
            Summary dict with counts and timing.
        """
        start = time.time()
        workspace = self.config.resolved_path

        if not workspace.is_dir():
            logger.error("Workspace path does not exist: %s", workspace)
            return {"error": f"Workspace path not found: {workspace}"}

        # Check for legacy JSON index and migrate
        self._migrate_legacy_index()

        # Discover all supported files
        files = self._discover_files(workspace)
        logger.info("Discovered %d files in workspace", len(files))

        # Get currently indexed file hashes for change detection
        existing_hashes: dict[str, str] = {}
        if self._conn:
            rows = self._conn.execute(
                "SELECT file_path, file_hash FROM file_meta"
            ).fetchall()
            existing_hashes = {row[0]: row[1] for row in rows}

        # Track which files are still present
        current_files: set[str] = set()
        errors: list[str] = []

        for file_path in files:
            str_path = str(file_path)
            current_files.add(str_path)

            try:
                stat = file_path.stat()

                # Skip files exceeding size limit
                if stat.st_size > self.config.max_file_size:
                    logger.debug("Skipping %s (%.1fMB > %.1fMB limit)",
                                 file_path.name,
                                 stat.st_size / 1_000_000,
                                 self.config.max_file_size / 1_000_000)
                    continue

                file_hash = self._compute_file_hash(file_path)

                # Skip if unchanged
                if existing_hashes.get(str_path) == file_hash:
                    continue

                text = extract_text(file_path)
                if not text or not text.strip():
                    logger.warning("No text extracted from %s", file_path)
                    continue

                chunks = self._chunk_text(text, file_path)
                self._store_file(file_path, chunks, file_hash, stat)
                logger.info("Indexed %s (%d chunks)", file_path.name, len(chunks))

            except Exception as e:
                logger.error("Error indexing %s: %s", file_path, e)
                errors.append(f"{file_path.name}: {e}")

        # Remove files that no longer exist in workspace
        stale_files = set(existing_hashes.keys()) - current_files
        for stale_path in stale_files:
            self._remove_file_from_db(stale_path)
            logger.info("Removed stale file from index: %s", Path(stale_path).name)

        elapsed = time.time() - start
        summary = {
            "documents": self.document_count,
            "chunks": self.chunk_count,
            "elapsed_seconds": round(elapsed, 2),
            "errors": errors,
        }

        logger.info(
            "Index built: %d documents, %d chunks in %.2fs",
            summary["documents"], summary["chunks"], elapsed,
        )
        return summary

    def update_file(self, file_path: Path) -> None:
        """Update the index for a single file (added or modified).

        Args:
            file_path: Absolute path to the file.
        """
        if not self._is_supported_file(file_path):
            return

        str_path = str(file_path)

        try:
            stat = file_path.stat()
        except FileNotFoundError:
            self._remove_file_from_db(str_path)
            return

        # Skip files exceeding size limit
        if stat.st_size > self.config.max_file_size:
            logger.debug("Skipping %s (exceeds max_file_size)", file_path.name)
            return

        file_hash = self._compute_file_hash(file_path)

        # Skip if unchanged
        if self._conn:
            row = self._conn.execute(
                "SELECT file_hash FROM file_meta WHERE file_path = ?",
                (str_path,),
            ).fetchone()
            if row and row[0] == file_hash:
                return

        # Extract and chunk
        text = extract_text(file_path)
        if text and text.strip():
            chunks = self._chunk_text(text, file_path)
            self._store_file(file_path, chunks, file_hash, stat)
            logger.info("Updated index for %s (%d chunks)", file_path.name, len(chunks))
        else:
            self._remove_file_from_db(str_path)

    def remove_file(self, file_path: Path) -> None:
        """Remove a file from the index.

        Args:
            file_path: Absolute path to the removed file.
        """
        str_path = str(file_path)
        if self._conn:
            row = self._conn.execute(
                "SELECT file_path FROM file_meta WHERE file_path = ?",
                (str_path,),
            ).fetchone()
            if row:
                self._remove_file_from_db(str_path)
                logger.info("Removed %s from index", file_path.name)

    def search(self, query: str, max_results: Optional[int] = None) -> list[SearchResult]:
        """Search the index for relevant document chunks.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of SearchResult objects sorted by relevance (BM25).
        """
        if not self._conn:
            return []

        limit = max_results or self.config.max_results

        # Prepare FTS5 query: escape special characters and add * suffix for prefix matching
        fts_query = self._prepare_fts_query(query)
        if not fts_query:
            return []

        try:
            # Query with diversity limit: fetch more than needed, then apply per-file cap
            fetch_limit = limit * 5  # Fetch extra for diversity filtering
            rows = self._conn.execute(
                """
                SELECT file_path, file_name, chunk_index, total_chunks, text,
                       rank
                FROM chunks
                WHERE chunks MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, fetch_limit),
            ).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 query failed for %r: %s", query, e)
            return []

        results: list[SearchResult] = []
        seen_files: dict[str, int] = {}

        for row in rows:
            if len(results) >= limit:
                break

            file_path = row[0]

            # Diversity: max 3 chunks from the same file
            file_count = seen_files.get(file_path, 0)
            if file_count >= 3:
                continue
            seen_files[file_path] = file_count + 1

            # BM25 rank is negative (lower = better), convert to positive score
            score = -float(row[5])

            chunk_text = row[4]
            snippet = self._create_snippet(chunk_text, query)

            # Look up file metadata for the chunk
            meta_row = self._conn.execute(
                "SELECT file_hash, file_modified, file_size FROM file_meta WHERE file_path = ?",
                (file_path,),
            ).fetchone()

            chunk = DocumentChunk(
                chunk_id=f"{file_path}:{row[2]}",
                file_path=file_path,
                file_name=row[1],
                chunk_index=int(row[2]),
                total_chunks=int(row[3]),
                text=chunk_text,
                file_hash=meta_row[0] if meta_row else "",
                file_modified=float(meta_row[1]) if meta_row else 0.0,
                file_size=int(meta_row[2]) if meta_row else 0,
            )

            results.append(SearchResult(chunk=chunk, score=score, snippet=snippet))

        return results

    def get_document_text(self, file_path: str) -> str | None:
        """Get the full text of a document by reassembling its chunks.

        Args:
            file_path: File path (absolute or relative to workspace).

        Returns:
            Full document text, or None if not found.
        """
        if not self._conn:
            return None

        workspace = self.config.resolved_path
        candidates = [file_path, str(workspace / file_path)]

        for candidate in candidates:
            rows = self._conn.execute(
                "SELECT text FROM chunks WHERE file_path = ? ORDER BY CAST(chunk_index AS INTEGER)",
                (candidate,),
            ).fetchall()
            if rows:
                return "\n\n".join(row[0] for row in rows)

        return None

    # --- Private methods ---

    def _store_file(self, file_path: Path, chunks: list[DocumentChunk],
                    file_hash: str, stat: os.stat_result) -> None:
        """Store a file's chunks and metadata in the database atomically."""
        if not self._conn:
            return

        str_path = str(file_path)

        # Remove old data for this file
        self._remove_file_from_db(str_path)

        # Insert chunks
        for chunk in chunks:
            self._conn.execute(
                "INSERT INTO chunks(file_path, file_name, chunk_index, total_chunks, text) VALUES(?, ?, ?, ?, ?)",
                (str_path, file_path.name, chunk.chunk_index, chunk.total_chunks, chunk.text),
            )

        # Insert file metadata
        self._conn.execute(
            """INSERT OR REPLACE INTO file_meta(file_path, file_name, file_hash, file_modified, file_size, total_chunks, indexed_at)
               VALUES(?, ?, ?, ?, ?, ?, ?)""",
            (str_path, file_path.name, file_hash, stat.st_mtime, stat.st_size, len(chunks), time.time()),
        )
        self._conn.commit()

    def _remove_file_from_db(self, file_path: str) -> None:
        """Remove all data for a file from the database."""
        if not self._conn:
            return
        # FTS5 requires DELETE with rowid; use content match instead
        self._conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        self._conn.execute("DELETE FROM file_meta WHERE file_path = ?", (file_path,))
        self._conn.commit()

    def _migrate_legacy_index(self) -> None:
        """Handle migration from old JSON index format."""
        workspace = self.config.resolved_path
        legacy_path = workspace / LEGACY_INDEX_FILENAME

        if not legacy_path.exists():
            return

        # Don't try to parse potentially-corrupted JSON — just build fresh
        logger.info("Found legacy JSON index at %s — will build fresh SQLite index", legacy_path)
        try:
            backup_path = legacy_path.with_suffix(".json.old")
            legacy_path.rename(backup_path)
            logger.info("Renamed legacy index to %s", backup_path.name)
        except Exception as e:
            logger.warning("Could not rename legacy index: %s", e)

    def _prepare_fts_query(self, query: str) -> str:
        """Prepare a user query for FTS5 MATCH syntax.

        Handles special characters and converts to a safe FTS5 query.
        """
        # Remove FTS5 special characters that could cause syntax errors
        cleaned = re.sub(r'[^\w\s]', ' ', query)
        terms = cleaned.split()
        if not terms:
            return ""

        # Join terms with implicit AND (FTS5 default)
        # Use quotes around individual terms that might be FTS5 keywords
        safe_terms = []
        for term in terms:
            term = term.strip()
            if term:
                safe_terms.append(f'"{term}"')

        return " ".join(safe_terms)

    def _discover_files(self, workspace: Path) -> list[Path]:
        """Recursively discover all supported files in the workspace."""
        files: list[Path] = []
        excluded_dirs = set(self.config.excluded_dirs)

        for root, dirs, filenames in os.walk(workspace):
            # Filter out excluded directories
            dirs[:] = [
                d for d in dirs
                if d not in excluded_dirs and not d.startswith(".")
            ]

            for filename in filenames:
                file_path = Path(root) / filename
                if self._is_supported_file(file_path):
                    files.append(file_path)

        return sorted(files)

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if a file is supported for indexing."""
        name = file_path.name
        if name.startswith(".") or name.startswith("~$"):
            return False
        return file_path.suffix.lower() in self.config.file_extensions

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute a fast hash of a file for change detection."""
        stat = file_path.stat()
        # Use size + mtime for speed; fall back to content hash for small files
        if stat.st_size < 1_000_000:  # < 1MB: use content hash
            content = file_path.read_bytes()
            return hashlib.md5(content).hexdigest()
        else:
            return f"{stat.st_size}:{stat.st_mtime_ns}"

    def _chunk_text(self, text: str, file_path: Path) -> list[DocumentChunk]:
        """Split text into overlapping chunks.

        Uses paragraph-aware splitting to avoid breaking in the middle of
        sentences when possible.
        """
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        # Normalize whitespace
        text = re.sub(r"\r\n", "\n", text)

        # Split on paragraph boundaries first
        paragraphs = re.split(r"\n{2,}", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for para in paragraphs:
            para_len = len(para)

            # If a single paragraph exceeds chunk size, split it by sentences
            if para_len > chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split long paragraph by sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sent_chunk: list[str] = []
                sent_length = 0

                for sent in sentences:
                    if sent_length + len(sent) > chunk_size and sent_chunk:
                        chunks.append(" ".join(sent_chunk))
                        # Keep overlap
                        overlap_text = " ".join(sent_chunk)
                        if len(overlap_text) > overlap:
                            overlap_text = overlap_text[-overlap:]
                        sent_chunk = [overlap_text, sent]
                        sent_length = len(overlap_text) + len(sent)
                    else:
                        sent_chunk.append(sent)
                        sent_length += len(sent)

                if sent_chunk:
                    chunks.append(" ".join(sent_chunk))
                continue

            if current_length + para_len > chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Carry overlap from the end of the current chunk
                overlap_paras: list[str] = []
                overlap_len = 0
                for p in reversed(current_chunk):
                    if overlap_len + len(p) <= overlap:
                        overlap_paras.insert(0, p)
                        overlap_len += len(p)
                    else:
                        break
                current_chunk = overlap_paras + [para]
                current_length = overlap_len + para_len
            else:
                current_chunk.append(para)
                current_length += para_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        # If no chunks were created, use the whole text
        if not chunks and text.strip():
            chunks = [text.strip()]

        # Apply per-file chunk limit
        if len(chunks) > self.config.max_chunks_per_file:
            logger.warning(
                "Truncating %s from %d to %d chunks",
                file_path.name, len(chunks), self.config.max_chunks_per_file,
            )
            chunks = chunks[: self.config.max_chunks_per_file]

        stat = file_path.stat()
        file_hash = self._compute_file_hash(file_path)

        return [
            DocumentChunk(
                chunk_id=f"{file_hash}:{i}",
                file_path=str(file_path),
                file_name=file_path.name,
                chunk_index=i,
                total_chunks=len(chunks),
                text=chunk_text,
                file_hash=file_hash,
                file_modified=stat.st_mtime,
                file_size=stat.st_size,
            )
            for i, chunk_text in enumerate(chunks)
        ]

    def _create_snippet(self, text: str, query: str, max_length: int = 300) -> str:
        """Create a relevance-highlighted snippet from chunk text."""
        query_terms = set(query.lower().split())
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Score sentences by number of query terms they contain
        scored = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for term in query_terms if term in sent_lower)
            scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Build snippet from top scoring sentences
        snippet_parts: list[str] = []
        length = 0
        for _, sent in scored:
            if length + len(sent) > max_length:
                break
            snippet_parts.append(sent)
            length += len(sent)

        return " ".join(snippet_parts) if snippet_parts else text[:max_length] + "..."
