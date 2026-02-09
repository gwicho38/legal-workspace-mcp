"""Document indexing and search engine using TF-IDF.

Handles chunking documents, building a searchable index, and persisting
the index to disk so it survives server restarts.
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import WorkspaceConfig
from .extractors import extract_text

logger = logging.getLogger(__name__)


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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentChunk":
        return cls(**data)


@dataclass
class SearchResult:
    """A search result with relevance score."""

    chunk: DocumentChunk
    score: float
    snippet: str


class DocumentIndex:
    """TF-IDF based document index with file-watching support.

    The index automatically handles:
    - Chunking documents into searchable pieces
    - Building a TF-IDF matrix for fast search
    - Persisting the index to disk
    - Incremental updates when files change
    """

    def __init__(self, config: WorkspaceConfig):
        self.config = config
        self._chunks: list[DocumentChunk] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._file_hashes: dict[str, str] = {}  # path -> hash
        self._last_rebuild: float = 0
        self._dirty: bool = False

    @property
    def document_count(self) -> int:
        """Number of unique documents in the index."""
        return len(set(c.file_path for c in self._chunks))

    @property
    def chunk_count(self) -> int:
        """Total number of chunks in the index."""
        return len(self._chunks)

    @property
    def indexed_files(self) -> list[str]:
        """List of all indexed file paths (relative to workspace)."""
        workspace = self.config.resolved_path
        unique = sorted(set(c.file_path for c in self._chunks))
        result = []
        for p in unique:
            try:
                result.append(str(Path(p).relative_to(workspace)))
            except ValueError:
                result.append(p)
        return result

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

        # Discover all supported files
        files = self._discover_files(workspace)
        logger.info("Discovered %d files in workspace", len(files))

        # Extract and chunk all documents
        new_chunks: list[DocumentChunk] = []
        new_hashes: dict[str, str] = {}
        errors: list[str] = []

        for file_path in files:
            try:
                file_hash = self._compute_file_hash(file_path)
                new_hashes[str(file_path)] = file_hash

                # Skip if unchanged
                if (
                    str(file_path) in self._file_hashes
                    and self._file_hashes[str(file_path)] == file_hash
                ):
                    # Reuse existing chunks
                    existing = [c for c in self._chunks if c.file_path == str(file_path)]
                    new_chunks.extend(existing)
                    continue

                text = extract_text(file_path)
                if not text or not text.strip():
                    logger.warning("No text extracted from %s", file_path)
                    continue

                chunks = self._chunk_text(text, file_path)
                new_chunks.extend(chunks)
                logger.info("Indexed %s (%d chunks)", file_path.name, len(chunks))

            except Exception as e:
                logger.error("Error indexing %s: %s", file_path, e)
                errors.append(f"{file_path.name}: {e}")

        self._chunks = new_chunks
        self._file_hashes = new_hashes
        self._rebuild_tfidf()
        self._last_rebuild = time.time()
        self._dirty = True

        elapsed = time.time() - start
        summary = {
            "documents": self.document_count,
            "chunks": self.chunk_count,
            "elapsed_seconds": round(elapsed, 2),
            "errors": errors,
        }

        # Persist to disk
        self._save_index()

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
        file_hash = self._compute_file_hash(file_path)

        # Skip if unchanged
        if str_path in self._file_hashes and self._file_hashes[str_path] == file_hash:
            return

        # Remove old chunks for this file
        self._chunks = [c for c in self._chunks if c.file_path != str_path]

        # Extract and chunk
        text = extract_text(file_path)
        if text and text.strip():
            chunks = self._chunk_text(text, file_path)
            self._chunks.extend(chunks)
            self._file_hashes[str_path] = file_hash
            logger.info("Updated index for %s (%d chunks)", file_path.name, len(chunks))
        else:
            # Remove from hashes if no text
            self._file_hashes.pop(str_path, None)

        self._rebuild_tfidf()
        self._dirty = True
        self._save_index()

    def remove_file(self, file_path: Path) -> None:
        """Remove a file from the index.

        Args:
            file_path: Absolute path to the removed file.
        """
        str_path = str(file_path)
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.file_path != str_path]
        self._file_hashes.pop(str_path, None)

        if len(self._chunks) != before:
            logger.info("Removed %s from index", file_path.name)
            self._rebuild_tfidf()
            self._dirty = True
            self._save_index()

    def search(self, query: str, max_results: Optional[int] = None) -> list[SearchResult]:
        """Search the index for relevant document chunks.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        if not self._chunks or self._tfidf_matrix is None or self._vectorizer is None:
            return []

        limit = max_results or self.config.max_results

        # Transform query using the fitted vectorizer
        query_vec = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        # Get top results above a minimum threshold
        min_score = 0.01
        top_indices = np.argsort(similarities)[::-1]

        results: list[SearchResult] = []
        seen_files: dict[str, int] = {}  # Track results per file for diversity

        for idx in top_indices:
            if len(results) >= limit:
                break

            score = float(similarities[idx])
            if score < min_score:
                break

            chunk = self._chunks[idx]

            # Light diversity: max 3 chunks from the same file in results
            file_count = seen_files.get(chunk.file_path, 0)
            if file_count >= 3:
                continue
            seen_files[chunk.file_path] = file_count + 1

            snippet = self._create_snippet(chunk.text, query)
            results.append(SearchResult(chunk=chunk, score=score, snippet=snippet))

        return results

    def get_document_text(self, file_path: str) -> str | None:
        """Get the full text of a document by reassembling its chunks.

        Args:
            file_path: File path (absolute or relative to workspace).

        Returns:
            Full document text, or None if not found.
        """
        # Try to match as absolute path first, then relative
        workspace = self.config.resolved_path
        candidates = [file_path, str(workspace / file_path)]

        matching_chunks = []
        for candidate in candidates:
            matching_chunks = sorted(
                [c for c in self._chunks if c.file_path == candidate],
                key=lambda c: c.chunk_index,
            )
            if matching_chunks:
                break

        if not matching_chunks:
            return None

        # Reassemble from chunks (handle overlap by using the first occurrence)
        return "\n\n".join(c.text for c in matching_chunks)

    def load_persisted_index(self) -> bool:
        """Load a previously persisted index from disk.

        Returns:
            True if an index was successfully loaded.
        """
        index_path = self.config.index_path
        if not index_path.exists():
            return False

        try:
            with open(index_path, "r") as f:
                data = json.load(f)

            self._chunks = [DocumentChunk.from_dict(c) for c in data.get("chunks", [])]
            self._file_hashes = data.get("file_hashes", {})

            if self._chunks:
                self._rebuild_tfidf()
                logger.info("Loaded persisted index: %d chunks", len(self._chunks))
                return True
        except Exception as e:
            logger.error("Failed to load persisted index: %s", e)

        return False

    # --- Private methods ---

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

    def _rebuild_tfidf(self) -> None:
        """Rebuild the TF-IDF matrix from current chunks."""
        if not self._chunks:
            self._vectorizer = None
            self._tfidf_matrix = None
            return

        texts = [c.text for c in self._chunks]
        self._vectorizer = TfidfVectorizer(
            max_features=50000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)

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

    def _save_index(self) -> None:
        """Persist the index to disk using atomic write.

        Writes to a temporary file first, then atomically replaces the
        target. This prevents corruption if the process is killed mid-write.
        """
        index_path = self.config.index_path
        tmp_path = index_path.with_suffix(".json.tmp")
        try:
            data = {
                "version": 1,
                "built_at": time.time(),
                "chunks": [c.to_dict() for c in self._chunks],
                "file_hashes": self._file_hashes,
            }
            with open(tmp_path, "w") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, index_path)
            self._dirty = False
        except Exception as e:
            logger.error("Failed to save index: %s", e)
            # Clean up partial temp file
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
