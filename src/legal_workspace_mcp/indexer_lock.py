"""Pick one indexer among the servers that share an index (issue #12).

Every Claude session starts its own server on the same index file. If each ran
the startup build and a file watcher, N sessions would do N times the work and
fight for the write lock. The server that holds an exclusive, non-blocking
``flock`` on ``<index_path>.indexer.lock`` is the only writer. The OS drops the
lock when that process exits, even on a crash, so another server can take over.
"""

import fcntl
import logging
import os
from pathlib import Path
from typing import Optional

from .config import INDEXER_LOCK_SUFFIX

logger = logging.getLogger(__name__)


class IndexerLock:
    def __init__(self, index_path: Path):
        self.path = index_path.with_name(index_path.name + INDEXER_LOCK_SUFFIX)
        self._fd: Optional[int] = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def try_acquire(self) -> bool:
        """Take the lock if no other process holds it. Never blocks."""
        if self._fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd
        logger.info("This server is the indexer for %s (pid %d)", self.path, os.getpid())
        return True

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None
