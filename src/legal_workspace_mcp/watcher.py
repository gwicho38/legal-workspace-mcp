"""File system watcher for automatic index updates.

Uses the watchdog library to monitor the workspace directory for changes
and triggers index updates with debouncing to avoid excessive reprocessing.
"""

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

from .config import WorkspaceConfig, WATCHER_DEBOUNCE_SECONDS

if TYPE_CHECKING:
    from .indexer import DocumentIndex

logger = logging.getLogger(__name__)


class _DebouncedHandler(FileSystemEventHandler):
    """File system event handler with debouncing.

    Collects file change events and processes them in batches
    after a quiet period, to avoid reindexing on every keystroke
    during active editing.
    """

    def __init__(self, index: "DocumentIndex", config: WorkspaceConfig):
        super().__init__()
        self._index = index
        self._config = config
        self._pending_updates: set[str] = set()
        self._pending_removes: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._debounce_seconds = WATCHER_DEBOUNCE_SECONDS

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule_update(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule_update(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule_remove(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule_remove(event.src_path)
            if hasattr(event, "dest_path"):
                self._schedule_update(event.dest_path)

    def _schedule_update(self, path: str) -> None:
        """Schedule a file for re-indexing after the debounce period."""
        file_path = Path(path)
        if not self._is_relevant(file_path):
            return

        with self._lock:
            self._pending_removes.discard(path)
            self._pending_updates.add(path)
            self._reset_timer()

    def _schedule_remove(self, path: str) -> None:
        """Schedule a file for removal from the index."""
        with self._lock:
            self._pending_updates.discard(path)
            self._pending_removes.add(path)
            self._reset_timer()

    def _reset_timer(self) -> None:
        """Reset the debounce timer."""
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._debounce_seconds, self._process_pending)
        self._timer.daemon = True
        self._timer.start()

    def _process_pending(self) -> None:
        """Process all pending file changes."""
        with self._lock:
            updates = list(self._pending_updates)
            removes = list(self._pending_removes)
            self._pending_updates.clear()
            self._pending_removes.clear()

        # Process removals first
        for path in removes:
            try:
                self._index.remove_file(Path(path))
            except Exception as e:
                logger.error("Error removing %s from index: %s", path, e)

        # Then process updates
        for path in updates:
            try:
                file_path = Path(path)
                if file_path.exists():
                    self._index.update_file(file_path)
            except Exception as e:
                logger.error("Error updating %s in index: %s", path, e)

        if updates or removes:
            logger.info(
                "Watcher processed %d updates, %d removals",
                len(updates), len(removes),
            )

    def _is_relevant(self, file_path: Path) -> bool:
        """Check if a file change is relevant for indexing."""
        # Skip hidden files
        if file_path.name.startswith("."):
            return False
        # Skip unsupported extensions
        if file_path.suffix.lower() not in self._config.file_extensions:
            return False
        # Skip excluded directories
        parts = file_path.parts
        for excluded in self._config.excluded_dirs:
            if excluded in parts:
                return False
        return True


class WorkspaceWatcher:
    """Watches a workspace directory for file changes and updates the index.

    Usage:
        watcher = WorkspaceWatcher(index, config)
        watcher.start()
        # ... server runs ...
        watcher.stop()
    """

    def __init__(self, index: "DocumentIndex", config: WorkspaceConfig):
        self._index = index
        self._config = config
        self._observer: Observer | None = None
        self._running = False

    def start(self) -> None:
        """Start watching the workspace directory."""
        if self._running:
            return

        workspace = self._config.resolved_path
        if not workspace.is_dir():
            logger.error("Cannot watch non-existent directory: %s", workspace)
            return

        handler = _DebouncedHandler(self._index, self._config)
        self._observer = Observer()
        self._observer.schedule(handler, str(workspace), recursive=True)
        self._observer.daemon = True
        self._observer.start()
        self._running = True
        logger.info("Started watching workspace: %s", workspace)

    def stop(self) -> None:
        """Stop watching the workspace directory."""
        if self._observer and self._running:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._running = False
            logger.info("Stopped watching workspace")

    @property
    def is_running(self) -> bool:
        return self._running
