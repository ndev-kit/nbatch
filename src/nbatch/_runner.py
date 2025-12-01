"""Batch runner for orchestrating batch operations.

This module provides BatchRunner, a class that orchestrates batch processing
with threading, progress callbacks, logging, and cancellation support.
Works with or without napari.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nbatch._context import BatchContext
from nbatch._discovery import discover_files, is_batch_input
from nbatch._logging import BatchLogger, batch_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence

# Module-level logger
_logger = logging.getLogger('nbatch.runner')

# Check for napari availability
try:
    from napari.qt.threading import create_worker

    HAS_NAPARI = True
except ImportError:
    HAS_NAPARI = False
    create_worker = None


class BatchRunner:
    """Orchestrates batch operations with threading, progress, and cancellation.

    BatchRunner provides a clean interface for running batch operations in
    napari widgets or standalone scripts. It handles threading, progress
    callbacks, logging, and graceful cancellation.

    Parameters
    ----------
    on_start : Callable[[int], None] | None, optional
        Called when batch starts, receives total item count. Use to initialize
        progress bars (e.g., ``on_start=lambda total: progress_bar.setMaximum(total)``).
    on_item_complete : Callable[[Any, BatchContext], None] | None, optional
        Called after each item completes successfully. Receives the result
        and BatchContext. Use for progress bars and adding results to viewer.
    on_complete : Callable[[], None] | None, optional
        Called when the entire batch finishes (not called if cancelled).
    on_error : Callable[[BatchContext, Exception], None] | None, optional
        Called when an item fails. Receives context and the exception.
        Note: This is for UI notification; error handling policy is set
        on the @batch decorator.
    on_cancel : Callable[[], None] | None, optional
        Called when the batch is cancelled by the user.

    Attributes
    ----------
    is_running : bool
        True if a batch is currently being processed.
    was_cancelled : bool
        True if the last batch was cancelled before completion.
    error_count : int
        Number of errors encountered in the current/last batch.

    Examples
    --------
    Basic usage in a napari widget:

    >>> runner = BatchRunner(
    ...     on_start=lambda total: progress_bar.setMaximum(total),
    ...     on_item_complete=lambda r, ctx: progress_bar.setValue(ctx.index + 1),
    ...     on_complete=lambda: print("Done!"),
    ... )
    >>> runner.run(process_image, files)

    With cancellation:

    >>> cancel_button.clicked.connect(runner.cancel)
    >>> runner.run(process_image, files)

    Synchronous usage (scripts, testing):

    >>> runner = BatchRunner(
    ...     on_item_complete=lambda r, ctx: print(f"{ctx.progress:.0%}"),
    ... )
    >>> runner.run(process_image, files, threaded=False)

    With logging:

    >>> runner.run(
    ...     process_image,
    ...     files,
    ...     log_file="output/batch.log",
    ...     log_header={"Input": str(input_dir), "Files": len(files)},
    ... )

    Passing additional arguments to the function (no partial needed!):

    >>> runner.run(
    ...     process_image,
    ...     files,
    ...     output_dir=output_path,  # passed to process_image
    ...     sigma=2.0,               # passed to process_image
    ... )
    """

    def __init__(
        self,
        on_start: Callable[[int], None] | None = None,
        on_item_complete: Callable[[Any, BatchContext], None] | None = None,
        on_complete: Callable[[], None] | None = None,
        on_error: Callable[[BatchContext, Exception], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ):
        self._on_start = on_start
        self._on_item_complete = on_item_complete
        self._on_complete = on_complete
        self._on_error = on_error
        self._on_cancel = on_cancel

        # State tracking
        self._worker = None  # napari worker or thread
        self._cancel_requested = False
        self._is_running = False
        self._was_cancelled = False
        self._error_count = 0
        self._lock = threading.Lock()

        # For logging within run
        self._batch_log: BatchLogger | None = None

    @property
    def is_running(self) -> bool:
        """Check if a batch is currently running."""
        with self._lock:
            return self._is_running

    @property
    def was_cancelled(self) -> bool:
        """Check if the last batch was cancelled."""
        with self._lock:
            return self._was_cancelled

    @property
    def error_count(self) -> int:
        """Number of errors encountered in the current/last batch."""
        with self._lock:
            return self._error_count

    def cancel(self) -> None:
        """Request cancellation of the running batch.

        The current item will complete, then processing stops.
        The on_cancel callback will be called.
        """
        with self._lock:
            if not self._is_running:
                return
            self._cancel_requested = True
            self._was_cancelled = True  # Set immediately for threaded cases
            # Store local reference to avoid race condition with _handle_finished
            worker = self._worker if HAS_NAPARI else None

        # If using napari worker, request quit
        if worker is not None:
            with contextlib.suppress(RuntimeError):
                worker.quit()

    def run(
        self,
        func: Callable[..., Any],
        items: Any,
        *args: Any,
        threaded: bool = True,
        log_file: str | Path | None = None,
        log_header: Mapping[str, object] | None = None,
        patterns: str | Sequence[str] = '*',
        recursive: bool = False,
        **kwargs: Any,
    ) -> None:
        """Run a batch operation.

        Parameters
        ----------
        func : Callable
            The function to run on each item. Can be @batch decorated or plain.
            If plain, items are processed directly. If @batch decorated, the
            decorator's settings (on_error, etc.) are respected.
        items : Any
            Items to process. Can be:
            - A directory Path (files discovered using patterns)
            - A list/tuple of items
            - Any iterable
        *args : Any
            Additional positional arguments passed to func.
        threaded : bool, optional
            If True (default), run in background thread. Requires napari for
            full functionality, falls back to ThreadPoolExecutor without napari.
            If False, run synchronously (blocks until complete).
        log_file : str | Path | None, optional
            If provided, write logs to this file.
        log_header : Mapping[str, object] | None, optional
            Header metadata to write at start of log.
        patterns : str | Sequence[str], optional
            Glob pattern(s) for file discovery when items is a directory.
            Default is '*'.
        recursive : bool, optional
            If True, search directories recursively. Default is False.
        **kwargs : Any
            Additional keyword arguments passed to func.

        Raises
        ------
        RuntimeError
            If a batch is already running.
        """
        with self._lock:
            if self._is_running:
                raise RuntimeError(
                    'A batch is already running. Call cancel() first.'
                )
            self._is_running = True
            self._cancel_requested = False
            self._was_cancelled = False
            self._error_count = 0

        # Normalize items to a list
        items_list = self._normalize_items(items, patterns, recursive)

        # Call on_start callback with total count
        if self._on_start is not None:
            self._on_start(len(items_list))

        if threaded and HAS_NAPARI:
            self._run_napari_threaded(
                func, items_list, args, kwargs, log_file, log_header
            )
        elif threaded:
            self._run_thread_fallback(
                func, items_list, args, kwargs, log_file, log_header
            )
        else:
            self._run_sync(
                func, items_list, args, kwargs, log_file, log_header
            )

    def _normalize_items(
        self,
        items: Any,
        patterns: str | Sequence[str],
        recursive: bool,
    ) -> list[Any]:
        """Normalize items input to a list."""
        if isinstance(items, Path):
            if items.is_dir():
                return discover_files(
                    items, patterns=patterns, recursive=recursive
                )
            else:
                # Single file path
                return [items]
        elif isinstance(items, list | tuple):
            # Check if all items are Path objects (use file discovery)
            if items and all(isinstance(item, Path) for item in items):
                return discover_files(items)
            # Otherwise treat as generic items (strings, numbers, etc.)
            return list(items)
        elif is_batch_input(items):
            return list(items)  # type: ignore[arg-type]
        else:
            # Single item - wrap in list
            return [items]

    def _run_napari_threaded(
        self,
        func: Callable[..., Any],
        items: list[Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        log_file: str | Path | None,
        log_header: Mapping[str, object] | None,
    ) -> None:
        """Run batch using napari's create_worker for Qt-safe threading."""

        def _worker_func():
            """Generator function for napari worker."""
            yield from self._process_items(
                func, items, args, kwargs, log_file, log_header
            )

        self._worker = create_worker(_worker_func)
        self._worker.yielded.connect(self._handle_yielded)
        self._worker.finished.connect(self._handle_finished)
        self._worker.start()

    def _run_thread_fallback(
        self,
        func: Callable[..., Any],
        items: list[Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        log_file: str | Path | None,
        log_header: Mapping[str, object] | None,
    ) -> None:
        """Run batch using standard threading (fallback when napari unavailable)."""
        import concurrent.futures

        # Store executor to allow shutdown after completion
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def _thread_target():
            try:
                for result, ctx, error in self._process_items(
                    func, items, args, kwargs, log_file, log_header
                ):
                    self._handle_yielded((result, ctx, error))
            finally:
                self._handle_finished()
                # Shutdown executor to release resources
                self._executor.shutdown(wait=False)

        self._worker = self._executor.submit(_thread_target)

    def _run_sync(
        self,
        func: Callable[..., Any],
        items: list[Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        log_file: str | Path | None,
        log_header: Mapping[str, object] | None,
    ) -> None:
        """Run batch synchronously (blocking)."""
        try:
            for result, ctx, error in self._process_items(
                func, items, args, kwargs, log_file, log_header
            ):
                self._handle_yielded((result, ctx, error))
        finally:
            self._handle_finished()

    def _process_items(
        self,
        func: Callable[..., Any],
        items: list[Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        log_file: str | Path | None,
        log_header: Mapping[str, object] | None,
    ) -> Generator[tuple[Any, BatchContext, Exception | None], None, None]:
        """Process items and yield (result, context, error) tuples.

        This is the core processing loop. It handles:
        - Creating BatchContext for each item
        - Calling the function
        - Catching errors
        - Checking for cancellation
        - Logging (if configured)
        """
        total = len(items)

        # Set up logging context if requested
        log_context = (
            batch_logger(
                log_file=log_file,
                header=log_header,
                console=log_file is None,  # Console only if no file
            )
            if log_file is not None or log_header is not None
            else None
        )

        try:
            if log_context is not None:
                self._batch_log = log_context.__enter__()

            for index, item in enumerate(items):
                # Check for cancellation before processing
                with self._lock:
                    if self._cancel_requested:
                        self._was_cancelled = True
                        break

                ctx = BatchContext(index=index, total=total, item=item)

                try:
                    result = func(item, *args, **kwargs)

                    if self._batch_log is not None:
                        self._batch_log(ctx, f'Completed: {_item_name(item)}')

                    yield (result, ctx, None)

                except Exception as e:
                    _logger.exception('%s - Error processing item: %s', ctx, e)

                    if self._batch_log is not None:
                        self._batch_log.error(
                            ctx, f'Failed: {_item_name(item)} - {e}'
                        )

                    yield (None, ctx, e)

        finally:
            if log_context is not None:
                log_context.__exit__(None, None, None)
            self._batch_log = None

    def _handle_yielded(
        self, value: tuple[Any, BatchContext, Exception | None]
    ) -> None:
        """Handle a yielded result from the worker."""
        result, ctx, error = value

        if error is not None:
            with self._lock:
                self._error_count += 1
            if self._on_error is not None:
                self._on_error(ctx, error)
        else:
            if self._on_item_complete is not None:
                self._on_item_complete(result, ctx)

    def _handle_finished(self) -> None:
        """Handle batch completion."""
        with self._lock:
            was_cancelled = self._was_cancelled
            self._is_running = False
            self._worker = None

        if was_cancelled:
            if self._on_cancel is not None:
                self._on_cancel()
        else:
            if self._on_complete is not None:
                self._on_complete()


def _item_name(item: Any) -> str:
    """Get a display name for an item."""
    if isinstance(item, Path):
        return item.name
    elif hasattr(item, 'name'):
        name = item.name
        # Avoid returning a method or function
        if callable(name):
            return str(item)
        return str(name)
    else:
        return str(item)
