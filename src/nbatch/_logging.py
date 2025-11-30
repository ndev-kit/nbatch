"""Logging utilities for batch processing.

This module provides utilities for setting up and managing logging during
batch operations, including a context manager for scoped log file handling.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from nbatch._context import BatchContext

# Default format for batch processing logs
DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


class BatchLogger:
    """Logger wrapper with context-aware logging methods.

    Provides convenient methods for logging within batch processing,
    automatically formatting messages with batch context information.

    Parameters
    ----------
    logger : logging.Logger
        The underlying logger instance.

    Examples
    --------
    >>> blog = BatchLogger(logging.getLogger('batch'))
    >>> ctx = BatchContext(index=0, total=10, item='image.tif')
    >>> blog(ctx, 'Processing complete')  # INFO level
    >>> blog.error(ctx, 'Failed to process')  # ERROR level
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def __call__(
        self,
        ctx: BatchContext,
        message: str,
        level: int = logging.INFO,
    ) -> None:
        """Log a message with batch context.

        Parameters
        ----------
        ctx : BatchContext
            Current batch context.
        message : str
            Message to log.
        level : int, optional
            Logging level. Default is INFO.
        """
        formatted = f'{ctx} - {message}'
        self._logger.log(level, formatted)

    def info(self, ctx: BatchContext, message: str) -> None:
        """Log an INFO level message with batch context."""
        self(ctx, message, logging.INFO)

    def warning(self, ctx: BatchContext, message: str) -> None:
        """Log a WARNING level message with batch context."""
        self(ctx, message, logging.WARNING)

    def error(self, ctx: BatchContext, message: str) -> None:
        """Log an ERROR level message with batch context."""
        self(ctx, message, logging.ERROR)

    def debug(self, ctx: BatchContext, message: str) -> None:
        """Log a DEBUG level message with batch context."""
        self(ctx, message, logging.DEBUG)

    def exception(self, ctx: BatchContext, message: str) -> None:
        """Log an ERROR level message with exception info."""
        formatted = f'{ctx} - {message}'
        self._logger.exception(formatted)


@contextmanager
def batch_logger(
    log_file: str | Path | None = None,
    header: Mapping[str, object] | None = None,
    level: int = logging.INFO,
    console: bool = True,
    file_mode: Literal['w', 'a'] = 'a',
) -> Generator[BatchLogger, None, None]:
    """Context manager for batch processing logging.

    Sets up logging for batch operations with optional file and console output.
    By default, logs to console only. If a log file is provided, logs are
    appended to preserve history across runs.

    Parameters
    ----------
    log_file : str | Path | None, optional
        Path to the log file. If provided, logs will be written to this file.
        Parent directories are created if needed. Default is None (no file).
    header : Mapping[str, object] | None, optional
        Dictionary of metadata to write at the start of the batch.
        Useful for recording batch parameters. Default is None.
    level : int, optional
        Logging level. Default is INFO.
    console : bool, optional
        Whether to also log to console (stderr). Default is True.
    file_mode : {'w', 'a'}, optional
        File mode - 'w' to overwrite, 'a' to append. Default is 'a' (append).

    Yields
    ------
    BatchLogger
        A BatchLogger instance for context-aware logging.

    Examples
    --------
    Console only (default):

    >>> with batch_logger() as log:
    ...     for result, ctx in batch_process(files):
    ...         log(ctx, f'Processed: {result}')

    With file logging:

    >>> with batch_logger('output/process.log') as log:
    ...     for result, ctx in batch_process(files):
    ...         log(ctx, f'Processed: {result}')

    With header metadata:

    >>> with batch_logger(
    ...     'output/predict.log',
    ...     header={'Model': 'classifier.clf', 'Files': 100}
    ... ) as log:
    ...     for result, ctx in predict(files):
    ...         log(ctx, f'Result: {result}')

    File only (no console output):

    >>> with batch_logger('output/quiet.log', console=False) as log:
    ...     for result, ctx in batch_process(files):
    ...         log(ctx, f'Processed: {result}')
    """
    # Generate unique logger name to avoid conflicts
    logger_id = id(log_file) if log_file else id(header)
    logger_name = f'nbatch.batch_{logger_id}'

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Clear any existing handlers (in case of reuse)
    logger.handlers.clear()

    # Prevent propagation to root logger (avoid duplicate messages)
    logger.propagate = False

    # Create formatter
    formatter = logging.Formatter(
        DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT
    )

    # Track handlers for cleanup
    handlers: list[logging.Handler] = []

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        handlers.append(console_handler)

    # Add file handler if log_file provided
    file_handler = None
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode=file_mode)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        handlers.append(file_handler)

    # Write header if provided
    if header:
        timestamp = datetime.now().strftime(DEFAULT_DATE_FORMAT)
        logger.info('=' * 60)
        logger.info('Batch processing started at %s', timestamp)
        logger.info('-' * 60)
        for key, value in header.items():
            logger.info('%s: %s', key, value)
        logger.info('=' * 60)

    try:
        yield BatchLogger(logger)
    finally:
        # Write footer if we wrote a header
        if header:
            timestamp = datetime.now().strftime(DEFAULT_DATE_FORMAT)
            logger.info('=' * 60)
            logger.info('Batch processing completed at %s', timestamp)
            logger.info('=' * 60)

        # Clean up all handlers
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
