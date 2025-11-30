"""Batch processing decorator.

This module provides the @batch decorator that transforms a single-item
processing function into one that handles both single items and batches.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    ParamSpec,
    TypeVar,
    overload,
)

from nbatch._context import BatchContext
from nbatch._discovery import discover_files, is_batch_input

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence

# Type variables for generic typing
P = ParamSpec('P')
T = TypeVar('T')

# Error handling modes
OnErrorType = Literal['raise', 'continue', 'skip']

# Module-level logger for batch operations
_logger = logging.getLogger('nbatch')


@overload
def batch(
    func: Callable[P, T],
    *,
    on_error: OnErrorType = 'raise',
    with_context: Literal[False] = False,
    patterns: str | Sequence[str] = '*',
    recursive: bool = False,
) -> Callable[..., T | Generator[T, None, None]]: ...


@overload
def batch(
    func: Callable[P, T],
    *,
    on_error: OnErrorType = 'raise',
    with_context: Literal[True],
    patterns: str | Sequence[str] = '*',
    recursive: bool = False,
) -> Callable[..., T | Generator[tuple[T, BatchContext], None, None]]: ...


@overload
def batch(
    func: None = None,
    *,
    on_error: OnErrorType = 'raise',
    with_context: bool = False,
    patterns: str | Sequence[str] = '*',
    recursive: bool = False,
) -> Callable[[Callable[P, T]], Callable[..., Any]]: ...


def batch(
    func: Callable[P, T] | None = None,
    *,
    on_error: OnErrorType = 'raise',
    with_context: bool = False,
    patterns: str | Sequence[str] = '*',
    recursive: bool = False,
) -> Callable[..., Any]:
    """Decorator that enables batch processing for a function.

    Transforms a function that processes a single item into one that can
    handle both single items and batches (lists, directories). When given
    a batch input, returns a generator that yields results for each item.

    Parameters
    ----------
    func : Callable | None
        The function to decorate. If None, returns a decorator.
    on_error : {'raise', 'continue', 'skip'}, optional
        Error handling mode:
        - 'raise': Re-raise exceptions immediately (default)
        - 'continue': Log error and yield None for failed items
        - 'skip': Log error and skip failed items (don't yield)
    with_context : bool, optional
        If True, yield (result, BatchContext) tuples instead of just results.
        Default is False.
    patterns : str | Sequence[str], optional
        Glob pattern(s) for file discovery when input is a directory.
        Default is '*' (all files).
    recursive : bool, optional
        If True, search directories recursively. Default is False.

    Returns
    -------
    Callable
        Decorated function that handles both single and batch inputs.

    Examples
    --------
    Basic usage - single item returns directly:

    >>> @batch
    ... def process(path: Path) -> str:
    ...     return path.stem.upper()
    >>> process(Path("image.tif"))
    'IMAGE'

    Batch input returns generator:

    >>> results = process([Path("a.tif"), Path("b.tif")])
    >>> list(results)
    ['A', 'B']

    With context for progress tracking:

    >>> @batch(with_context=True)
    ... def process(path: Path) -> str:
    ...     return path.stem
    >>> for result, ctx in process(files):
    ...     print(f"{ctx.progress:.0%}: {result}")
    10%: image1
    20%: image2
    ...

    Error handling:

    >>> @batch(on_error='continue')
    ... def risky(path: Path) -> str:
    ...     if 'bad' in path.name:
    ...         raise ValueError("Bad file!")
    ...     return path.stem
    >>> list(risky([Path("good.tif"), Path("bad.tif"), Path("ok.tif")]))
    ['good', None, 'ok']

    With napari thread_worker:

    >>> @thread_worker
    ... def run_batch():
    ...     for result, ctx in process_image(files, with_context=True):
    ...         yield ctx  # Enables progress updates
    ...
    >>> worker = run_batch()
    >>> worker.yielded.connect(update_progress_bar)
    >>> worker.start()
    """

    def decorator(fn: Callable[P, T]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(
            first_arg: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            # Determine if this is batch or single-item processing
            if not is_batch_input(first_arg):
                # Single item - call directly and return result
                return fn(first_arg, *args, **kwargs)

            # Batch processing - discover files and return generator
            if isinstance(first_arg, Path) and first_arg.is_dir():
                items = discover_files(
                    first_arg, patterns=patterns, recursive=recursive
                )
            elif isinstance(first_arg, (list, tuple)):
                # Could be paths or other items
                if first_arg and isinstance(first_arg[0], (str, Path)):
                    items = discover_files(first_arg)
                else:
                    items = list(first_arg)
            else:
                items = list(first_arg)

            return _batch_generator(
                fn,
                items,
                args,
                kwargs,
                on_error=on_error,
                with_context=with_context,
            )

        return wrapper

    # Handle both @batch and @batch(...) syntax
    if func is not None:
        return decorator(func)
    return decorator


def _batch_generator(
    func: Callable[..., T],
    items: Iterable[Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    on_error: OnErrorType,
    with_context: bool,
) -> Generator[T | tuple[T, BatchContext] | None, None, None]:
    """Internal generator that processes items in batch.

    Parameters
    ----------
    func : Callable
        The function to call for each item.
    items : Iterable
        Items to process.
    args : tuple
        Additional positional arguments for func.
    kwargs : dict
        Keyword arguments for func.
    on_error : OnErrorType
        Error handling mode.
    with_context : bool
        Whether to yield (result, context) tuples.

    Yields
    ------
    T | tuple[T, BatchContext] | None
        Results, optionally paired with context.
    """
    items_list = list(items)
    total = len(items_list)

    for index, item in enumerate(items_list):
        ctx = BatchContext(index=index, total=total, item=item)

        try:
            result = func(item, *args, **kwargs)

            if with_context:
                yield (result, ctx)
            else:
                yield result

        except Exception as e:
            if on_error == 'raise':
                raise

            # Log the error
            _logger.exception('%s - Error processing item: %s', ctx, e)

            if on_error == 'continue':
                if with_context:
                    yield (None, ctx)
                else:
                    yield None
            # on_error == 'skip': don't yield anything
