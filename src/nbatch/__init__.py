"""nbatch: Lightweight batch processing utilities.

This package provides a lightweight foundation for batch
processing operations. It's designed to work with napari plugins but has
no napari or Qt dependencies in the core modules.

Core Components
---------------
batch : decorator
    Transform single-item functions into batch-capable functions.
BatchContext : dataclass
    Holds metadata about current item being processed.
discover_files : function
    Discover files from paths, directories, or iterables.
batch_logger : context manager
    Scoped logging for batch operations with optional file output.
BatchRunner : class
    Orchestrates batch operations with threading, progress, and cancellation.
    Uses napari's threading when available, falls back to standard threads.

Examples
--------
Basic batch processing:

>>> from nbatch import batch, BatchContext
>>> @batch
... def process(path):
...     return path.stem.upper()
>>> process(Path("image.tif"))  # Single item
'IMAGE'
>>> list(process([Path("a.tif"), Path("b.tif")]))  # Batch
['A', 'B']

With progress tracking:

>>> @batch(with_context=True)
... def process(path):
...     return path.stem
>>> for result, ctx in process(files):
...     print(f"{ctx.progress:.0%} complete")

With logging:

>>> from nbatch import batch_logger
>>> with batch_logger(log_file="output/log.txt", header={"Files": 100}) as log:
...     for result, ctx in process(files):
...         log(ctx, f"Processed: {result}")

With BatchRunner (for widgets):

>>> from nbatch import BatchRunner
>>> runner = BatchRunner(
...     on_item_complete=lambda result, ctx: progress_bar.setValue(ctx.index + 1),
...     on_complete=lambda: print("Done!"),
... )
>>> runner.run(process, files)  # Threaded, non-blocking
>>> runner.cancel()  # Cancel if needed
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = 'unknown'

from nbatch._context import BatchContext
from nbatch._decorator import batch
from nbatch._discovery import discover_files, is_batch_input
from nbatch._logging import BatchLogger, batch_logger
from nbatch._runner import BatchRunner

__all__ = [
    # Core decorator
    'batch',
    # Context
    'BatchContext',
    # Discovery
    'discover_files',
    'is_batch_input',
    # Logging
    'batch_logger',
    'BatchLogger',
    # Runner
    'BatchRunner',
]
