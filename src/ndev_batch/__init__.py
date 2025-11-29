"""ndev-batch: Zero-dependency batch processing utilities.

This package provides a lightweight, dependency-free foundation for batch
processing operations. It's designed to work with napari plugins but has
no napari or Qt dependencies itself.

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

Examples
--------
Basic batch processing:

>>> from ndev_batch import batch, BatchContext
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

>>> from ndev_batch import batch_logger
>>> with batch_logger(log_file="output/log.txt", header={"Files": 100}) as log:
...     for result, ctx in process(files, with_context=True):
...         log(ctx, f"Processed: {result}")
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = 'unknown'

from ndev_batch._context import BatchContext
from ndev_batch._decorator import batch
from ndev_batch._discovery import discover_files, is_batch_input
from ndev_batch._logging import BatchLogger, batch_logger

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
]
