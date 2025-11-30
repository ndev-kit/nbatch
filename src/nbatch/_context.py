"""Batch processing context for tracking iteration progress.

This module provides the BatchContext dataclass that holds metadata about
the current item being processed in a batch operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BatchContext:
    """Context object providing metadata about current batch item.

    This dataclass is passed to batch-decorated functions (when with_context=True)
    and yielded alongside results to enable progress tracking.

    Parameters
    ----------
    index : int
        Zero-based index of current item in the batch.
    total : int
        Total number of items in the batch.
    item : Any
        The current item being processed.

    Examples
    --------
    >>> ctx = BatchContext(index=2, total=10, item="file.tif")
    >>> ctx.progress
    0.3
    >>> ctx.is_first
    False
    >>> ctx.is_last
    False
    """

    index: int
    total: int
    item: Any

    @property
    def progress(self) -> float:
        """Return progress as fraction (0.0 to 1.0).

        Returns
        -------
        float
            Progress through batch, calculated as (index + 1) / total.
            Returns 0.0 if total is 0 to avoid division by zero.

        Examples
        --------
        >>> ctx = BatchContext(index=4, total=10, item=None)
        >>> ctx.progress
        0.5
        """
        if self.total == 0:
            return 0.0
        return (self.index + 1) / self.total

    @property
    def is_first(self) -> bool:
        """Return True if this is the first item.

        Returns
        -------
        bool
            True if index is 0.

        Examples
        --------
        >>> ctx = BatchContext(index=0, total=5, item=None)
        >>> ctx.is_first
        True
        """
        return self.index == 0

    @property
    def is_last(self) -> bool:
        """Return True if this is the last item.

        Returns
        -------
        bool
            True if index equals total - 1.

        Examples
        --------
        >>> ctx = BatchContext(index=4, total=5, item=None)
        >>> ctx.is_last
        True
        """
        return self.index == self.total - 1

    def __str__(self) -> str:
        """Return human-readable string representation.

        Returns
        -------
        str
            Format: "[index+1/total] item"

        Examples
        --------
        >>> ctx = BatchContext(index=2, total=10, item="image.tif")
        >>> str(ctx)
        '[3/10] image.tif'
        """
        return f'[{self.index + 1}/{self.total}] {self.item}'
