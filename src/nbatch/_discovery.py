"""File discovery utilities for batch processing.

This module provides utilities for discovering files to process,
supporting both directory scanning and explicit file lists.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from natsort import natsorted

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def discover_files(
    source: str | Path | Iterable[str | Path],
    patterns: str | Sequence[str] = '*',
    recursive: bool = False,
) -> list[Path]:
    """Discover files from a path, directory, or iterable of paths.

    This utility normalizes various input types into a consistent list of
    Path objects for batch processing.

    Parameters
    ----------
    source : str | Path | Iterable[str | Path]
        The source to discover files from. Can be:
        - A single file path (returned as single-item list)
        - A directory path (scanned for files matching patterns)
        - An iterable of file paths (converted to Path objects)
    patterns : str | Sequence[str], optional
        Glob pattern(s) to match when source is a directory.
        Default is '*' (all files). Multiple patterns can be provided
        as a sequence, e.g., ['*.tif', '*.tiff', '*.png'].
    recursive : bool, optional
        If True and source is a directory, search recursively using '**'.
        Default is False.

    Returns
    -------
    list[Path]
        List of discovered file paths, naturally sorted (like file explorers).

    Raises
    ------
    FileNotFoundError
        If source is a path that doesn't exist.
    ValueError
        If source is empty or no files match the patterns.

    Examples
    --------
    Single file:

    >>> discover_files("image.tif")
    [PosixPath('image.tif')]

    Directory with pattern:

    >>> discover_files("/data/images", patterns="*.tif")
    [PosixPath('/data/images/a.tif'), PosixPath('/data/images/b.tif')]

    Multiple patterns:

    >>> discover_files("/data", patterns=["*.tif", "*.png"])
    [PosixPath('/data/img1.png'), PosixPath('/data/img2.tif')]

    Explicit list:

    >>> discover_files(["file1.tif", "file2.tif"])
    [PosixPath('file1.tif'), PosixPath('file2.tif')]
    """
    # Handle string conversion to Path
    if isinstance(source, str):
        source = Path(source)

    # Single Path
    if isinstance(source, Path):
        if not source.exists():
            raise FileNotFoundError(f'Source path does not exist: {source}')

        if source.is_file():
            return [source]

        if source.is_dir():
            return _discover_from_directory(source, patterns, recursive)

        # Path exists but is neither file nor directory (rare edge case)
        raise ValueError(f'Source is neither a file nor directory: {source}')

    # Iterable of paths
    files = [Path(p) if isinstance(p, str) else p for p in source]

    if not files:
        raise ValueError('Source iterable is empty')

    # Verify all files exist
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f'The following files do not exist: {missing[:5]}'
            + (f' ... and {len(missing) - 5} more' if len(missing) > 5 else '')
        )

    return natsorted(files)


def _discover_from_directory(
    directory: Path,
    patterns: str | Sequence[str],
    recursive: bool,
) -> list[Path]:
    """Discover files from a directory using glob patterns.

    Parameters
    ----------
    directory : Path
        Directory to search.
    patterns : str | Sequence[str]
        Glob pattern(s) to match.
    recursive : bool
        Whether to search recursively.

    Returns
    -------
    list[Path]
        Sorted list of matching file paths.

    Raises
    ------
    ValueError
        If no files match the patterns.
    """
    # Normalize patterns to list
    if isinstance(patterns, str):
        patterns = [patterns]

    # Collect all matching files
    files: set[Path] = set()

    for pattern in patterns:
        # Add recursive prefix if needed
        if recursive and not pattern.startswith('**'):
            pattern = f'**/{pattern}'

        matches = directory.glob(pattern)
        # Only include files, not directories
        files.update(p for p in matches if p.is_file())

    if not files:
        raise ValueError(
            f'No files found in {directory} matching patterns: {patterns}'
        )

    return natsorted(files)


def is_batch_input(item: object) -> bool:
    """Check if an item represents batch input (multiple items).

    This helper determines whether the input should trigger batch processing
    (iteration) or single-item processing.

    Parameters
    ----------
    item : object
        The input to check.

    Returns
    -------
    bool
        True if item is a list, tuple, or directory Path.
        False for single files, strings, or other objects.

    Examples
    --------
    >>> is_batch_input(Path("/data/images"))  # directory
    True
    >>> is_batch_input([Path("a.tif"), Path("b.tif")])
    True
    >>> is_batch_input(Path("single.tif"))  # file
    False
    >>> is_batch_input("single.tif")
    False
    """
    if isinstance(item, list | tuple):
        return True
    return isinstance(item, Path) and item.is_dir()
