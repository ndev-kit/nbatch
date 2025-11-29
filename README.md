# ndev-batch

[![License BSD-3](https://img.shields.io/pypi/l/ndev-batch.svg?color=green)](https://github.com/ndev-kit/ndev-batch/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ndev-batch.svg?color=green)](https://pypi.org/project/ndev-batch)
[![Python Version](https://img.shields.io/pypi/pyversions/ndev-batch.svg?color=green)](https://python.org)
[![tests](https://github.com/ndev-kit/ndev-batch/workflows/tests/badge.svg)](https://github.com/ndev-kit/ndev-batch/actions)
[![codecov](https://codecov.io/gh/ndev-kit/ndev-batch/branch/main/graph/badge.svg)](https://codecov.io/gh/ndev-kit/ndev-batch)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/ndev-batch)](https://napari-hub.org/plugins/ndev-batch)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

**Lightweight batch processing utilities for the ndev-kit ecosystem.**

ndev-batch provides a foundation for batch processing operations. It's designed to work seamlessly with napari plugins but has **no napari or Qt dependencies**.

## Features

- **`@batch` decorator** - Transform single-item functions into batch-capable functions
- **`BatchContext`** - Track progress through batch operations
- **`discover_files()`** - Flexible file discovery with natural sorting (like file explorers)
- **`batch_logger`** - Scoped logging for batch operations with headers/footers
- **Minimal dependencies** - Only requires natsort for natural file ordering

## Installation

```bash
pip install ndev-batch
```

For development:

```bash
pip install -e . --group dev
```

## Quick Start

### Basic Batch Processing

The `@batch` decorator transforms a function that processes a single item into one that handles both single items and batches:

```python
from pathlib import Path
from ndev_batch import batch

@batch
def process_image(path: Path) -> str:
    # Your processing logic here
    return path.stem.upper()

# Single item - returns result directly
result = process_image(Path("image.tif"))
# Returns: "IMAGE"

# List of items - returns generator
results = process_image([Path("a.tif"), Path("b.tif")])
list(results)
# Returns: ["A", "B"]

# Directory - discovers files and returns generator
results = process_image(Path("/data/images"))
# Processes all files in directory
```

### Progress Tracking

Use `with_context=True` to get progress information:

```python
@batch(with_context=True)
def process_image(path: Path) -> str:
    return path.stem

for result, ctx in process_image(files):
    print(f"{ctx.progress:.0%} complete: {result}")
    # 10% complete: image1
    # 20% complete: image2
    # ...
```

The `BatchContext` provides:

- `ctx.index` - Zero-based index of current item
- `ctx.total` - Total number of items
- `ctx.item` - The current item being processed
- `ctx.progress` - Progress as fraction (0.0 to 1.0)
- `ctx.is_first` / `ctx.is_last` - Boolean flags

### Error Handling

Control how errors are handled with `on_error`:

```python
# 'raise' (default) - Re-raise exceptions immediately
@batch(on_error='raise')
def strict_process(path): ...

# 'continue' - Log error and yield None for failed items
@batch(on_error='continue')
def lenient_process(path): ...
# Results: ["good", None, "ok"]

# 'skip' - Log error and skip failed items entirely
@batch(on_error='skip')
def skip_errors(path): ...
# Results: ["good", "ok"]
```

### File Discovery

Control which files are processed:

```python
# Custom glob patterns
@batch(patterns='*.tif')
def process_tiffs(path): ...

# Multiple patterns
@batch(patterns=['*.tif', '*.tiff', '*.png'])
def process_images(path): ...

# Non-recursive (top-level only)
@batch(recursive=False)
def process_top_level(path): ...
```

Or use `discover_files()` directly:

```python
from ndev_batch import discover_files

# From directory with patterns
files = discover_files("/data/images", patterns=["*.tif", "*.png"])

# From explicit list
files = discover_files([path1, path2, path3])
```

### Logging

Use `batch_logger` for structured logging. By default, it outputs to the console (stderr). Optionally log to a file:

```python
from ndev_batch import batch, batch_logger

@batch(with_context=True)
def process(path):
    return path.stem

# Console only (default)
with batch_logger() as log:
    for result, ctx in process(files):
        log(ctx, f"Processed: {result}")

# With file logging (appends by default)
with batch_logger(log_file="output/process.log", header={"Files": 100}) as log:
    for result, ctx in process(files):
        log(ctx, f"Processed: {result}")
        # Or use log.info(), log.warning(), log.error()

# File only (no console output)
with batch_logger(log_file="output/quiet.log", console=False) as log:
    for result, ctx in process(files):
        log(ctx, f"Processed: {result}")
```

Log file output:
```
============================================================
Batch processing started at 2025-01-29 10:30:00
------------------------------------------------------------
Files: 100
============================================================
2025-01-29 10:30:01 - INFO - [1/100] image1.tif - Processed: image1
2025-01-29 10:30:02 - INFO - [2/100] image2.tif - Processed: image2
...
============================================================
Batch processing completed at 2025-01-29 10:35:00
============================================================
```

## Integration with napari

ndev-batch is designed to work seamlessly with napari's `@thread_worker`:

```python
from napari.qt.threading import thread_worker
from ndev_batch import batch, batch_logger

@batch(with_context=True, on_error='continue')
def process_image(path, model, output_dir):
    # Your processing logic
    result = model.predict(load_image(path))
    save_result(result, output_dir / path.name)
    return result

# In your widget
def run_batch(self):
    @thread_worker
    def _run():
        with batch_logger(log_file=self.output_dir / 'log.txt') as log:
            for result, ctx in process_image(
                self.input_dir,
                model=self.model,
                output_dir=self.output_dir,
            ):
                log(ctx, f"Processed: {ctx.item.name}")
                yield ctx  # Enables progress updates
    
    worker = _run()
    worker.yielded.connect(
        lambda ctx: self.progress_bar.setValue(int(ctx.progress * 100))
    )
    worker.start()
```

## API Reference

### `@batch` Decorator

```python
@batch(
    on_error: Literal['raise', 'continue', 'skip'] = 'raise',
    with_context: bool = False,
    patterns: str | Sequence[str] = '*',
    recursive: bool = False,
)
```

### `BatchContext`

```python
@dataclass(frozen=True)
class BatchContext:
    index: int      # Zero-based index
    total: int      # Total items
    item: Any       # Current item
    
    @property
    def progress(self) -> float: ...  # (index + 1) / total
    @property
    def is_first(self) -> bool: ...   # index == 0
    @property
    def is_last(self) -> bool: ...    # index == total - 1
```

### `discover_files()`

```python
def discover_files(
    source: str | Path | Iterable[str | Path],
    patterns: str | Sequence[str] = '*',
    recursive: bool = False,
) -> list[Path]: ...
```

### `batch_logger`

```python
@contextmanager
def batch_logger(
    log_file: str | Path | None = None,  # Optional file path
    header: Mapping[str, object] | None = None,  # Metadata to write at start
    level: int = logging.INFO,
    console: bool = True,  # Output to stderr
    file_mode: Literal['w', 'a'] = 'a',  # Append by default
) -> Generator[BatchLogger, None, None]: ...
```

## Contributing

Contributions are welcome! Please ensure tests pass before submitting a pull request:

```bash
pytest --cov=src/ndev_batch
```

## License

Distributed under the terms of the [BSD-3](http://opensource.org/licenses/BSD-3-Clause) license.

## Part of ndev-kit

ndev-batch is part of the [ndev-kit](https://github.com/ndev-kit) ecosystem for no-code bioimage analysis in napari.
