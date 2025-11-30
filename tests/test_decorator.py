"""Tests for the @batch decorator."""

from pathlib import Path

import pytest

from nbatch import BatchContext, batch


class TestBatchDecorator:
    """Test suite for @batch decorator."""

    def test_single_item_returns_directly(self, tmp_path):
        """Test that single item input returns result directly."""
        test_file = tmp_path / 'test.tif'
        test_file.touch()

        @batch
        def process(path: Path) -> str:
            return path.stem

        result = process(test_file)

        assert result == 'test'
        assert not hasattr(result, '__iter__') or isinstance(result, str)

    def test_list_returns_generator(self, tmp_path):
        """Test that list input returns generator."""
        files = [tmp_path / 'a.tif', tmp_path / 'b.tif']
        for f in files:
            f.touch()

        @batch
        def process(path: Path) -> str:
            return path.stem

        result = process(files)

        # Should be a generator
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')

        results = list(result)
        assert results == ['a', 'b']

    def test_directory_returns_generator(self, tmp_path):
        """Test that directory input returns generator."""
        (tmp_path / 'x.tif').touch()
        (tmp_path / 'y.tif').touch()

        @batch
        def process(path: Path) -> str:
            return path.stem

        result = process(tmp_path)

        results = list(result)
        assert 'x' in results
        assert 'y' in results

    def test_directory_with_patterns(self, tmp_path):
        """Test directory processing with custom patterns."""
        (tmp_path / 'a.tif').touch()
        (tmp_path / 'b.tif').touch()
        (tmp_path / 'c.png').touch()

        @batch(patterns='*.tif')
        def process(path: Path) -> str:
            return path.stem

        results = list(process(tmp_path))

        assert len(results) == 2
        assert 'c' not in results

    def test_with_context_false(self, tmp_path):
        """Test that with_context=False yields only results."""
        files = [tmp_path / 'a.tif', tmp_path / 'b.tif']
        for f in files:
            f.touch()

        @batch(with_context=False)
        def process(path: Path) -> str:
            return path.stem

        results = list(process(files))

        assert results == ['a', 'b']
        assert all(isinstance(r, str) for r in results)

    def test_with_context_true(self, tmp_path):
        """Test that with_context=True yields (result, context) tuples."""
        files = [tmp_path / 'a.tif', tmp_path / 'b.tif']
        for f in files:
            f.touch()

        @batch(with_context=True)
        def process(path: Path) -> str:
            return path.stem

        results = list(process(files))

        assert len(results) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

        # Check first result
        result, ctx = results[0]
        assert result == 'a'
        assert isinstance(ctx, BatchContext)
        assert ctx.index == 0
        assert ctx.total == 2

        # Check last result
        result, ctx = results[1]
        assert result == 'b'
        assert ctx.index == 1
        assert ctx.is_last is True

    def test_on_error_raise(self, tmp_path):
        """Test on_error='raise' re-raises exceptions."""
        files = [tmp_path / 'a_good.tif', tmp_path / 'b_bad.tif']
        for f in files:
            f.touch()

        @batch(on_error='raise')
        def process(path: Path) -> str:
            if 'bad' in path.name:
                raise ValueError('Bad file!')
            return path.stem

        gen = process(files)

        # First item should work (a_good comes before b_bad alphabetically)
        assert next(gen) == 'a_good'

        # Second should raise
        with pytest.raises(ValueError, match='Bad file'):
            next(gen)

    def test_on_error_continue(self, tmp_path):
        """Test on_error='continue' yields None for failed items."""
        files = [tmp_path / 'a.tif', tmp_path / 'bad.tif', tmp_path / 'c.tif']
        for f in files:
            f.touch()

        @batch(on_error='continue')
        def process(path: Path) -> str:
            if 'bad' in path.name:
                raise ValueError('Bad file!')
            return path.stem

        results = list(process(files))

        assert results == ['a', None, 'c']

    def test_on_error_continue_with_context(self, tmp_path):
        """Test on_error='continue' with context yields (None, ctx)."""
        files = [tmp_path / 'a_good.tif', tmp_path / 'b_bad.tif']
        for f in files:
            f.touch()

        @batch(on_error='continue', with_context=True)
        def process(path: Path) -> str:
            if 'bad' in path.name:
                raise ValueError('Bad file!')
            return path.stem

        results = list(process(files))

        # a_good comes before b_bad alphabetically
        result1, ctx1 = results[0]
        assert result1 == 'a_good'

        result2, ctx2 = results[1]
        assert result2 is None
        assert ctx2.item.name == 'b_bad.tif'

    def test_on_error_skip(self, tmp_path):
        """Test on_error='skip' doesn't yield failed items."""
        files = [tmp_path / 'a.tif', tmp_path / 'bad.tif', tmp_path / 'c.tif']
        for f in files:
            f.touch()

        @batch(on_error='skip')
        def process(path: Path) -> str:
            if 'bad' in path.name:
                raise ValueError('Bad file!')
            return path.stem

        results = list(process(files))

        assert results == ['a', 'c']
        assert len(results) == 2

    def test_decorator_without_parentheses(self, tmp_path):
        """Test @batch without parentheses works."""
        test_file = tmp_path / 'test.tif'
        test_file.touch()

        @batch
        def process(path: Path) -> str:
            return path.stem

        assert process(test_file) == 'test'

    def test_decorator_with_parentheses(self, tmp_path):
        """Test @batch() with empty parentheses works."""
        test_file = tmp_path / 'test.tif'
        test_file.touch()

        @batch()
        def process(path: Path) -> str:
            return path.stem

        assert process(test_file) == 'test'

    def test_additional_args(self, tmp_path):
        """Test that additional arguments are passed through."""
        files = [tmp_path / 'a.tif', tmp_path / 'b.tif']
        for f in files:
            f.touch()

        @batch
        def process(path: Path, suffix: str, prefix: str = '') -> str:
            return f'{prefix}{path.stem}{suffix}'

        results = list(process(files, '_processed', prefix='img_'))

        assert results == ['img_a_processed', 'img_b_processed']

    def test_kwargs_only(self, tmp_path):
        """Test with keyword-only arguments."""
        test_file = tmp_path / 'test.tif'
        test_file.touch()

        @batch
        def process(path: Path, *, multiplier: int = 1) -> int:
            return len(path.stem) * multiplier

        assert process(test_file, multiplier=2) == 8  # 'test' * 2

    def test_non_path_items(self):
        """Test batch processing with non-path items."""
        items = [1, 2, 3, 4, 5]

        @batch
        def square(x: int) -> int:
            return x * x

        results = list(square(items))

        assert results == [1, 4, 9, 16, 25]

    def test_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""

        @batch
        def my_function(x):
            """My docstring."""
            return x

        assert my_function.__name__ == 'my_function'
        assert my_function.__doc__ == 'My docstring.'

    def test_recursive_false(self, tmp_path):
        """Test recursive=False only processes top-level directory."""
        (tmp_path / 'top.tif').touch()
        subdir = tmp_path / 'sub'
        subdir.mkdir()
        (subdir / 'nested.tif').touch()

        @batch(recursive=False, patterns='*.tif')
        def process(path: Path) -> str:
            return path.stem

        results = list(process(tmp_path))

        assert results == ['top']

    def test_context_progress_tracking(self, tmp_path):
        """Test that context enables accurate progress tracking."""
        files = [tmp_path / f'{i}.tif' for i in range(10)]
        for f in files:
            f.touch()

        @batch(with_context=True)
        def process(path: Path) -> str:
            return path.stem

        progress_values = []
        for _result, ctx in process(files):
            progress_values.append(ctx.progress)

        # Progress should go from 0.1 to 1.0
        assert progress_values[0] == 0.1
        assert progress_values[-1] == 1.0
        assert progress_values == [i / 10 for i in range(1, 11)]


class TestBatchIntegration:
    """Integration tests for @batch with other components."""

    def test_with_batch_logger(self, tmp_path):
        """Test @batch works well with batch_logger."""
        from nbatch import batch_logger

        files = [tmp_path / 'a.tif', tmp_path / 'b.tif']
        for f in files:
            f.touch()
        log_file = tmp_path / 'process.log'

        @batch(with_context=True)
        def process(path: Path) -> str:
            return path.stem.upper()

        with batch_logger(log_file, header={'Files': len(files)}) as log:
            for result, ctx in process(files):
                log(ctx, f'Processed: {result}')

        content = log_file.read_text()
        assert 'Files: 2' in content
        assert 'Processed: A' in content
        assert 'Processed: B' in content

    def test_generator_can_be_iterated_once(self, tmp_path):
        """Test that generator follows normal generator semantics."""
        files = [tmp_path / 'a.tif', tmp_path / 'b.tif']
        for f in files:
            f.touch()

        @batch
        def process(path: Path) -> str:
            return path.stem

        gen = process(files)

        # First iteration
        results1 = list(gen)
        assert results1 == ['a', 'b']

        # Second iteration should be empty (generator exhausted)
        results2 = list(gen)
        assert results2 == []

    def test_empty_directory(self, tmp_path):
        """Test handling of empty directory raises ValueError."""
        empty_dir = tmp_path / 'empty'
        empty_dir.mkdir()

        @batch
        def process(path: Path) -> str:
            return path.stem

        with pytest.raises(ValueError, match='No files found'):
            list(process(empty_dir))
