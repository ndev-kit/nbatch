"""Tests for BatchRunner."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from nbatch import BatchRunner, batch


class TestBatchRunnerBasic:
    """Test basic BatchRunner functionality."""

    def test_init_default(self):
        """Test BatchRunner initializes with default values."""
        runner = BatchRunner()
        assert runner.is_running is False
        assert runner.was_cancelled is False

    def test_init_with_callbacks(self):
        """Test BatchRunner initializes with callbacks."""
        on_item = MagicMock()
        on_complete = MagicMock()
        on_error = MagicMock()
        on_cancel = MagicMock()

        runner = BatchRunner(
            on_item_complete=on_item,
            on_complete=on_complete,
            on_error=on_error,
            on_cancel=on_cancel,
        )

        assert runner._on_item_complete is on_item
        assert runner._on_complete is on_complete
        assert runner._on_error is on_error
        assert runner._on_cancel is on_cancel


class TestBatchRunnerSync:
    """Test BatchRunner synchronous execution."""

    def test_run_sync_simple_function(self):
        """Test running a simple function synchronously."""
        results = []

        def process(item):
            return item * 2

        runner = BatchRunner(
            on_item_complete=lambda r, ctx: results.append(r),
        )
        runner.run(process, [1, 2, 3], threaded=False)

        assert results == [2, 4, 6]
        assert runner.is_running is False

    def test_run_sync_with_batch_decorated(self):
        """Test running a @batch decorated function synchronously."""
        results = []

        @batch
        def process(item):
            return item.upper()

        runner = BatchRunner(
            on_item_complete=lambda r, ctx: results.append(r),
        )
        runner.run(process, ['a', 'b', 'c'], threaded=False)

        assert results == ['A', 'B', 'C']

    def test_run_sync_with_args_kwargs(self):
        """Test passing additional args and kwargs."""
        results = []

        def process(item, multiplier, suffix=''):
            return f'{item * multiplier}{suffix}'

        runner = BatchRunner(
            on_item_complete=lambda r, ctx: results.append(r),
        )
        runner.run(process, [1, 2, 3], 10, suffix='!', threaded=False)

        assert results == ['10!', '20!', '30!']

    def test_run_sync_context_values(self):
        """Test BatchContext values in callbacks."""
        contexts = []

        def process(item):
            return item

        runner = BatchRunner(
            on_item_complete=lambda r, ctx: contexts.append(ctx),
        )
        runner.run(process, ['a', 'b', 'c'], threaded=False)

        assert len(contexts) == 3

        assert contexts[0].index == 0
        assert contexts[0].total == 3
        assert contexts[0].item == 'a'
        assert contexts[0].is_first is True
        assert contexts[0].is_last is False

        assert contexts[1].index == 1
        assert contexts[1].is_first is False
        assert contexts[1].is_last is False

        assert contexts[2].index == 2
        assert contexts[2].is_first is False
        assert contexts[2].is_last is True

    def test_run_sync_on_complete_called(self):
        """Test on_complete callback is called after all items."""
        on_complete = MagicMock()

        def process(item):
            return item

        runner = BatchRunner(on_complete=on_complete)
        runner.run(process, [1, 2, 3], threaded=False)

        on_complete.assert_called_once()

    def test_run_sync_on_complete_not_called_on_cancel(self):
        """Test on_complete is NOT called when cancelled."""
        on_complete = MagicMock()
        on_cancel = MagicMock()

        def process(item):
            # Cancel after first item
            if item == 2:
                runner.cancel()
            return item

        runner = BatchRunner(
            on_complete=on_complete,
            on_cancel=on_cancel,
        )
        runner.run(process, [1, 2, 3], threaded=False)

        on_complete.assert_not_called()
        on_cancel.assert_called_once()


class TestBatchRunnerErrorHandling:
    """Test BatchRunner error handling."""

    def test_run_sync_error_callback(self):
        """Test on_error callback is called on exceptions."""
        errors = []

        def process(item):
            if item == 'bad':
                raise ValueError('Bad item!')
            return item

        runner = BatchRunner(
            on_error=lambda ctx, e: errors.append((ctx.item, str(e))),
        )
        runner.run(process, ['good', 'bad', 'ok'], threaded=False)

        assert len(errors) == 1
        assert errors[0][0] == 'bad'
        assert 'Bad item!' in errors[0][1]

    def test_run_sync_continues_after_error(self):
        """Test processing continues after an error."""
        results = []

        def process(item):
            if item == 2:
                raise ValueError('Skip this')
            return item * 10

        runner = BatchRunner(
            on_item_complete=lambda r, ctx: results.append(r),
        )
        runner.run(process, [1, 2, 3], threaded=False)

        # Item 2 failed, so only items 1 and 3 succeeded
        assert results == [10, 30]


class TestBatchRunnerCancellation:
    """Test BatchRunner cancellation."""

    def test_cancel_not_running(self):
        """Test cancel when not running does nothing."""
        runner = BatchRunner()
        runner.cancel()  # Should not raise
        assert runner.was_cancelled is False

    def test_cancel_sync(self):
        """Test cancellation during sync run."""
        processed = []

        def process(item):
            processed.append(item)
            if item == 2:
                runner.cancel()
            return item

        runner = BatchRunner()
        runner.run(process, [1, 2, 3, 4, 5], threaded=False)

        # Should stop after item 2 (current item completes)
        assert processed == [1, 2]
        assert runner.was_cancelled is True
        assert runner.is_running is False

    def test_was_cancelled_reset_on_new_run(self):
        """Test was_cancelled is reset when starting a new run."""

        def process(item):
            if item == 2:
                runner.cancel()
            return item

        runner = BatchRunner()

        # First run - cancel
        runner.run(process, [1, 2, 3], threaded=False)
        assert runner.was_cancelled is True

        # Second run - complete normally
        runner.run(lambda x: x, [1, 2, 3], threaded=False)
        assert runner.was_cancelled is False


class TestBatchRunnerFileDiscovery:
    """Test BatchRunner file discovery integration."""

    def test_run_with_directory(self, tmp_path):
        """Test running with a directory path."""
        # Create test files
        (tmp_path / 'a.txt').write_text('a')
        (tmp_path / 'b.txt').write_text('b')
        (tmp_path / 'c.txt').write_text('c')

        results = []

        def process(file_path):
            return file_path.stem

        runner = BatchRunner(
            on_item_complete=lambda r, ctx: results.append(r),
        )
        runner.run(process, tmp_path, patterns='*.txt', threaded=False)

        assert sorted(results) == ['a', 'b', 'c']

    def test_run_with_file_list(self, tmp_path):
        """Test running with a list of file paths."""
        files = [
            tmp_path / 'x.txt',
            tmp_path / 'y.txt',
        ]
        for f in files:
            f.write_text('content')

        results = []

        def process(file_path):
            return file_path.stem

        runner = BatchRunner(
            on_item_complete=lambda r, ctx: results.append(r),
        )
        runner.run(process, files, threaded=False)

        assert sorted(results) == ['x', 'y']

    def test_run_with_single_file(self, tmp_path):
        """Test running with a single file path."""
        file = tmp_path / 'single.txt'
        file.write_text('content')

        results = []

        def process(file_path):
            return file_path.stem

        runner = BatchRunner(
            on_item_complete=lambda r, ctx: results.append(r),
        )
        runner.run(process, file, threaded=False)

        assert results == ['single']


class TestBatchRunnerLogging:
    """Test BatchRunner logging integration."""

    def test_run_with_log_file(self, tmp_path):
        """Test logging to file during run."""
        log_file = tmp_path / 'batch.log'

        def process(item):
            return item * 2

        runner = BatchRunner()
        runner.run(
            process,
            [1, 2, 3],
            threaded=False,
            log_file=log_file,
            log_header={'Test': 'value', 'Items': 3},
        )

        assert log_file.exists()
        content = log_file.read_text()
        assert 'Test: value' in content
        assert 'Items: 3' in content
        assert 'Completed' in content

    def test_run_logs_errors(self, tmp_path):
        """Test errors are logged to file."""
        log_file = tmp_path / 'errors.log'

        def process(item):
            if item == 'fail':
                raise ValueError('Intentional failure')
            return item

        runner = BatchRunner()
        runner.run(
            process,
            ['ok', 'fail', 'also_ok'],
            threaded=False,
            log_file=log_file,
        )

        content = log_file.read_text()
        assert 'Failed' in content
        assert 'fail' in content


class TestBatchRunnerReusability:
    """Test that BatchRunner is reusable."""

    def test_run_multiple_times(self):
        """Test runner can be used for multiple batches."""
        all_results = []
        batch_count = [0]

        def on_complete():
            batch_count[0] += 1

        runner = BatchRunner(
            on_item_complete=lambda r, ctx: all_results.append(r),
            on_complete=on_complete,
        )

        # First batch
        runner.run(lambda x: x * 2, [1, 2, 3], threaded=False)
        assert all_results == [2, 4, 6]
        assert batch_count[0] == 1

        # Second batch
        runner.run(lambda x: x + 10, [1, 2], threaded=False)
        assert all_results == [2, 4, 6, 11, 12]
        assert batch_count[0] == 2

    def test_cannot_run_while_running(self):
        """Test that starting a new run while running raises error."""
        # This is tricky to test without threading, but we can test the check
        runner = BatchRunner()

        # Manually set running state
        runner._is_running = True

        with pytest.raises(RuntimeError, match='already running'):
            runner.run(lambda x: x, [1, 2, 3], threaded=False)

        # Reset for cleanup
        runner._is_running = False


class TestBatchRunnerThreaded:
    """Test BatchRunner threaded execution (without napari)."""

    def test_run_threaded_fallback_no_napari(self, monkeypatch):
        """Test threaded execution using fallback (concurrent.futures)."""
        # Force the fallback path by pretending napari isn't available
        import nbatch._runner as runner_module

        monkeypatch.setattr(runner_module, 'HAS_NAPARI', False)

        results = []
        completed = []

        def process(item):
            time.sleep(0.01)  # Small delay
            return item * 2

        runner = BatchRunner(
            on_item_complete=lambda r, ctx: results.append(r),
            on_complete=lambda: completed.append(True),
        )

        # Run threaded (will use fallback since we monkeypatched HAS_NAPARI)
        runner.run(process, [1, 2, 3], threaded=True)

        # Wait for completion
        timeout = 5.0
        start = time.time()
        while runner.is_running and (time.time() - start) < timeout:
            time.sleep(0.05)

        assert runner.is_running is False
        assert sorted(results) == [2, 4, 6]
        assert completed == [True]


class TestBatchRunnerIntegrationWithBatchDecorator:
    """Test BatchRunner integration with @batch decorator."""

    def test_batch_decorated_function(self):
        """Test runner works with @batch decorated functions."""
        results = []

        @batch(on_error='continue')
        def process(item):
            if item == 'bad':
                raise ValueError('Bad!')
            return item.upper()

        runner = BatchRunner(
            on_item_complete=lambda r, ctx: results.append(r),
        )

        # Note: @batch returns generator for batch input, but runner
        # calls function for each item individually
        runner.run(process, ['a', 'bad', 'c'], threaded=False)

        # Since runner calls per-item, error handling is in runner
        # The @batch decorator's on_error doesn't apply here
        assert 'A' in results
        assert 'C' in results
