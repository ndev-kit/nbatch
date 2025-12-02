"""Tests for BatchRunner."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from nbatch import BatchRunner, batch

# Check for napari/Qt availability for threading tests
try:
    from napari.qt.threading import create_worker  # noqa: F401

    HAS_NAPARI = True
except ImportError:
    HAS_NAPARI = False


class TestBatchRunnerBasic:
    """Test basic BatchRunner functionality."""

    def test_init_default(self):
        """Test BatchRunner initializes with default values."""
        runner = BatchRunner()
        assert runner.is_running is False
        assert runner.was_cancelled is False

    def test_init_with_callbacks(self):
        """Test BatchRunner initializes with callbacks."""
        on_start = MagicMock()
        on_item = MagicMock()
        on_complete = MagicMock()
        on_error = MagicMock()
        on_cancel = MagicMock()

        runner = BatchRunner(
            on_start=on_start,
            on_item_complete=on_item,
            on_complete=on_complete,
            on_error=on_error,
            on_cancel=on_cancel,
        )

        assert runner._on_start is on_start
        assert runner._on_item_complete is on_item
        assert runner._on_complete is on_complete
        assert runner._on_error is on_error
        assert runner._on_cancel is on_cancel

    def test_error_count_initial(self):
        """Test error_count is 0 initially."""
        runner = BatchRunner()
        assert runner.error_count == 0

    def test_on_start_callback(self):
        """Test on_start callback is called with total count."""
        on_start = MagicMock()

        def process(item):
            return item

        runner = BatchRunner(on_start=on_start)
        runner.run(process, [1, 2, 3, 4, 5], threaded=False)

        on_start.assert_called_once_with(5)

    def test_error_count_tracking(self):
        """Test error_count tracks errors during batch."""

        def process(item):
            if item in [2, 4]:
                raise ValueError(f'Error on {item}')
            return item

        runner = BatchRunner()
        runner.run(process, [1, 2, 3, 4, 5], threaded=False)

        assert runner.error_count == 2

    def test_error_count_reset_on_new_run(self):
        """Test error_count is reset at start of each run."""

        def fail_some(item):
            if item == 2:
                raise ValueError('Error')
            return item

        def succeed_all(item):
            return item

        runner = BatchRunner()

        # First run with error
        runner.run(fail_some, [1, 2, 3], threaded=False)
        assert runner.error_count == 1

        # Second run without errors - error_count should reset
        runner.run(succeed_all, [1, 2, 3], threaded=False)
        assert runner.error_count == 0


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

    def test_kwargs_replaces_partial(self):
        """Test kwargs pattern as alternative to functools.partial.

        This demonstrates the recommended pattern for passing extra arguments
        to batch functions without using functools.partial.
        """
        from functools import partial

        results_kwargs = []
        results_partial = []

        def process_item(item, output_dir, sigma=1.0):
            """Simulate processing with configurable parameters."""
            return f'{item}_sigma{sigma}_to_{output_dir}'

        # Using kwargs (recommended pattern - cleaner!)
        runner1 = BatchRunner(
            on_item_complete=lambda r, ctx: results_kwargs.append(r),
        )
        runner1.run(
            process_item,
            ['img1', 'img2'],
            output_dir='output',
            sigma=2.5,
            threaded=False,
        )

        # Using partial (old pattern - still works but more verbose)
        runner2 = BatchRunner(
            on_item_complete=lambda r, ctx: results_partial.append(r),
        )
        process_func = partial(process_item, output_dir='output', sigma=2.5)
        runner2.run(process_func, ['img1', 'img2'], threaded=False)

        # Both approaches produce identical results
        assert results_kwargs == results_partial
        assert results_kwargs == [
            'img1_sigma2.5_to_output',
            'img2_sigma2.5_to_output',
        ]

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

        monkeypatch.setattr(runner_module, '_HAS_NAPARI', False)

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


@pytest.mark.skipif(not HAS_NAPARI, reason='requires napari and Qt')
class TestBatchRunnerNapariThreading:
    """Test BatchRunner with napari's threading (requires Qt event loop)."""

    def test_run_napari_threaded_basic(self, qtbot):
        """Test basic napari threaded execution."""
        results = []
        completed = []

        def process(item):
            return item * 2

        runner = BatchRunner(
            on_item_complete=lambda r, ctx: results.append(r),
            on_complete=lambda: completed.append(True),
        )

        runner.run(process, [1, 2, 3], threaded=True)

        # Use qtbot to wait for completion
        def check_complete():
            assert not runner.is_running

        qtbot.waitUntil(check_complete, timeout=5000)

        assert sorted(results) == [2, 4, 6]
        assert completed == [True]

    def test_run_napari_threaded_with_context(self, qtbot):
        """Test napari threaded execution provides correct context."""
        contexts = []

        def process(item):
            return item

        runner = BatchRunner(
            on_item_complete=lambda r, ctx: contexts.append(
                (ctx.index, ctx.total, ctx.item)
            ),
        )

        runner.run(process, ['a', 'b', 'c'], threaded=True)

        qtbot.waitUntil(lambda: not runner.is_running, timeout=5000)

        assert len(contexts) == 3
        assert contexts[0] == (0, 3, 'a')
        assert contexts[1] == (1, 3, 'b')
        assert contexts[2] == (2, 3, 'c')

    def test_run_napari_threaded_with_errors(self, qtbot):
        """Test napari threaded execution handles errors."""
        results = []
        errors = []

        def process(item):
            if item == 2:
                raise ValueError('Item 2 fails!')
            return item * 10

        runner = BatchRunner(
            on_item_complete=lambda r, ctx: results.append(r),
            on_error=lambda ctx, e: errors.append((ctx.item, str(e))),
        )

        runner.run(process, [1, 2, 3], threaded=True)

        qtbot.waitUntil(lambda: not runner.is_running, timeout=5000)

        assert 10 in results
        assert 30 in results
        assert len(errors) == 1
        assert errors[0][0] == 2
        assert 'Item 2 fails!' in errors[0][1]

    def test_run_napari_threaded_cancellation(self, qtbot):
        """Test cancellation during napari threaded execution."""
        processed = []
        cancelled = []

        def process(item):
            processed.append(item)
            time.sleep(0.1)  # Slow enough to allow cancellation
            return item

        runner = BatchRunner(
            on_cancel=lambda: cancelled.append(True),
        )

        runner.run(process, [1, 2, 3, 4, 5], threaded=True)

        # Wait a bit then cancel
        qtbot.wait(150)  # Let first item or two process
        runner.cancel()

        qtbot.waitUntil(lambda: not runner.is_running, timeout=5000)

        # Should have processed fewer than all items
        assert len(processed) < 5
        assert runner.was_cancelled is True
        assert cancelled == [True]

    def test_run_napari_threaded_with_logging(self, qtbot, tmp_path):
        """Test napari threaded execution with logging."""
        log_file = tmp_path / 'napari_batch.log'

        def process(item):
            return item.upper()

        runner = BatchRunner()
        runner.run(
            process,
            ['a', 'b', 'c'],
            threaded=True,
            log_file=log_file,
            log_header={'Test': 'napari threading'},
        )

        qtbot.waitUntil(lambda: not runner.is_running, timeout=5000)

        assert log_file.exists()
        content = log_file.read_text()
        assert 'Test: napari threading' in content
        assert 'Completed' in content

    def test_run_napari_threaded_with_files(self, qtbot, tmp_path):
        """Test napari threaded execution with file discovery."""
        # Create test files
        for name in ['img1.tif', 'img2.tif', 'img3.tif']:
            (tmp_path / name).write_text(name)

        results = []

        def process(path):
            return path.stem

        runner = BatchRunner(
            on_item_complete=lambda r, ctx: results.append(r),
        )

        runner.run(process, tmp_path, patterns='*.tif', threaded=True)

        qtbot.waitUntil(lambda: not runner.is_running, timeout=5000)

        assert sorted(results) == ['img1', 'img2', 'img3']

    def test_run_napari_threaded_progress_updates(self, qtbot):
        """Test that progress updates work correctly for UI integration."""
        progress_values = []

        def process(item):
            return item

        def on_progress(result, ctx):
            # Simulate progress bar update
            progress_values.append(ctx.index + 1)

        runner = BatchRunner(on_item_complete=on_progress)
        runner.run(process, [1, 2, 3, 4, 5], threaded=True)

        qtbot.waitUntil(lambda: not runner.is_running, timeout=5000)

        # Progress should have been recorded for each item
        assert progress_values == [1, 2, 3, 4, 5]

    def test_runner_reusable_napari_threaded(self, qtbot):
        """Test runner can be reused for multiple napari threaded batches."""
        results = []
        batch_count = [0]

        def on_complete():
            batch_count[0] += 1

        runner = BatchRunner(
            on_item_complete=lambda r, ctx: results.append(r),
            on_complete=on_complete,
        )

        # First batch
        runner.run(lambda x: x * 2, [1, 2], threaded=True)
        qtbot.waitUntil(lambda: not runner.is_running, timeout=5000)

        assert sorted(results) == [2, 4]
        assert batch_count[0] == 1

        # Second batch
        runner.run(lambda x: x + 10, [1, 2], threaded=True)
        qtbot.waitUntil(lambda: not runner.is_running, timeout=5000)

        assert sorted(results) == [2, 4, 11, 12]
        assert batch_count[0] == 2

    def test_on_start_callback_threaded(self, qtbot):
        """Test on_start callback is called with total count in threaded mode."""
        start_total = []

        def process(item):
            return item

        runner = BatchRunner(on_start=lambda total: start_total.append(total))

        runner.run(process, [1, 2, 3, 4, 5], threaded=True)

        qtbot.waitUntil(lambda: not runner.is_running, timeout=5000)

        assert start_total == [5]

    def test_error_count_threaded(self, qtbot):
        """Test error_count tracks errors correctly in threaded mode."""

        def process(item):
            if item in [2, 4]:
                raise ValueError(f'Error on {item}')
            return item

        runner = BatchRunner()
        runner.run(process, [1, 2, 3, 4, 5], threaded=True)

        qtbot.waitUntil(lambda: not runner.is_running, timeout=5000)

        assert runner.error_count == 2
