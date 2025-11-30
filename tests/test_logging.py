"""Tests for logging utilities."""

import logging

import pytest

from nbatch import BatchContext, BatchLogger, batch_logger


class TestBatchLogger:
    """Test suite for BatchLogger class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        logger = logging.getLogger('test_batch_logger')
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        return logger

    @pytest.fixture
    def blog(self, mock_logger):
        """Create a BatchLogger instance."""
        return BatchLogger(mock_logger)

    @pytest.fixture
    def sample_ctx(self):
        """Create a sample BatchContext."""
        return BatchContext(index=2, total=10, item='image.tif')

    def test_call_logs_info(self, blog, sample_ctx, caplog):
        """Test that __call__ logs at INFO level by default."""
        with caplog.at_level(logging.INFO):
            blog(sample_ctx, 'Processing complete')

        assert 'Processing complete' in caplog.text
        assert '[3/10]' in caplog.text

    def test_info_method(self, blog, sample_ctx, caplog):
        """Test info method."""
        with caplog.at_level(logging.INFO):
            blog.info(sample_ctx, 'Info message')

        assert 'INFO' in caplog.text
        assert 'Info message' in caplog.text

    def test_warning_method(self, blog, sample_ctx, caplog):
        """Test warning method."""
        with caplog.at_level(logging.WARNING):
            blog.warning(sample_ctx, 'Warning message')

        assert 'WARNING' in caplog.text
        assert 'Warning message' in caplog.text

    def test_error_method(self, blog, sample_ctx, caplog):
        """Test error method."""
        with caplog.at_level(logging.ERROR):
            blog.error(sample_ctx, 'Error message')

        assert 'ERROR' in caplog.text
        assert 'Error message' in caplog.text

    def test_debug_method(self, blog, sample_ctx, caplog):
        """Test debug method."""
        with caplog.at_level(logging.DEBUG):
            blog.debug(sample_ctx, 'Debug message')

        assert 'DEBUG' in caplog.text
        assert 'Debug message' in caplog.text

    def test_exception_method(self, blog, sample_ctx, caplog):
        """Test exception method includes traceback."""
        with caplog.at_level(logging.ERROR):
            try:
                raise ValueError('Test error')
            except ValueError:
                blog.exception(sample_ctx, 'Exception occurred')

        assert 'Exception occurred' in caplog.text
        assert '[3/10]' in caplog.text


class TestBatchLoggerContextManager:
    """Test suite for batch_logger context manager."""

    def test_yields_batch_logger(self):
        """Test that context manager yields BatchLogger."""
        with batch_logger(console=False) as log:
            assert isinstance(log, BatchLogger)

    def test_console_output_by_default(self, capsys):
        """Test that console output is enabled by default."""
        ctx = BatchContext(index=0, total=5, item='test.tif')

        with batch_logger() as log:
            log(ctx, 'Console message')

        captured = capsys.readouterr()
        assert 'Console message' in captured.err
        assert '[1/5]' in captured.err

    def test_console_disabled(self, caplog):
        """Test that console output can be disabled."""
        ctx = BatchContext(index=0, total=5, item='test.tif')

        with caplog.at_level(logging.INFO), batch_logger(console=False) as log:
            log(ctx, 'Should not appear')

        assert 'Should not appear' not in caplog.text

    def test_creates_log_file_when_specified(self, tmp_path):
        """Test that context manager creates log file when specified."""
        log_file = tmp_path / 'batch.log'

        with batch_logger(log_file=log_file, console=False):
            pass

        assert log_file.exists()

    def test_no_file_created_by_default(self, tmp_path):
        """Test that no file is created when log_file not specified."""
        with batch_logger(console=False):
            pass

        # No files should be created in tmp_path
        assert list(tmp_path.iterdir()) == []

    def test_logs_messages_to_file(self, tmp_path):
        """Test that messages are logged to file."""
        log_file = tmp_path / 'batch.log'
        ctx = BatchContext(index=0, total=5, item='test.tif')

        with batch_logger(log_file=log_file, console=False) as log:
            log(ctx, 'Processing item')

        content = log_file.read_text()
        assert 'Processing item' in content
        assert '[1/5]' in content

    def test_header_written(self, tmp_path):
        """Test that header is written to log file."""
        log_file = tmp_path / 'batch.log'
        header = {'Model': 'classifier.clf', 'Files': 100, 'Output': '/data'}

        with batch_logger(log_file=log_file, header=header, console=False):
            pass

        content = log_file.read_text()
        assert 'Model: classifier.clf' in content
        assert 'Files: 100' in content
        assert 'Output: /data' in content
        assert 'Batch processing started' in content

    def test_footer_written_when_header_provided(self, tmp_path):
        """Test that footer is written to log file when header is provided."""
        log_file = tmp_path / 'batch.log'
        header = {'Files': 10}

        with batch_logger(log_file=log_file, header=header, console=False):
            pass

        content = log_file.read_text()
        assert 'Batch processing completed' in content

    def test_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created."""
        log_file = tmp_path / 'nested' / 'dir' / 'batch.log'

        with batch_logger(log_file=log_file, console=False):
            pass

        assert log_file.exists()

    def test_append_mode_by_default(self, tmp_path):
        """Test that append mode is the default."""
        log_file = tmp_path / 'batch.log'
        ctx = BatchContext(index=0, total=1, item='test')

        # First run
        with batch_logger(log_file=log_file, console=False) as log:
            log(ctx, 'First message')

        # Second run should append
        with batch_logger(log_file=log_file, console=False) as log:
            log(ctx, 'Second message')

        content = log_file.read_text()
        assert 'First message' in content
        assert 'Second message' in content

    def test_write_mode_overwrites(self, tmp_path):
        """Test that write mode overwrites the file."""
        log_file = tmp_path / 'batch.log'
        ctx = BatchContext(index=0, total=1, item='test')

        # First run
        with batch_logger(
            log_file=log_file, file_mode='w', console=False
        ) as log:
            log(ctx, 'First message')

        # Second run should overwrite
        with batch_logger(
            log_file=log_file, file_mode='w', console=False
        ) as log:
            log(ctx, 'Second message')

        content = log_file.read_text()
        assert 'First message' not in content
        assert 'Second message' in content

    def test_cleanup_on_exception(self, tmp_path):
        """Test that cleanup happens even on exception."""
        log_file = tmp_path / 'batch.log'
        header = {'Task': 'test'}

        with (
            pytest.raises(RuntimeError),
            batch_logger(
                log_file=log_file, header=header, console=False
            ) as log,
        ):
            ctx = BatchContext(index=0, total=1, item='test')
            log(ctx, 'Before error')
            raise RuntimeError('Test error')

        # File should still have footer because header was provided
        content = log_file.read_text()
        assert 'Before error' in content
        assert 'Batch processing completed' in content

    def test_custom_level(self, tmp_path):
        """Test custom logging level."""
        log_file = tmp_path / 'batch.log'
        ctx = BatchContext(index=0, total=1, item='test')

        with batch_logger(
            log_file=log_file, level=logging.WARNING, console=False
        ) as log:
            log.info(ctx, 'Info message')
            log.warning(ctx, 'Warning message')

        content = log_file.read_text()
        # INFO should not appear (below threshold)
        # But WARNING should
        assert 'Warning message' in content

    def test_both_console_and_file(self, tmp_path, capsys):
        """Test that both console and file can be enabled."""
        log_file = tmp_path / 'batch.log'
        ctx = BatchContext(index=0, total=1, item='test')

        with batch_logger(log_file=log_file, console=True) as log:
            log(ctx, 'Both outputs')

        # Should be in both console (stderr) and file
        captured = capsys.readouterr()
        assert 'Both outputs' in captured.err

        content = log_file.read_text()
        assert 'Both outputs' in content
