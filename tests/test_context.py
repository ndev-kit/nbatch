"""Tests for BatchContext dataclass."""

import pytest

from ndev_batch import BatchContext


class TestBatchContext:
    """Test suite for BatchContext."""

    def test_creation(self):
        """Test basic BatchContext creation."""
        ctx = BatchContext(index=0, total=10, item='test.tif')

        assert ctx.index == 0
        assert ctx.total == 10
        assert ctx.item == 'test.tif'

    def test_frozen(self):
        """Test that BatchContext is immutable."""
        ctx = BatchContext(index=0, total=10, item='test.tif')

        with pytest.raises(AttributeError):
            ctx.index = 5

    def test_progress_calculation(self):
        """Test progress property calculation."""
        # First item (index 0) -> 10% complete
        ctx = BatchContext(index=0, total=10, item=None)
        assert ctx.progress == 0.1

        # Middle item
        ctx = BatchContext(index=4, total=10, item=None)
        assert ctx.progress == 0.5

        # Last item (index 9) -> 100% complete
        ctx = BatchContext(index=9, total=10, item=None)
        assert ctx.progress == 1.0

    def test_progress_empty_batch(self):
        """Test progress with zero total (edge case)."""
        ctx = BatchContext(index=0, total=0, item=None)
        assert ctx.progress == 0.0

    def test_is_first(self):
        """Test is_first property."""
        first = BatchContext(index=0, total=10, item=None)
        assert first.is_first is True

        not_first = BatchContext(index=5, total=10, item=None)
        assert not_first.is_first is False

    def test_is_last(self):
        """Test is_last property."""
        last = BatchContext(index=9, total=10, item=None)
        assert last.is_last is True

        not_last = BatchContext(index=5, total=10, item=None)
        assert not_last.is_last is False

    def test_is_first_and_last_single_item(self):
        """Test single-item batch is both first and last."""
        ctx = BatchContext(index=0, total=1, item='only.tif')
        assert ctx.is_first is True
        assert ctx.is_last is True

    def test_str_representation(self):
        """Test string representation."""
        ctx = BatchContext(index=2, total=10, item='image.tif')
        assert str(ctx) == '[3/10] image.tif'

    def test_str_with_path_item(self):
        """Test string representation with Path item."""
        from pathlib import Path

        ctx = BatchContext(index=0, total=5, item=Path('/data/img.tif'))
        result = str(ctx)
        assert '[1/5]' in result
        assert 'img.tif' in result

    def test_equality(self):
        """Test dataclass equality."""
        ctx1 = BatchContext(index=0, total=10, item='a.tif')
        ctx2 = BatchContext(index=0, total=10, item='a.tif')
        ctx3 = BatchContext(index=1, total=10, item='a.tif')

        assert ctx1 == ctx2
        assert ctx1 != ctx3

    def test_hash(self):
        """Test that BatchContext is hashable (frozen dataclass)."""
        ctx = BatchContext(index=0, total=10, item='test.tif')
        # Should be able to use as dict key or in set
        d = {ctx: 'value'}
        assert d[ctx] == 'value'
