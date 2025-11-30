"""Tests for file discovery utilities."""

from pathlib import Path

import pytest

from nbatch import discover_files, is_batch_input


class TestDiscoverFiles:
    """Test suite for discover_files function."""

    def test_single_file(self, tmp_path):
        """Test discovery of single file."""
        test_file = tmp_path / 'image.tif'
        test_file.touch()

        result = discover_files(test_file)

        assert result == [test_file]

    def test_single_file_string(self, tmp_path):
        """Test discovery with string path."""
        test_file = tmp_path / 'image.tif'
        test_file.touch()

        result = discover_files(str(test_file))

        assert result == [test_file]

    def test_directory_all_files(self, tmp_path):
        """Test discovery from directory with default pattern."""
        (tmp_path / 'a.tif').touch()
        (tmp_path / 'b.tif').touch()
        (tmp_path / 'c.png').touch()

        result = discover_files(tmp_path)

        assert len(result) == 3
        assert all(isinstance(p, Path) for p in result)
        # Results should be sorted
        assert result == sorted(result)

    def test_directory_with_pattern(self, tmp_path):
        """Test discovery with specific glob pattern."""
        (tmp_path / 'a.tif').touch()
        (tmp_path / 'b.tif').touch()
        (tmp_path / 'c.png').touch()

        result = discover_files(tmp_path, patterns='*.tif')

        assert len(result) == 2
        assert all(p.suffix == '.tif' for p in result)

    def test_directory_multiple_patterns(self, tmp_path):
        """Test discovery with multiple glob patterns."""
        (tmp_path / 'a.tif').touch()
        (tmp_path / 'b.tiff').touch()
        (tmp_path / 'c.png').touch()
        (tmp_path / 'd.jpg').touch()

        result = discover_files(
            tmp_path, patterns=['*.tif', '*.tiff', '*.png']
        )

        assert len(result) == 3
        assert not any(p.suffix == '.jpg' for p in result)

    def test_recursive_discovery(self, tmp_path):
        """Test recursive directory discovery."""
        # Create nested structure
        (tmp_path / 'a.tif').touch()
        subdir = tmp_path / 'subdir'
        subdir.mkdir()
        (subdir / 'b.tif').touch()
        subsubdir = subdir / 'deep'
        subsubdir.mkdir()
        (subsubdir / 'c.tif').touch()

        result = discover_files(tmp_path, patterns='*.tif', recursive=True)

        assert len(result) == 3

    def test_non_recursive_discovery(self, tmp_path):
        """Test non-recursive directory discovery."""
        (tmp_path / 'a.tif').touch()
        subdir = tmp_path / 'subdir'
        subdir.mkdir()
        (subdir / 'b.tif').touch()

        result = discover_files(tmp_path, patterns='*.tif', recursive=False)

        assert len(result) == 1
        assert result[0].name == 'a.tif'

    def test_iterable_of_paths(self, tmp_path):
        """Test discovery from iterable of paths."""
        files = [tmp_path / 'a.tif', tmp_path / 'b.tif']
        for f in files:
            f.touch()

        result = discover_files(files)

        assert len(result) == 2
        assert result == sorted(files)

    def test_iterable_of_strings(self, tmp_path):
        """Test discovery from iterable of string paths."""
        file1 = tmp_path / 'a.tif'
        file2 = tmp_path / 'b.tif'
        file1.touch()
        file2.touch()

        result = discover_files([str(file1), str(file2)])

        assert len(result) == 2
        assert all(isinstance(p, Path) for p in result)

    def test_nonexistent_file_raises(self, tmp_path):
        """Test that nonexistent file raises FileNotFoundError."""
        fake_file = tmp_path / 'nonexistent.tif'

        with pytest.raises(FileNotFoundError):
            discover_files(fake_file)

    def test_nonexistent_in_iterable_raises(self, tmp_path):
        """Test that nonexistent file in iterable raises."""
        real_file = tmp_path / 'real.tif'
        real_file.touch()
        fake_file = tmp_path / 'fake.tif'

        with pytest.raises(FileNotFoundError):
            discover_files([real_file, fake_file])

    def test_empty_iterable_raises(self):
        """Test that empty iterable raises ValueError."""
        with pytest.raises(ValueError, match='empty'):
            discover_files([])

    def test_no_matching_files_raises(self, tmp_path):
        """Test that no matching files raises ValueError."""
        (tmp_path / 'file.txt').touch()

        with pytest.raises(ValueError, match='No files found'):
            discover_files(tmp_path, patterns='*.tif')

    def test_excludes_directories(self, tmp_path):
        """Test that directories are not included in results."""
        (tmp_path / 'file.tif').touch()
        subdir = tmp_path / 'subdir.tif'  # Directory with .tif name
        subdir.mkdir()

        result = discover_files(tmp_path, patterns='*.tif', recursive=False)

        assert len(result) == 1
        assert result[0].is_file()

    def test_natural_sorting(self, tmp_path):
        """Test that files are naturally sorted (like file explorers).

        Natural sort: image1, image2, image10
        Alphabetical: image1, image10, image2
        """
        # Create files that would sort differently with natural vs alphabetical
        names = ['image1.tif', 'image10.tif', 'image2.tif', 'image20.tif']
        for name in names:
            (tmp_path / name).touch()

        result = discover_files(tmp_path, patterns='*.tif')
        result_names = [p.name for p in result]

        # With natsort: should be 1, 2, 10, 20
        # Without natsort: would be 1, 10, 2, 20
        # We check that 2 comes before 10 (natural sort behavior)
        assert result_names.index('image2.tif') < result_names.index(
            'image10.tif'
        )

        # The important thing is the function works either way
        assert len(result) == 4


class TestIsBatchInput:
    """Test suite for is_batch_input helper."""

    def test_list_is_batch(self):
        """Test that list is considered batch input."""
        assert is_batch_input([1, 2, 3]) is True
        assert is_batch_input([]) is True  # Empty list still batch type

    def test_tuple_is_batch(self):
        """Test that tuple is considered batch input."""
        assert is_batch_input((1, 2, 3)) is True

    def test_directory_is_batch(self, tmp_path):
        """Test that directory Path is considered batch input."""
        assert is_batch_input(tmp_path) is True

    def test_file_is_not_batch(self, tmp_path):
        """Test that file Path is not batch input."""
        test_file = tmp_path / 'test.tif'
        test_file.touch()

        assert is_batch_input(test_file) is False

    def test_string_is_not_batch(self):
        """Test that string is not batch input (even if path-like)."""
        assert is_batch_input('/path/to/dir') is False
        assert is_batch_input('file.tif') is False

    def test_other_types_not_batch(self):
        """Test that other types are not batch input."""
        assert is_batch_input(42) is False
        assert is_batch_input(None) is False
        assert is_batch_input({'a': 1}) is False
