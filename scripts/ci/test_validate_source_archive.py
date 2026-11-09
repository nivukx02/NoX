#!/usr/bin/env python3
"""Regression tests for the NoX ZIP validator, using only the Python standard library."""
from __future__ import annotations

import stat
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

from validate_source_archive import validate_archive


class ValidateSourceArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def archive(self, name: str, entries: list[tuple[str, bytes, int | None]]) -> Path:
        path = self.directory / name
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as bundle:
            for entry_name, data, mode in entries:
                info = zipfile.ZipInfo(entry_name)
                if mode is not None:
                    info.external_attr = mode << 16
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    bundle.writestr(info, data)
        return path

    def test_valid_project_archive(self) -> None:
        archive = self.archive("valid.zip", [
            ("NoX-main/", b"", stat.S_IFDIR | 0o755),
            ("NoX-main/settings.gradle.kts", b"rootProject.name = 'NoX'", stat.S_IFREG | 0o644),
            ("NoX-main/app/build.gradle.kts", b"plugins {}", stat.S_IFREG | 0o644),
        ])
        validate_archive(archive)

    def test_traversal_absolute_drive_and_noncanonical_paths(self) -> None:
        unsafe = (
            "../escape.txt", "NoX-main/../../escape.txt", "/absolute.txt",
            "C:/absolute.txt", "NoX-main\\escape.txt", "NoX-main/./settings.gradle.kts",
            "NoX-main//settings.gradle.kts",
        )
        for index, name in enumerate(unsafe):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_archive(self.archive(f"unsafe-{index}.zip", [(name, b"x", None)]))

    def test_symlinks_and_special_files(self) -> None:
        for index, mode in enumerate((stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o600)):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    validate_archive(self.archive(f"special-{index}.zip", [
                        ("NoX-main/item", b"target", mode),
                    ]))

    def test_duplicate_paths(self) -> None:
        archive = self.archive("duplicate.zip", [
            ("NoX-main/settings.gradle.kts", b"first", None),
            ("NoX-main/settings.gradle.kts", b"second", None),
        ])
        with self.assertRaisesRegex(ValueError, "Duplicate archive path"):
            validate_archive(archive)

    def test_per_entry_and_total_expansion_limits(self) -> None:
        archive = self.archive("large.zip", [
            ("NoX-main/a.kt", b"12345", None),
            ("NoX-main/b.kt", b"67890", None),
        ])
        with self.assertRaisesRegex(ValueError, "member is too large"):
            validate_archive(archive, maximum_entry_size=4, maximum_total_size=100)
        with self.assertRaisesRegex(ValueError, "expands beyond"):
            validate_archive(archive, maximum_entry_size=10, maximum_total_size=8)

    def test_entry_count_limit(self) -> None:
        archive = self.archive("many.zip", [
            ("NoX-main/a.kt", b"a", None), ("NoX-main/b.kt", b"b", None),
        ])
        with self.assertRaisesRegex(ValueError, "contains 2 entries"):
            validate_archive(archive, maximum_entries=1)

    def test_crc_corruption(self) -> None:
        archive = self.archive("crc.zip", [("NoX-main/a.kt", b"known-payload", None)])
        raw = bytearray(archive.read_bytes())
        offset = raw.find(b"known-payload")
        self.assertGreaterEqual(offset, 0)
        raw[offset] ^= 1
        archive.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "Corrupt ZIP member"):
            validate_archive(archive)

    def test_missing_archive(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            validate_archive(self.directory / "missing.zip")


if __name__ == "__main__":
    unittest.main(verbosity=2)
