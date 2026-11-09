#!/usr/bin/env python3
"""Validate NoX source ZIPs before extraction without writing to the checkout."""
from __future__ import annotations

import argparse
import stat
import sys
import zipfile
from pathlib import Path

MAX_ENTRY_SIZE_BYTES = 512 * 1024**2
MAX_TOTAL_SIZE_BYTES = 2 * 1024**3
MAX_ENTRY_COUNT = 100_000


def validate_archive(
    archive_path: str | Path,
    *,
    maximum_entry_size: int = MAX_ENTRY_SIZE_BYTES,
    maximum_total_size: int = MAX_TOTAL_SIZE_BYTES,
    maximum_entries: int = MAX_ENTRY_COUNT,
) -> None:
    """Raise ValueError if a ZIP is unsafe, structurally invalid, or oversized."""
    path = Path(archive_path)
    if not path.is_file():
        raise ValueError(f"Archive does not exist or is not a file: {path}")
    if maximum_entry_size < 0 or maximum_total_size < 0 or maximum_entries < 0:
        raise ValueError("Archive validation limits must be non-negative")

    seen: set[str] = set()
    try:
        with zipfile.ZipFile(path) as bundle:
            entries = bundle.infolist()
            if len(entries) > maximum_entries:
                raise ValueError(
                    f"Archive contains {len(entries)} entries; limit is {maximum_entries}"
                )
            total_size = 0
            for item in entries:
                name = item.filename
                if not name or "\\" in name or name.startswith("/"):
                    raise ValueError(f"Unsafe archive path: {name!r}")

                is_directory = item.is_dir()
                normalized = name[:-1] if is_directory else name
                parts = normalized.split("/")
                if (
                    not normalized
                    or any(part in ("", ".", "..") for part in parts)
                    or (parts and ":" in parts[0])
                ):
                    raise ValueError(f"Unsafe archive path: {name!r}")

                canonical_name = "/".join(parts)
                if canonical_name in seen:
                    raise ValueError(f"Duplicate archive path: {name!r}")
                seen.add(canonical_name)

                file_type = stat.S_IFMT(item.external_attr >> 16)
                if file_type == stat.S_IFLNK:
                    raise ValueError(
                        f"Symbolic links are not accepted in source archives: {name!r}"
                    )
                if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise ValueError(
                        f"Special filesystem entries are not accepted: {name!r}"
                    )

                if item.file_size > maximum_entry_size:
                    raise ValueError(
                        f"Archive member is too large: {name!r} ({item.file_size} bytes)"
                    )
                total_size += item.file_size
                if total_size > maximum_total_size:
                    raise ValueError(
                        f"Archive expands beyond the {maximum_total_size}-byte safety limit"
                    )

            # Consume member data only after every metadata/path/count/size constraint passed.
            bad_entry = bundle.testzip()
            if bad_entry is not None:
                raise ValueError(f"Corrupt ZIP member: {bad_entry}")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Archive could not be validated: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="ZIP file to validate")
    args = parser.parse_args()
    try:
        validate_archive(args.archive)
    except ValueError as error:
        print(f"Archive validation failed: {error}", file=sys.stderr)
        return 1
    print(f"Archive validation passed: {args.archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
