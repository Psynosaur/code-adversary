"""Diff parsing — unified diff to structured DiffFile/DiffHunk objects."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiffHunk:
    """A single hunk from a unified diff."""

    file_path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header: str
    lines: list[str] = field(default_factory=list)
    added_lines: list[tuple[int, str]] = field(default_factory=list)  # (line_no, text)
    removed_lines: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class DiffFile:
    """Parsed diff for a single file."""

    old_path: Optional[str]
    new_path: Optional[str]
    hunks: list[DiffHunk] = field(default_factory=list)
    is_new: bool = False
    is_deleted: bool = False
    is_renamed: bool = False

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or "<unknown>"

    @property
    def added_line_count(self) -> int:
        return sum(len(h.added_lines) for h in self.hunks)

    @property
    def removed_line_count(self) -> int:
        return sum(len(h.removed_lines) for h in self.hunks)


def parse_diff(diff_text: str) -> list[DiffFile]:
    """Parse a unified diff into structured DiffFile objects."""
    files: list[DiffFile] = []
    current_file: Optional[DiffFile] = None
    current_hunk: Optional[DiffHunk] = None
    new_line_no = 0

    for raw_line in diff_text.splitlines():
        line = raw_line

        # New file header
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                old_path = parts[2].removeprefix("a/")
                new_path = parts[3].removeprefix("b/")
                current_file = DiffFile(old_path=old_path, new_path=new_path)
                files.append(current_file)
                current_hunk = None
            continue

        if current_file is None:
            continue

        # File metadata
        if line.startswith("new file"):
            current_file.is_new = True
            continue
        if line.startswith("deleted file"):
            current_file.is_deleted = True
            continue
        if line.startswith("rename from") or line.startswith("rename to"):
            current_file.is_renamed = True
            continue
        if (
            line.startswith("index ")
            or line.startswith("---")
            or line.startswith("+++")
        ):
            continue

        # Hunk header
        hunk_match = re.match(
            r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(.*)", line
        )
        if hunk_match:
            old_start = int(hunk_match.group(1))
            old_count = int(hunk_match.group(2) or 1)
            new_start = int(hunk_match.group(3))
            new_count = int(hunk_match.group(4) or 1)
            header = hunk_match.group(5).strip()
            current_hunk = DiffHunk(
                file_path=current_file.path,
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                header=header,
            )
            current_file.hunks.append(current_hunk)
            new_line_no = new_start
            continue

        if current_hunk is None:
            continue

        # Diff content lines
        current_hunk.lines.append(line)
        if line.startswith("+"):
            current_hunk.added_lines.append((new_line_no, line[1:]))
            new_line_no += 1
        elif line.startswith("-"):
            current_hunk.removed_lines.append((new_line_no, line[1:]))
        else:
            # Context line
            new_line_no += 1

    return files


def diff_stats(files: list[DiffFile]) -> dict:
    """Generate summary statistics for parsed diff files."""
    total_added = sum(f.added_line_count for f in files)
    total_removed = sum(f.removed_line_count for f in files)
    return {
        "files_changed": len(files),
        "lines_added": total_added,
        "lines_removed": total_removed,
        "new_files": [f.path for f in files if f.is_new],
        "deleted_files": [f.path for f in files if f.is_deleted],
        "renamed_files": [f.path for f in files if f.is_renamed],
    }
