"""Diff chunking — split large diffs into LLM-friendly chunks."""

from __future__ import annotations

import logging

from reviewer.config import DIFF_CHUNK_MAX_CHARS
from reviewer.diff_parser import DiffFile, DiffHunk

logger = logging.getLogger(__name__)


def reconstruct_diff(files: list[DiffFile]) -> str:
    """Reconstruct a unified diff string from parsed DiffFile objects."""
    parts: list[str] = []
    for f in files:
        parts.append(f"diff --git a/{f.old_path or f.path} b/{f.path}")
        if f.is_new:
            parts.append("new file mode 100644")
        elif f.is_deleted:
            parts.append("deleted file mode 100644")
        parts.append(
            f"--- {'a/' + (f.old_path or f.path) if not f.is_new else '/dev/null'}"
        )
        parts.append(f"+++ {'b/' + f.path if not f.is_deleted else '/dev/null'}")
        for hunk in f.hunks:
            parts.append(
                f"@@ -{hunk.old_start},{hunk.old_count} "
                f"+{hunk.new_start},{hunk.new_count} @@ {hunk.header}"
            )
            for line in hunk.lines:
                parts.append(line)
    return "\n".join(parts)


def chunk_files(files: list[DiffFile]) -> list[list[DiffFile]]:
    """Split parsed diff files into chunks that fit within token limits.

    Each chunk contains one or more files. A single file that exceeds the
    limit is split into multiple synthetic DiffFile objects, each containing
    a subset of hunks that fit under the limit.
    """
    if not files:
        return []

    chunks: list[list[DiffFile]] = []
    current_chunk: list[DiffFile] = []
    current_chars = 0

    for f in files:
        # Estimate char count for this file's diff
        file_chars = (
            sum(len(line) for hunk in f.hunks for line in hunk.lines) + 200
        )  # overhead for headers

        if file_chars > DIFF_CHUNK_MAX_CHARS:
            # Oversized file — flush current chunk, then split this file by hunks
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_chars = 0

            chunks.extend(_split_file_into_chunks(f))
            continue

        if current_chunk and (current_chars + file_chars) > DIFF_CHUNK_MAX_CHARS:
            # Start a new chunk
            chunks.append(current_chunk)
            current_chunk = [f]
            current_chars = file_chars
        else:
            current_chunk.append(f)
            current_chars += file_chars

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _split_file_into_chunks(f: DiffFile) -> list[list[DiffFile]]:
    """Split a single oversized DiffFile into multiple chunks by grouping hunks.

    If the file has only one hunk (e.g., a large new file), that hunk is
    split into synthetic sub-hunks by line count.

    Returns a list of chunks, where each chunk contains a single synthetic
    DiffFile with a subset of the original hunks.
    """
    hunks = f.hunks

    # If single oversized hunk, split it into sub-hunks first
    if len(hunks) == 1 and sum(len(l) for l in hunks[0].lines) > DIFF_CHUNK_MAX_CHARS:
        hunks = _split_hunk(hunks[0])

    chunks: list[list[DiffFile]] = []
    current_hunks: list[DiffHunk] = []
    current_chars = 0

    for hunk in hunks:
        hunk_chars = sum(len(line) for line in hunk.lines) + 100

        if current_hunks and (current_chars + hunk_chars) > DIFF_CHUNK_MAX_CHARS:
            # Emit a chunk with hunks so far
            part = DiffFile(
                old_path=f.old_path,
                new_path=f.new_path,
                hunks=current_hunks,
                is_new=f.is_new,
                is_deleted=f.is_deleted,
                is_renamed=f.is_renamed,
            )
            chunks.append([part])
            current_hunks = [hunk]
            current_chars = hunk_chars
        else:
            current_hunks.append(hunk)
            current_chars += hunk_chars

    if current_hunks:
        part = DiffFile(
            old_path=f.old_path,
            new_path=f.new_path,
            hunks=current_hunks,
            is_new=f.is_new,
            is_deleted=f.is_deleted,
            is_renamed=f.is_renamed,
        )
        chunks.append([part])

    return chunks


def _split_hunk(hunk: DiffHunk) -> list[DiffHunk]:
    """Split a single oversized hunk into smaller sub-hunks by line count.

    Tries to split at blank-line boundaries (between functions/classes)
    for better context. Falls back to splitting at DIFF_CHUNK_MAX_CHARS.
    """
    # Target ~DIFF_CHUNK_MAX_CHARS per sub-hunk
    lines = hunk.lines
    target_chars = DIFF_CHUNK_MAX_CHARS

    sub_hunks: list[DiffHunk] = []
    start_idx = 0
    current_chars = 0
    last_blank_idx: int | None = None  # Track blank lines for clean splits
    new_line_no = hunk.new_start

    for i, line in enumerate(lines):
        line_len = len(line)
        current_chars += line_len

        # Track blank added lines (good split points between functions)
        if line in ("+", "+ "):
            last_blank_idx = i

        if current_chars >= target_chars and i > start_idx:
            # Prefer splitting at a blank line if one exists in the last 20% of the chunk
            split_at = i
            threshold = start_idx + int((i - start_idx) * 0.8)
            if last_blank_idx is not None and last_blank_idx >= threshold:
                split_at = last_blank_idx + 1

            sub_lines = lines[start_idx:split_at]
            added = [
                (new_line_no + j, l[1:])
                for j, l in enumerate(sub_lines)
                if l.startswith("+")
            ]
            removed = [(new_line_no, l[1:]) for l in sub_lines if l.startswith("-")]

            sub_hunks.append(
                DiffHunk(
                    file_path=hunk.file_path,
                    old_start=hunk.old_start,
                    old_count=0,
                    new_start=new_line_no,
                    new_count=len(added),
                    header=f"{hunk.header} (part {len(sub_hunks) + 1})",
                    lines=sub_lines,
                    added_lines=added,
                    removed_lines=removed,
                )
            )

            # Advance
            new_line_no += sum(
                1 for l in sub_lines if l.startswith("+") or not l.startswith("-")
            )
            start_idx = split_at
            current_chars = 0
            last_blank_idx = None

    # Remaining lines
    if start_idx < len(lines):
        sub_lines = lines[start_idx:]
        added = [
            (new_line_no + j, l[1:])
            for j, l in enumerate(sub_lines)
            if l.startswith("+")
        ]
        removed = [(new_line_no, l[1:]) for l in sub_lines if l.startswith("-")]

        sub_hunks.append(
            DiffHunk(
                file_path=hunk.file_path,
                old_start=hunk.old_start,
                old_count=0,
                new_start=new_line_no,
                new_count=len(added),
                header=f"{hunk.header} (part {len(sub_hunks) + 1})",
                lines=sub_lines,
                added_lines=added,
                removed_lines=removed,
            )
        )

    return sub_hunks
