"""Core review logic — diff parsing, pattern matching, finding generation."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from reviewer.config import (
    ANTI_PATTERNS,
    CHALLENGE_ANGLES,
    DEFAULT_FOCUS_AREAS,
    DIFF_CHUNK_MAX_CHARS,
    PERSONA,
    STRUCTURAL_CHECKS,
)
from reviewer.models import (
    Category,
    ReviewFinding,
    ReviewReport,
    Severity,
)
from reviewer import llm
from reviewer.llm import ProgressCallback
from reviewer.prompts import (
    REVIEW_DIFF_SYSTEM,
    REVIEW_PATTERN_SYSTEM,
    CHALLENGE_DECISION_SYSTEM,
    build_review_diff_message,
    build_review_pattern_message,
    build_challenge_decision_message,
)

logger = logging.getLogger(__name__)


# ── Diff Parsing ─────────────────────────────────────────────────────────────


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


# ── Diff Chunking ────────────────────────────────────────────────────────────

# Use the configured chunk size limit
_MAX_CHUNK_CHARS = DIFF_CHUNK_MAX_CHARS


def _reconstruct_diff(files: list[DiffFile]) -> str:
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


def _chunk_files(files: list[DiffFile]) -> list[list[DiffFile]]:
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

        if file_chars > _MAX_CHUNK_CHARS:
            # Oversized file — flush current chunk, then split this file by hunks
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_chars = 0

            chunks.extend(_split_file_into_chunks(f))
            continue

        if current_chunk and (current_chars + file_chars) > _MAX_CHUNK_CHARS:
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
    if len(hunks) == 1 and sum(len(l) for l in hunks[0].lines) > _MAX_CHUNK_CHARS:
        hunks = _split_hunk(hunks[0])

    chunks: list[list[DiffFile]] = []
    current_hunks: list[DiffHunk] = []
    current_chars = 0

    for hunk in hunks:
        hunk_chars = sum(len(line) for line in hunk.lines) + 100

        if current_hunks and (current_chars + hunk_chars) > _MAX_CHUNK_CHARS:
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
    for better context. Falls back to splitting at _MAX_CHUNK_CHARS.
    """
    # Target ~_MAX_CHUNK_CHARS per sub-hunk
    lines = hunk.lines
    target_chars = _MAX_CHUNK_CHARS

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


# ── Pattern Matching ─────────────────────────────────────────────────────────


def _match_anti_patterns(files: list[DiffFile]) -> list[ReviewFinding]:
    """Run regex-based anti-pattern detection on added lines only."""
    findings: list[ReviewFinding] = []

    for diff_file in files:
        for hunk in diff_file.hunks:
            for line_no, line_text in hunk.added_lines:
                for name, pattern_def in ANTI_PATTERNS.items():
                    if re.search(pattern_def["pattern"], line_text):
                        # Skip assert pattern for test files
                        if name == "assert_in_production" and _is_test_file(
                            diff_file.path
                        ):
                            continue
                        # Skip print statements in test files
                        if name == "print_statement" and _is_test_file(diff_file.path):
                            continue

                        findings.append(
                            ReviewFinding(
                                severity=Severity(pattern_def["severity"]),
                                category=Category(pattern_def["category"]),
                                title=pattern_def["title"],
                                description=pattern_def["description"],
                                file_path=diff_file.path,
                                line_range=str(line_no),
                                code_snippet=line_text.strip(),
                            )
                        )

    return findings


def _is_test_file(path: str) -> bool:
    """Check if a file path looks like a test file."""
    path_lower = path.lower()
    return (
        "test" in path_lower
        or "spec" in path_lower
        or path_lower.startswith("tests/")
        or "/tests/" in path_lower
    )


# ── Structural Analysis ─────────────────────────────────────────────────────


def _check_function_length(files: list[DiffFile]) -> list[ReviewFinding]:
    """Flag functions that appear to be very long based on diff context."""
    findings: list[ReviewFinding] = []

    for diff_file in files:
        if diff_file.is_deleted:
            continue

        # Count added lines per hunk — if a single hunk adds 50+ lines
        # that all look like they're in one function, flag it
        for hunk in diff_file.hunks:
            if len(hunk.added_lines) >= 50:
                # Check if there's a function definition near the start
                func_name = None
                for _, line_text in hunk.added_lines[:5]:
                    func_match = re.match(
                        r"\s*(?:def|function|async\s+(?:def|function))\s+(\w+)",
                        line_text,
                    )
                    if func_match:
                        func_name = func_match.group(1)
                        break

                if func_name:
                    findings.append(
                        ReviewFinding(
                            severity=Severity.WARNING,
                            category=Category.COMPLEXITY,
                            title=f"Long function: {func_name}()",
                            description=(
                                f"Function `{func_name}` appears to span {len(hunk.added_lines)}+ "
                                f"lines. Consider breaking it into smaller, focused functions."
                            ),
                            file_path=diff_file.path,
                            line_range=f"{hunk.new_start}-{hunk.new_start + len(hunk.added_lines)}",
                        )
                    )

    return findings


def _check_large_files(files: list[DiffFile]) -> list[ReviewFinding]:
    """Flag new files that are very large."""
    findings: list[ReviewFinding] = []

    for diff_file in files:
        if diff_file.is_new and diff_file.added_line_count > 500:
            findings.append(
                ReviewFinding(
                    severity=Severity.WARNING,
                    category=Category.ARCHITECTURE,
                    title=f"Large new file: {diff_file.path}",
                    description=(
                        f"New file with {diff_file.added_line_count} lines. "
                        f"Consider splitting into multiple modules with clear separation of concerns."
                    ),
                    file_path=diff_file.path,
                )
            )

    return findings


def _check_error_handling(files: list[DiffFile]) -> list[ReviewFinding]:
    """Check for I/O operations without error handling in added code."""
    findings: list[ReviewFinding] = []
    io_patterns = [
        (r"\bopen\s*\(", "file open"),
        (r"requests\.\w+\s*\(", "HTTP request"),
        (r"urllib", "URL operation"),
        (r"\.connect\s*\(", "connection"),
        (r"subprocess\.\w+\s*\(", "subprocess call"),
    ]

    for diff_file in files:
        for hunk in diff_file.hunks:
            # Collect all added lines to check context
            added_texts = [text for _, text in hunk.added_lines]
            joined = "\n".join(added_texts)

            for pattern, op_name in io_patterns:
                matches = list(re.finditer(pattern, joined))
                for match in matches:
                    # Check if there's a try/except nearby
                    # (simple heuristic: within 5 lines before or after)
                    match_pos = match.start()
                    context_start = max(0, match_pos - 200)
                    context_end = min(len(joined), match_pos + 200)
                    context = joined[context_start:context_end]

                    if "try" not in context and "except" not in context:
                        # Find the line number
                        line_idx = joined[:match_pos].count("\n")
                        if line_idx < len(hunk.added_lines):
                            line_no, line_text = hunk.added_lines[line_idx]
                            findings.append(
                                ReviewFinding(
                                    severity=Severity.WARNING,
                                    category=Category.ERROR_HANDLING,
                                    title=f"Unguarded {op_name}",
                                    description=(
                                        f"A {op_name} operation without visible error handling. "
                                        f"I/O operations can fail — wrap in try/except or ensure "
                                        f"the caller handles exceptions."
                                    ),
                                    file_path=diff_file.path,
                                    line_range=str(line_no),
                                    code_snippet=line_text.strip(),
                                )
                            )

    return findings


def _check_missing_type_hints(files: list[DiffFile]) -> list[ReviewFinding]:
    """Flag Python function definitions without type hints."""
    findings: list[ReviewFinding] = []

    for diff_file in files:
        if not diff_file.path.endswith(".py"):
            continue
        if _is_test_file(diff_file.path):
            continue

        for hunk in diff_file.hunks:
            for line_no, line_text in hunk.added_lines:
                # Match def without return type annotation
                func_match = re.match(
                    r"\s*(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)\s*:", line_text
                )
                if func_match:
                    func_name = func_match.group(1)
                    params = func_match.group(2)

                    # Skip dunder methods and private helpers
                    if func_name.startswith("__") and func_name.endswith("__"):
                        continue

                    # Check for return type annotation
                    if "->" not in line_text:
                        findings.append(
                            ReviewFinding(
                                severity=Severity.SUGGESTION,
                                category=Category.TYPE_SAFETY,
                                title=f"Missing return type: {func_name}()",
                                description=(
                                    f"Function `{func_name}` lacks a return type annotation. "
                                    f"Add `-> ReturnType` for better IDE support and documentation."
                                ),
                                file_path=diff_file.path,
                                line_range=str(line_no),
                                code_snippet=line_text.strip(),
                            )
                        )

    return findings


def _check_duplicate_calls(files: list[DiffFile]) -> list[ReviewFinding]:
    """Detect repeated function calls within the same hunk (potential redundancy)."""
    findings: list[ReviewFinding] = []
    # Match function/method calls: word( or word.word(
    call_pattern = re.compile(r"(\b\w+(?:\.\w+)*\s*\([^)]*\))")

    for diff_file in files:
        if diff_file.is_deleted:
            continue

        for hunk in diff_file.hunks:
            calls: dict[str, list[int]] = {}
            for line_no, line_text in hunk.added_lines:
                # Normalize whitespace for matching
                for match in call_pattern.finditer(line_text):
                    call_str = re.sub(r"\s+", "", match.group(1))
                    # Skip trivial calls (len, str, int, print, etc.)
                    func_name = call_str.split("(")[0].split(".")[-1]
                    if func_name in (
                        "len",
                        "str",
                        "int",
                        "float",
                        "bool",
                        "list",
                        "dict",
                        "set",
                        "tuple",
                        "print",
                        "type",
                        "range",
                        "enumerate",
                        "isinstance",
                        "hasattr",
                        "getattr",
                        "setattr",
                        "super",
                        "append",
                        "extend",
                        "get",
                        "items",
                        "keys",
                        "values",
                        "strip",
                        "split",
                        "join",
                        "format",
                        "replace",
                        "lower",
                        "upper",
                        "startswith",
                        "endswith",
                    ):
                        continue
                    calls.setdefault(call_str, []).append(line_no)

            for call_str, line_nos in calls.items():
                if len(line_nos) >= 2:
                    func_name = call_str.split("(")[0]
                    findings.append(
                        ReviewFinding(
                            severity=Severity.WARNING,
                            category=Category.PERFORMANCE,
                            title=f"Repeated call: {func_name}()",
                            description=(
                                f"`{func_name}()` called {len(line_nos)} times with the same "
                                f"arguments in the same block (lines {', '.join(str(n) for n in line_nos)}). "
                                f"If the result doesn't change, cache it in a local variable."
                            ),
                            file_path=diff_file.path,
                            line_range=f"{line_nos[0]}-{line_nos[-1]}",
                            code_snippet=call_str,
                        )
                    )

    return findings


def _check_input_validation(files: list[DiffFile]) -> list[ReviewFinding]:
    """Flag public functions that accept parameters without any visible validation."""
    findings: list[ReviewFinding] = []

    for diff_file in files:
        if not diff_file.path.endswith(".py"):
            continue
        if diff_file.is_deleted or _is_test_file(diff_file.path):
            continue

        for hunk in diff_file.hunks:
            added_texts = [text for _, text in hunk.added_lines]
            joined = "\n".join(added_texts)

            for line_no, line_text in hunk.added_lines:
                func_match = re.match(
                    r"\s*(?:async\s+)?def\s+(\w+)\s*\(([^)]+)\)", line_text
                )
                if not func_match:
                    continue

                func_name = func_match.group(1)
                params_str = func_match.group(2)

                # Skip private, dunder, and self-only
                if func_name.startswith("_"):
                    continue
                params = [
                    p.strip().split(":")[0].split("=")[0].strip()
                    for p in params_str.split(",")
                ]
                params = [p for p in params if p not in ("self", "cls", "")]
                if not params:
                    continue

                # Look at the next ~15 lines for any validation
                func_start_idx = next(
                    (i for i, (ln, _) in enumerate(hunk.added_lines) if ln == line_no),
                    None,
                )
                if func_start_idx is None:
                    continue

                body_lines = [
                    text
                    for _, text in hunk.added_lines[
                        func_start_idx + 1 : func_start_idx + 16
                    ]
                ]
                body = "\n".join(body_lines)

                has_validation = any(
                    kw in body
                    for kw in (
                        "if not ",
                        "if ",
                        "raise ",
                        "ValueError",
                        "TypeError",
                        "assert ",
                        "validate",
                        "check",
                        "is None",
                    )
                )

                if not has_validation and len(params) >= 2:
                    findings.append(
                        ReviewFinding(
                            severity=Severity.SUGGESTION,
                            category=Category.CORRECTNESS,
                            title=f"No input validation: {func_name}()",
                            description=(
                                f"Public function `{func_name}` accepts {len(params)} parameters "
                                f"({', '.join(params)}) but the first 15 lines show no validation. "
                                f"Consider checking for None, empty, or out-of-range values."
                            ),
                            file_path=diff_file.path,
                            line_range=str(line_no),
                            code_snippet=line_text.strip(),
                        )
                    )

    return findings


def _check_mixed_responsibilities(files: list[DiffFile]) -> list[ReviewFinding]:
    """Flag functions that mix I/O operations with business logic."""
    findings: list[ReviewFinding] = []
    io_markers = re.compile(
        r"\b(?:open|read|write|connect|execute|fetch|request|send|recv|"
        r"subprocess|socket|cursor|commit|rollback|flush|close)\s*\("
    )
    logic_markers = re.compile(r"\b(?:if|for|while|return|yield)\b")

    for diff_file in files:
        if diff_file.is_deleted or _is_test_file(diff_file.path):
            continue

        for hunk in diff_file.hunks:
            # Track function boundaries in the hunk
            current_func: Optional[str] = None
            func_start: Optional[int] = None
            has_io = False
            logic_count = 0

            for line_no, line_text in hunk.added_lines:
                func_match = re.match(r"\s*(?:async\s+)?def\s+(\w+)", line_text)
                if func_match:
                    # Check previous function
                    if current_func and has_io and logic_count >= 5:
                        findings.append(
                            ReviewFinding(
                                severity=Severity.SUGGESTION,
                                category=Category.ARCHITECTURE,
                                title=f"Mixed responsibilities: {current_func}()",
                                description=(
                                    f"Function `{current_func}` contains both I/O operations and "
                                    f"significant logic ({logic_count} control flow statements). "
                                    f"Consider separating I/O from business logic for testability."
                                ),
                                file_path=diff_file.path,
                                line_range=str(func_start),
                            )
                        )
                    current_func = func_match.group(1)
                    func_start = line_no
                    has_io = False
                    logic_count = 0
                    continue

                if current_func:
                    if io_markers.search(line_text):
                        has_io = True
                    if logic_markers.search(line_text):
                        logic_count += 1

            # Check last function in hunk
            if current_func and has_io and logic_count >= 5:
                findings.append(
                    ReviewFinding(
                        severity=Severity.SUGGESTION,
                        category=Category.ARCHITECTURE,
                        title=f"Mixed responsibilities: {current_func}()",
                        description=(
                            f"Function `{current_func}` contains both I/O operations and "
                            f"significant logic ({logic_count} control flow statements). "
                            f"Consider separating I/O from business logic for testability."
                        ),
                        file_path=diff_file.path,
                        line_range=str(func_start),
                    )
                )

    return findings


# ── Memory Context Analysis ──────────────────────────────────────────────────


def _analyze_memory_context(
    files: list[DiffFile], memories: list[dict]
) -> list[ReviewFinding]:
    """Cross-reference diff changes with memory context to find inconsistencies."""
    findings: list[ReviewFinding] = []

    if not memories:
        return findings

    # Extract key terms from memories (decisions, patterns, preferences)
    memory_context = "\n".join(
        m.get("content", "") for m in memories if isinstance(m, dict)
    )

    # Check if the diff contradicts any stated decisions
    for diff_file in files:
        added_code = "\n".join(
            text
            for _, text in [
                line for hunk in diff_file.hunks for line in hunk.added_lines
            ]
        )

        # Look for framework/library mentions in memories vs diff
        # This is a heuristic — the LLM calling this tool does the deep analysis
        for memory in memories:
            if not isinstance(memory, dict):
                continue

            content = memory.get("content", "")
            title = memory.get("title", "")

            # Flag if memory mentions "do not use X" and diff introduces X
            not_use_patterns = re.findall(
                r"(?:do\s+not|don't|avoid|never)\s+(?:use|import|include)\s+(\w+)",
                content,
                re.IGNORECASE,
            )
            for forbidden in not_use_patterns:
                if re.search(rf"\b{re.escape(forbidden)}\b", added_code, re.IGNORECASE):
                    findings.append(
                        ReviewFinding(
                            severity=Severity.WARNING,
                            category=Category.CONSISTENCY,
                            title=f"Possible contradiction with prior decision",
                            description=(
                                f"Memory '{title}' indicates avoiding `{forbidden}`, "
                                f"but it appears in the diff for `{diff_file.path}`. "
                                f"Verify this is intentional."
                            ),
                            file_path=diff_file.path,
                            references=[f"memory: {memory.get('id', 'unknown')}"],
                        )
                    )

    return findings


# ── LLM Response Parsing ─────────────────────────────────────────────────────


def _parse_llm_findings(
    response_text: str, was_truncated: bool = False
) -> list[ReviewFinding]:
    """Parse LLM JSON response into ReviewFinding objects.

    If the response was truncated (max_tokens hit), attempts to salvage
    individual finding objects from the partial JSON.
    """
    findings: list[ReviewFinding] = []

    def _finding_from_dict(f: dict) -> ReviewFinding:
        try:
            severity = Severity(f.get("severity", "suggestion"))
        except ValueError:
            severity = Severity.SUGGESTION
        try:
            category = Category(f.get("category", "correctness"))
        except ValueError:
            category = Category.CORRECTNESS
        return ReviewFinding(
            severity=severity,
            category=category,
            title=f.get("title", "LLM finding"),
            description=f.get("description", ""),
            file_path=f.get("file_path"),
            line_range=f.get("line_range"),
            code_snippet=f.get("code_snippet"),
            suggestion=f.get("suggestion"),
        )

    try:
        # Strip markdown code fences if the LLM wrapped it
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()

        data = json.loads(cleaned)
        raw_findings = data.get("findings", [])

        for f in raw_findings:
            findings.append(_finding_from_dict(f))

        return findings

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Failed to parse LLM response as JSON: %s", e)

        # Salvage individual finding objects from truncated/malformed JSON
        # by extracting complete JSON objects that look like findings
        salvaged = _salvage_findings_from_partial_json(response_text)
        if salvaged:
            logger.info(
                "Salvaged %d findings from partial JSON response", len(salvaged)
            )
            for f in salvaged:
                findings.append(_finding_from_dict(f))
            if was_truncated:
                findings.append(
                    ReviewFinding(
                        severity=Severity.SUGGESTION,
                        category=Category.CORRECTNESS,
                        title="Response truncated",
                        description=(
                            "The LLM response was truncated (hit max_tokens limit). "
                            f"Salvaged {len(salvaged)} complete findings but there may be more. "
                            "Consider reviewing smaller diffs or increasing max_tokens."
                        ),
                    )
                )
            return findings

        # Last resort: return the raw text as a single finding
        if response_text.strip():
            findings.append(
                ReviewFinding(
                    severity=Severity.SUGGESTION,
                    category=Category.CORRECTNESS,
                    title="LLM review (unstructured)",
                    description=response_text.strip()[:2000],
                )
            )
        return findings


def _salvage_findings_from_partial_json(text: str) -> list[dict]:
    """Extract complete finding objects from a partial/truncated JSON response.

    Uses a brace-counting approach to find complete {...} objects that
    contain finding-like keys (severity, title, description).
    """
    findings = []
    # Find all potential JSON object starts within a "findings" array
    i = 0
    while i < len(text):
        if text[i] == "{":
            # Try to find the matching closing brace
            depth = 0
            j = i
            while j < len(text):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        # Found a complete object
                        candidate = text[i : j + 1]
                        try:
                            obj = json.loads(candidate)
                            # Check if it looks like a finding
                            if (
                                isinstance(obj, dict)
                                and "title" in obj
                                and "severity" in obj
                            ):
                                findings.append(obj)
                        except json.JSONDecodeError:
                            pass
                        break
                j += 1
        i += 1
    return findings


def _parse_llm_summary(response_text: str) -> str:
    """Extract summary from LLM JSON response."""
    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()

        data = json.loads(cleaned)
        return data.get("summary", "")
    except (json.JSONDecodeError, KeyError, TypeError):
        return ""


# ── Main Review Functions ────────────────────────────────────────────────────


def review_diff(
    diff_text: str,
    memories_json: Optional[str] = None,
    focus_areas: Optional[str] = None,
    on_progress: ProgressCallback | None = None,
) -> ReviewReport:
    """
    Main review function — regex first pass, then LLM adversarial analysis.

    Args:
        diff_text: Unified diff output (e.g., from `git diff`)
        memories_json: Optional JSON string of recent memory objects
        focus_areas: Optional comma-separated focus areas to prioritize
        on_progress: Optional callback for streaming progress updates
    """
    # Parse
    files = parse_diff(diff_text)
    if not files:
        report = ReviewReport(summary="No parseable diff content found.")
        return report

    stats = diff_stats(files)

    # Parse memories if provided
    memories: list[dict] = []
    if memories_json:
        try:
            parsed = json.loads(memories_json)
            if isinstance(parsed, list):
                memories = parsed
            elif isinstance(parsed, dict) and "data" in parsed:
                memories = parsed["data"]
        except (json.JSONDecodeError, TypeError):
            pass

    # Parse focus areas
    active_focus = DEFAULT_FOCUS_AREAS
    if focus_areas:
        active_focus = [a.strip().lower() for a in focus_areas.split(",")]

    # ── PASS 1: Regex + structural checks (deterministic, fast, free) ────
    regex_findings: list[ReviewFinding] = []
    regex_findings.extend(_match_anti_patterns(files))
    regex_findings.extend(_check_function_length(files))
    regex_findings.extend(_check_large_files(files))
    regex_findings.extend(_check_error_handling(files))
    regex_findings.extend(_check_missing_type_hints(files))
    regex_findings.extend(_check_duplicate_calls(files))
    regex_findings.extend(_check_input_validation(files))
    regex_findings.extend(_check_mixed_responsibilities(files))

    if memories:
        regex_findings.extend(_analyze_memory_context(files, memories))

    # ── PASS 2: LLM adversarial analysis (deep reasoning, chunked) ─────
    llm_findings: list[ReviewFinding] = []
    llm_summaries: list[str] = []

    # Chunk files to keep each LLM call within token limits
    chunks = _chunk_files(files)
    num_chunks = len(chunks)

    if num_chunks > 1:
        logger.info(
            "Diff split into %d chunks for LLM review (%d files total)",
            num_chunks,
            len(files),
        )

    for chunk_idx, chunk_files in enumerate(chunks):
        try:
            # Only include regex findings relevant to this chunk's files
            chunk_paths = {f.path for f in chunk_files}
            chunk_regex = [
                f
                for f in regex_findings
                if f.file_path is None or f.file_path in chunk_paths
            ]
            chunk_regex_json = json.dumps([f.to_dict() for f in chunk_regex], indent=2)

            # Reconstruct diff for just this chunk
            chunk_diff = _reconstruct_diff(chunk_files)
            chunk_file_names = ", ".join(f.path for f in chunk_files)

            user_message = build_review_diff_message(
                diff_text=chunk_diff,
                regex_findings_json=chunk_regex_json,
                memories_json=memories_json,
            )

            tool_label = (
                f"review_diff[{chunk_idx + 1}/{num_chunks}]"
                if num_chunks > 1
                else "review_diff"
            )
            logger.info(
                "LLM chunk %d/%d: files=[%s] (~%d chars)",
                chunk_idx + 1,
                num_chunks,
                chunk_file_names,
                len(chunk_diff),
            )

            response = llm.invoke(
                system_prompt=REVIEW_DIFF_SYSTEM,
                user_message=user_message,
                tool=tool_label,
                on_progress=on_progress,
            )
            response_text, stop_reason = response
            was_truncated = stop_reason == "max_tokens"
            chunk_findings = _parse_llm_findings(
                response_text, was_truncated=was_truncated
            )
            chunk_summary = _parse_llm_summary(response_text)

            llm_findings.extend(chunk_findings)
            if chunk_summary:
                llm_summaries.append(chunk_summary)

        except RuntimeError as e:
            logger.error(
                "LLM pass failed for chunk %d/%d: %s", chunk_idx + 1, num_chunks, e
            )
            llm_findings.append(
                ReviewFinding(
                    severity=Severity.SUGGESTION,
                    category=Category.CORRECTNESS,
                    title="LLM review unavailable for chunk",
                    description=(
                        f"Bedrock inference failed for files: "
                        f"{', '.join(f.path for f in chunk_files)}. "
                        f"Error: {e}. Showing regex-only results for these files."
                    ),
                )
            )

    llm_summary = " ".join(llm_summaries)

    # ── Merge findings ───────────────────────────────────────────────────
    all_findings = regex_findings + llm_findings

    # Filter by focus areas if specified
    if focus_areas:
        all_findings = [
            f
            for f in all_findings
            if str(f.category) in active_focus or str(f.severity) == "critical"
        ]

    # Sort: critical first, then warnings, then suggestions
    severity_order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.SUGGESTION: 2}
    all_findings.sort(key=lambda f: severity_order.get(f.severity, 99))

    # Build summary
    summary_parts = [
        f"Reviewed {stats['files_changed']} file(s): "
        f"+{stats['lines_added']}/-{stats['lines_removed']} lines.",
    ]
    if all_findings:
        crits = sum(1 for f in all_findings if f.severity == Severity.CRITICAL)
        warns = sum(1 for f in all_findings if f.severity == Severity.WARNING)
        suggs = sum(1 for f in all_findings if f.severity == Severity.SUGGESTION)
        parts = []
        if crits:
            parts.append(f"{crits} critical")
        if warns:
            parts.append(f"{warns} warning(s)")
        if suggs:
            parts.append(f"{suggs} suggestion(s)")
        summary_parts.append(f"Found {', '.join(parts)}.")
    else:
        summary_parts.append("No issues detected.")

    summary_parts.append(
        f"[Regex: {len(regex_findings)} findings, LLM: {len(llm_findings)} findings"
        + (f", {num_chunks} chunks" if num_chunks > 1 else "")
        + "]"
    )

    if llm_summary:
        summary_parts.append(f"LLM assessment: {llm_summary}")

    if memories:
        summary_parts.append(
            f"Cross-referenced against {len(memories)} memory entries."
        )

    report = ReviewReport(
        findings=all_findings,
        summary=" ".join(summary_parts),
        diff_stats=stats,
        memory_context_used=bool(memories),
    )

    return report


def review_pattern(
    pattern_description: str,
    code_snippet: str,
    on_progress: ProgressCallback | None = None,
) -> ReviewReport:
    """
    Scrutinize a specific code pattern — regex first pass, then LLM analysis.

    Args:
        pattern_description: What the pattern is supposed to do
        code_snippet: The actual code implementing the pattern
    """
    # ── PASS 1: Regex checks on the snippet ──────────────────────────────
    regex_findings: list[ReviewFinding] = []

    for name, pattern_def in ANTI_PATTERNS.items():
        if re.search(pattern_def["pattern"], code_snippet):
            regex_findings.append(
                ReviewFinding(
                    severity=Severity(pattern_def["severity"]),
                    category=Category(pattern_def["category"]),
                    title=pattern_def["title"],
                    description=pattern_def["description"],
                    code_snippet=code_snippet.strip()[:200],
                )
            )

    lines = code_snippet.strip().splitlines()

    # Nesting depth
    max_indent = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            level = indent // 4 if "    " in line[:indent] else indent
            max_indent = max(max_indent, level)

    if max_indent > 4:
        regex_findings.append(
            ReviewFinding(
                severity=Severity.WARNING,
                category=Category.COMPLEXITY,
                title="Deep nesting detected",
                description=(
                    f"Code has {max_indent} levels of nesting. "
                    f"Consider early returns, guard clauses, or extracting helper functions."
                ),
                code_snippet=code_snippet.strip()[:200],
            )
        )

    if len(lines) > 50:
        regex_findings.append(
            ReviewFinding(
                severity=Severity.SUGGESTION,
                category=Category.COMPLEXITY,
                title="Long code block",
                description=(
                    f"Code snippet is {len(lines)} lines. "
                    f"Consider splitting into smaller, focused functions."
                ),
            )
        )

    # Repeated calls
    call_pattern = re.compile(r"(\b\w+(?:\.\w+)*\s*\([^)]*\))")
    trivial = {
        "len",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "print",
        "type",
        "range",
        "enumerate",
        "isinstance",
        "append",
        "get",
        "strip",
        "split",
        "join",
        "format",
        "replace",
        "lower",
        "upper",
    }
    call_counts: dict[str, int] = {}
    for line in lines:
        for match in call_pattern.finditer(line):
            call_str = re.sub(r"\s+", "", match.group(1))
            func_name = call_str.split("(")[0].split(".")[-1]
            if func_name not in trivial:
                call_counts[call_str] = call_counts.get(call_str, 0) + 1

    for call_str, count in call_counts.items():
        if count >= 2:
            func_name = call_str.split("(")[0]
            regex_findings.append(
                ReviewFinding(
                    severity=Severity.WARNING,
                    category=Category.PERFORMANCE,
                    title=f"Repeated call: {func_name}()",
                    description=(
                        f"`{func_name}()` called {count} times with same args. "
                        f"Cache the result in a local variable if it doesn't change."
                    ),
                    code_snippet=call_str,
                )
            )

    # Hardcoded numeric params
    numeric_param_pattern = re.compile(
        r"(?:limit|timeout|max_retries|threshold|size|count|interval|delay|"
        r"retries|batch_size|page_size|max_size|buffer_size|chunk_size|"
        r"pool_size|workers|attempts)\s*[=:]\s*\d+"
    )
    for line in lines:
        match = numeric_param_pattern.search(line)
        if match:
            regex_findings.append(
                ReviewFinding(
                    severity=Severity.SUGGESTION,
                    category=Category.NAMING,
                    title="Hardcoded numeric parameter",
                    description=(
                        f"Found `{match.group(0)}`. Extract to a named constant "
                        f"or config value for clarity and single-point-of-change."
                    ),
                    code_snippet=match.group(0),
                )
            )

    # Mixed I/O and logic
    io_markers = re.compile(
        r"\b(?:open|read|write|connect|execute|fetch|request|send|"
        r"subprocess|socket|cursor|commit|flush|close)\s*\("
    )
    logic_markers = re.compile(r"\b(?:if|for|while|return|yield)\b")
    has_io = any(io_markers.search(line) for line in lines)
    logic_count = sum(1 for line in lines if logic_markers.search(line))
    if has_io and logic_count >= 5:
        regex_findings.append(
            ReviewFinding(
                severity=Severity.SUGGESTION,
                category=Category.ARCHITECTURE,
                title="Mixed I/O and logic",
                description=(
                    f"This code block contains I/O operations mixed with "
                    f"{logic_count} control flow statements. "
                    f"Separating I/O from business logic improves testability."
                ),
            )
        )

    # ── PASS 2: LLM analysis ────────────────────────────────────────────
    llm_findings: list[ReviewFinding] = []
    llm_summary = ""

    try:
        regex_findings_json = json.dumps(
            [f.to_dict() for f in regex_findings], indent=2
        )
        user_message = build_review_pattern_message(
            pattern_description=pattern_description,
            code_snippet=code_snippet,
            regex_findings_json=regex_findings_json,
        )
        response = llm.invoke(
            system_prompt=REVIEW_PATTERN_SYSTEM,
            user_message=user_message,
            tool="review_pattern",
            on_progress=on_progress,
        )
        response_text, stop_reason = response
        was_truncated = stop_reason == "max_tokens"
        llm_findings = _parse_llm_findings(response_text, was_truncated=was_truncated)
        llm_summary = _parse_llm_summary(response_text)
    except RuntimeError as e:
        logger.error("LLM pass failed for review_pattern: %s", e)

    # ── Merge ────────────────────────────────────────────────────────────
    all_findings = regex_findings + llm_findings

    summary = (
        f"Pattern review: '{pattern_description}'. "
        f"Analyzed {len(lines)} line(s). "
        f"[Regex: {len(regex_findings)}, LLM: {len(llm_findings)}]"
    )
    if llm_summary:
        summary += f" LLM: {llm_summary}"

    return ReviewReport(findings=all_findings, summary=summary)


def challenge_decision(
    decision: str,
    reasoning: str,
    alternatives: Optional[str] = None,
    on_progress: ProgressCallback | None = None,
) -> ReviewReport:
    """
    LLM-powered adversarial analysis of an architectural/design decision.

    The regex-based context-matching is kept as a fallback if the LLM call fails.

    Args:
        decision: The decision that was made
        reasoning: Why it was made
        alternatives: Optional comma-separated alternatives that were considered
    """
    # ── LLM pass (primary) ───────────────────────────────────────────────
    llm_findings: list[ReviewFinding] = []
    llm_summary = ""

    try:
        user_message = build_challenge_decision_message(
            decision=decision,
            reasoning=reasoning,
            alternatives=alternatives,
        )
        response = llm.invoke(
            system_prompt=CHALLENGE_DECISION_SYSTEM,
            user_message=user_message,
            tool="challenge_decision",
            on_progress=on_progress,
        )
        response_text, stop_reason = response
        was_truncated = stop_reason == "max_tokens"
        llm_findings = _parse_llm_findings(response_text, was_truncated=was_truncated)
        llm_summary = _parse_llm_summary(response_text)
    except RuntimeError as e:
        logger.error("LLM pass failed for challenge_decision: %s", e)

    # ── Fallback: regex-based challenges if LLM failed ───────────────────
    if not llm_findings:
        llm_findings = _challenge_decision_regex_fallback(
            decision, reasoning, alternatives
        )
        llm_summary = "LLM unavailable — showing heuristic challenges only."

    all_findings = llm_findings

    summary = (
        f"Adversarial challenge of decision: '{decision}'. "
        f"Generated {len(all_findings)} challenge points."
    )
    if llm_summary:
        summary += f" {llm_summary}"

    return ReviewReport(findings=all_findings, summary=summary)


def _challenge_decision_regex_fallback(
    decision: str,
    reasoning: str,
    alternatives: Optional[str] = None,
) -> list[ReviewFinding]:
    """
    Regex/heuristic-based challenge — used as fallback when LLM is unavailable.
    """
    findings: list[ReviewFinding] = []

    alt_list = []
    if alternatives:
        alt_list = [a.strip() for a in alternatives.split(",") if a.strip()]

    decision_lower = decision.lower()
    reasoning_lower = reasoning.lower()

    # ── Context-specific challenges based on decision content ────────────

    # Scaling concerns
    if any(
        w in decision_lower
        for w in ("sqlite", "file", "local", "single", "memory", "json")
    ):
        findings.append(
            ReviewFinding(
                severity=Severity.WARNING,
                category=Category.ARCHITECTURE,
                title="Scaling wall",
                description=(
                    f"The decision to use '{decision}' works for current scale. "
                    f"But what happens at 10x or 100x? Single-file/local/in-memory stores "
                    f"hit concurrency limits, lock contention, and size ceilings. "
                    f"What is the concrete migration path when that happens? "
                    f"Is there a point-of-no-return where switching costs exceed the benefit?"
                ),
            )
        )
    elif any(
        w in decision_lower
        for w in ("distributed", "microservice", "separate", "decouple")
    ):
        findings.append(
            ReviewFinding(
                severity=Severity.WARNING,
                category=Category.ARCHITECTURE,
                title="Premature distribution",
                description=(
                    f"Separating into '{decision}' adds network hops, partial failure modes, "
                    f"and operational complexity. The reasoning says: '{reasoning[:120]}'. "
                    f"Is this complexity justified by actual current load, or is it speculative? "
                    f"Could a simpler in-process approach work until proven otherwise?"
                ),
            )
        )

    # Stateless claims
    if "stateless" in decision_lower or "stateless" in reasoning_lower:
        findings.append(
            ReviewFinding(
                severity=Severity.WARNING,
                category=Category.ARCHITECTURE,
                title="Is it truly stateless?",
                description=(
                    f"The decision claims statelessness, but examine the edges: "
                    f"Are there in-memory caches, connection pools, loaded configs, or "
                    f"warm-up costs that effectively create hidden state? "
                    f"What happens if the process restarts mid-request?"
                ),
            )
        )

    # Security surface
    if any(
        w in decision_lower
        for w in ("api", "http", "endpoint", "server", "port", "expose")
    ):
        findings.append(
            ReviewFinding(
                severity=Severity.WARNING,
                category=Category.SECURITY,
                title="Attack surface expansion",
                description=(
                    f"Exposing '{decision}' creates network attack surface. "
                    f"What authentication/authorization guards it? "
                    f"What happens if an attacker sends malformed input? "
                    f"Is there rate limiting? Input size limits? "
                    f"The reasoning mentions: '{reasoning[:100]}' -- does it address these?"
                ),
            )
        )

    # Dependency risk
    if any(
        w in decision_lower
        for w in ("library", "framework", "dependency", "package", "use ")
    ):
        findings.append(
            ReviewFinding(
                severity=Severity.WARNING,
                category=Category.ARCHITECTURE,
                title="Dependency risk",
                description=(
                    f"'{decision}' introduces a dependency. "
                    f"What is the maintenance status of this dependency? "
                    f"What happens if it's abandoned, has a security vulnerability, "
                    f"or makes breaking API changes? How coupled is the codebase to it? "
                    f"Could the interface be wrapped to isolate the dependency?"
                ),
            )
        )

    # Data handling
    if any(
        w in decision_lower for w in ("store", "database", "persist", "cache", "save")
    ):
        findings.append(
            ReviewFinding(
                severity=Severity.WARNING,
                category=Category.CORRECTNESS,
                title="Data integrity under failure",
                description=(
                    f"'{decision}' involves data persistence. "
                    f"What happens during a crash mid-write? Is there WAL/journaling? "
                    f"What about concurrent writes from multiple processes? "
                    f"Is there a backup/recovery strategy? "
                    f"The reasoning says: '{reasoning[:100]}' -- does it cover failure modes?"
                ),
            )
        )

    # Always include these general challenges with context-specific framing
    findings.append(
        ReviewFinding(
            severity=Severity.SUGGESTION,
            category=Category.ARCHITECTURE,
            title="Reversibility",
            description=(
                f"How hard is it to reverse '{decision}'? "
                f"Decisions are cheap when reversible and expensive when not. "
                f"Estimate the cost (in time and code changes) to undo this if it turns out wrong."
            ),
        )
    )

    findings.append(
        ReviewFinding(
            severity=Severity.SUGGESTION,
            category=Category.ARCHITECTURE,
            title="Second-order effects",
            description=(
                f"'{decision}' solves the immediate problem. But what does it force downstream? "
                f"What constraints does it impose on the next 3 decisions that will follow? "
                f"Are any of those constraints unacceptable?"
            ),
        )
    )

    findings.append(
        ReviewFinding(
            severity=Severity.SUGGESTION,
            category=Category.ARCHITECTURE,
            title="Reasoning gaps",
            description=(
                f"The stated reasoning is: '{reasoning}'. "
                f"What unstated assumptions does this depend on? "
                f"If any of those assumptions turn out to be wrong, does the decision still hold?"
            ),
        )
    )

    # If no alternatives were considered, flag it more aggressively
    if not alt_list:
        findings.insert(
            0,
            ReviewFinding(
                severity=Severity.WARNING,
                category=Category.ARCHITECTURE,
                title="No alternatives evaluated",
                description=(
                    "No alternative approaches were provided. "
                    "A decision without explicit comparison to alternatives is not a decision -- "
                    "it's a default. What else was considered and concretely rejected?"
                ),
            ),
        )
    else:
        # Challenge each specific alternative
        for alt in alt_list:
            findings.append(
                ReviewFinding(
                    severity=Severity.SUGGESTION,
                    category=Category.ARCHITECTURE,
                    title=f"Why not {alt}?",
                    description=(
                        f"'{alt}' was listed as an alternative to '{decision}'. "
                        f"What specific, concrete disadvantage of '{alt}' made it lose? "
                        f"Is that disadvantage real at current scale, or only hypothetical? "
                        f"Would '{alt}' have been simpler to implement or maintain?"
                    ),
                )
            )

    return findings
