"""Static checks — regex anti-pattern matching, structural analysis, memory cross-ref."""

from __future__ import annotations

import re
from typing import Optional

from reviewer.config import ANTI_PATTERNS
from reviewer.diff_parser import DiffFile
from reviewer.models import (
    Category,
    ReviewFinding,
    Severity,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def is_test_file(path: str) -> bool:
    """Check if a file path looks like a test file."""
    path_lower = path.lower()
    return (
        "test" in path_lower
        or "spec" in path_lower
        or path_lower.startswith("tests/")
        or "/tests/" in path_lower
    )


# ── Anti-Pattern Matching ────────────────────────────────────────────────────


def match_anti_patterns(files: list[DiffFile]) -> list[ReviewFinding]:
    """Run regex-based anti-pattern detection on added lines only."""
    findings: list[ReviewFinding] = []

    for diff_file in files:
        for hunk in diff_file.hunks:
            for line_no, line_text in hunk.added_lines:
                for name, pattern_def in ANTI_PATTERNS.items():
                    if re.search(pattern_def["pattern"], line_text):
                        # Skip assert pattern for test files
                        if name == "assert_in_production" and is_test_file(
                            diff_file.path
                        ):
                            continue
                        # Skip print statements in test files
                        if name == "print_statement" and is_test_file(diff_file.path):
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


# ── Structural Analysis ─────────────────────────────────────────────────────


def check_function_length(files: list[DiffFile]) -> list[ReviewFinding]:
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


def check_large_files(files: list[DiffFile]) -> list[ReviewFinding]:
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


def check_error_handling(files: list[DiffFile]) -> list[ReviewFinding]:
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


def check_missing_type_hints(files: list[DiffFile]) -> list[ReviewFinding]:
    """Flag Python function definitions without type hints."""
    findings: list[ReviewFinding] = []

    for diff_file in files:
        if not diff_file.path.endswith(".py"):
            continue
        if is_test_file(diff_file.path):
            continue

        for hunk in diff_file.hunks:
            for line_no, line_text in hunk.added_lines:
                # Match def without return type annotation
                func_match = re.match(
                    r"\s*(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)\s*:", line_text
                )
                if func_match:
                    func_name = func_match.group(1)

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


def check_duplicate_calls(files: list[DiffFile]) -> list[ReviewFinding]:
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
                    if func_name in _TRIVIAL_CALLS:
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


# Shared set of trivial built-in/method names to skip in duplicate-call detection
_TRIVIAL_CALLS = frozenset(
    {
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
    }
)


def check_input_validation(files: list[DiffFile]) -> list[ReviewFinding]:
    """Flag public functions that accept parameters without any visible validation."""
    findings: list[ReviewFinding] = []

    for diff_file in files:
        if not diff_file.path.endswith(".py"):
            continue
        if diff_file.is_deleted or is_test_file(diff_file.path):
            continue

        for hunk in diff_file.hunks:
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


def check_mixed_responsibilities(files: list[DiffFile]) -> list[ReviewFinding]:
    """Flag functions that mix I/O operations with business logic."""
    findings: list[ReviewFinding] = []
    io_markers = re.compile(
        r"\b(?:open|read|write|connect|execute|fetch|request|send|recv|"
        r"subprocess|socket|cursor|commit|rollback|flush|close)\s*\("
    )
    logic_markers = re.compile(r"\b(?:if|for|while|return|yield)\b")

    for diff_file in files:
        if diff_file.is_deleted or is_test_file(diff_file.path):
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


def analyze_memory_context(
    files: list[DiffFile], memories: list[dict]
) -> list[ReviewFinding]:
    """Cross-reference diff changes with memory context to find inconsistencies."""
    findings: list[ReviewFinding] = []

    if not memories:
        return findings

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


# ── Snippet-Level Checks (for review_pattern) ───────────────────────────────


def check_snippet_anti_patterns(code_snippet: str) -> list[ReviewFinding]:
    """Run anti-pattern regex checks against a raw code snippet."""
    findings: list[ReviewFinding] = []

    for name, pattern_def in ANTI_PATTERNS.items():
        if re.search(pattern_def["pattern"], code_snippet):
            findings.append(
                ReviewFinding(
                    severity=Severity(pattern_def["severity"]),
                    category=Category(pattern_def["category"]),
                    title=pattern_def["title"],
                    description=pattern_def["description"],
                    code_snippet=code_snippet.strip()[:200],
                )
            )

    return findings


def check_snippet_nesting(lines: list[str]) -> list[ReviewFinding]:
    """Check nesting depth of a code snippet."""
    findings: list[ReviewFinding] = []

    max_indent = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            level = indent // 4 if "    " in line[:indent] else indent
            max_indent = max(max_indent, level)

    if max_indent > 4:
        findings.append(
            ReviewFinding(
                severity=Severity.WARNING,
                category=Category.COMPLEXITY,
                title="Deep nesting detected",
                description=(
                    f"Code has {max_indent} levels of nesting. "
                    f"Consider early returns, guard clauses, or extracting helper functions."
                ),
            )
        )

    return findings


def check_snippet_length(lines: list[str]) -> list[ReviewFinding]:
    """Flag overly long code snippets."""
    findings: list[ReviewFinding] = []

    if len(lines) > 50:
        findings.append(
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

    return findings


def check_snippet_repeated_calls(lines: list[str]) -> list[ReviewFinding]:
    """Detect repeated function calls in a code snippet."""
    findings: list[ReviewFinding] = []
    call_pattern = re.compile(r"(\b\w+(?:\.\w+)*\s*\([^)]*\))")
    call_counts: dict[str, int] = {}
    for line in lines:
        for match in call_pattern.finditer(line):
            call_str = re.sub(r"\s+", "", match.group(1))
            func_name = call_str.split("(")[0].split(".")[-1]
            if func_name not in _TRIVIAL_CALLS:
                call_counts[call_str] = call_counts.get(call_str, 0) + 1

    for call_str, count in call_counts.items():
        if count >= 2:
            func_name = call_str.split("(")[0]
            findings.append(
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

    return findings


def check_snippet_hardcoded_numerics(lines: list[str]) -> list[ReviewFinding]:
    """Flag hardcoded numeric parameters in a code snippet."""
    findings: list[ReviewFinding] = []
    numeric_param_pattern = re.compile(
        r"(?:limit|timeout|max_retries|threshold|size|count|interval|delay|"
        r"retries|batch_size|page_size|max_size|buffer_size|chunk_size|"
        r"pool_size|workers|attempts)\s*[=:]\s*\d+"
    )
    for line in lines:
        match = numeric_param_pattern.search(line)
        if match:
            findings.append(
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

    return findings


def check_snippet_mixed_io(lines: list[str]) -> list[ReviewFinding]:
    """Check if a code snippet mixes I/O with business logic."""
    findings: list[ReviewFinding] = []
    io_markers = re.compile(
        r"\b(?:open|read|write|connect|execute|fetch|request|send|"
        r"subprocess|socket|cursor|commit|flush|close)\s*\("
    )
    logic_markers = re.compile(r"\b(?:if|for|while|return|yield)\b")
    has_io = any(io_markers.search(line) for line in lines)
    logic_count = sum(1 for line in lines if logic_markers.search(line))
    if has_io and logic_count >= 5:
        findings.append(
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

    return findings
