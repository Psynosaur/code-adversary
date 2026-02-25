"""Core review orchestration — wires together parsing, checks, LLM, and reporting."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Optional

from reviewer.checks import (
    analyze_memory_context,
    check_duplicate_calls,
    check_error_handling,
    check_function_length,
    check_input_validation,
    check_large_files,
    check_missing_type_hints,
    check_mixed_responsibilities,
    check_snippet_anti_patterns,
    check_snippet_hardcoded_numerics,
    check_snippet_length,
    check_snippet_mixed_io,
    check_snippet_nesting,
    check_snippet_repeated_calls,
    match_anti_patterns,
)
from reviewer.chunking import chunk_files, reconstruct_diff
from reviewer.config import DEFAULT_FOCUS_AREAS, DIFF_CHUNK_MAX_CHARS
from reviewer.diff_parser import diff_stats, parse_diff
from reviewer.llm_parsing import parse_llm_findings, parse_llm_summary
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
    REVIEW_FILES_SYSTEM,
    CHALLENGE_DECISION_SYSTEM,
    build_review_diff_message,
    build_review_pattern_message,
    build_review_files_message,
    build_challenge_decision_message,
)

logger = logging.getLogger(__name__)


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
    regex_findings.extend(match_anti_patterns(files))
    regex_findings.extend(check_function_length(files))
    regex_findings.extend(check_large_files(files))
    regex_findings.extend(check_error_handling(files))
    regex_findings.extend(check_missing_type_hints(files))
    regex_findings.extend(check_duplicate_calls(files))
    regex_findings.extend(check_input_validation(files))
    regex_findings.extend(check_mixed_responsibilities(files))

    if memories:
        regex_findings.extend(analyze_memory_context(files, memories))

    # ── PASS 2: LLM adversarial analysis (deep reasoning, chunked) ─────
    llm_findings: list[ReviewFinding] = []
    llm_summaries: list[str] = []
    llm_chunk_failures = 0

    # Chunk files to keep each LLM call within token limits
    chunks = chunk_files(files)
    num_chunks = len(chunks)

    if num_chunks > 1:
        logger.info(
            "Diff split into %d chunks for LLM review (%d files total)",
            num_chunks,
            len(files),
        )

    for chunk_idx, chunk_file_list in enumerate(chunks):
        try:
            # Only include regex findings relevant to this chunk's files
            chunk_paths = {f.path for f in chunk_file_list}
            chunk_regex = [
                f
                for f in regex_findings
                if f.file_path is None or f.file_path in chunk_paths
            ]
            chunk_regex_json = json.dumps([f.to_dict() for f in chunk_regex], indent=2)

            # Reconstruct diff for just this chunk
            chunk_diff = reconstruct_diff(chunk_file_list)
            chunk_file_names = ", ".join(f.path for f in chunk_file_list)

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
            chunk_findings = parse_llm_findings(
                response_text, was_truncated=was_truncated
            )
            chunk_summary = parse_llm_summary(response_text)

            llm_findings.extend(chunk_findings)
            if chunk_summary:
                llm_summaries.append(chunk_summary)

        except RuntimeError as e:
            llm_chunk_failures += 1
            logger.error(
                "LLM pass failed for chunk %d/%d: %s", chunk_idx + 1, num_chunks, e
            )
            llm_findings.append(
                ReviewFinding(
                    severity=Severity.WARNING,
                    category=Category.CORRECTNESS,
                    title="LLM review unavailable for chunk",
                    description=(
                        f"Bedrock inference failed for files: "
                        f"{', '.join(f.path for f in chunk_file_list)}. "
                        f"Error: {e}. Showing regex-only results for these files."
                    ),
                )
            )

    llm_summary = " ".join(llm_summaries)

    # Surface total LLM failure if all chunks failed
    if llm_chunk_failures == num_chunks and num_chunks > 0:
        llm_findings.append(
            ReviewFinding(
                severity=Severity.WARNING,
                category=Category.CORRECTNESS,
                title="LLM analysis completely unavailable",
                description=(
                    f"All {num_chunks} LLM chunk(s) failed. "
                    f"Review contains regex-based findings only — "
                    f"deep analysis (logic errors, edge cases, design issues) was not performed."
                ),
            )
        )

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
    lines = code_snippet.strip().splitlines()

    # ── PASS 1: Regex checks on the snippet ──────────────────────────────
    regex_findings: list[ReviewFinding] = []
    regex_findings.extend(check_snippet_anti_patterns(code_snippet))
    regex_findings.extend(check_snippet_nesting(lines))
    regex_findings.extend(check_snippet_length(lines))
    regex_findings.extend(check_snippet_repeated_calls(lines))
    regex_findings.extend(check_snippet_hardcoded_numerics(lines))
    regex_findings.extend(check_snippet_mixed_io(lines))

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
        llm_findings = parse_llm_findings(response_text, was_truncated=was_truncated)
        llm_summary = parse_llm_summary(response_text)
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


def review_files(
    file_paths: list[str],
    file_contents: list[str],
    memories_json: Optional[str] = None,
    focus_areas: Optional[str] = None,
    on_progress: ProgressCallback | None = None,
) -> ReviewReport:
    """
    Review source files in their entirety — regex first pass, then LLM analysis.

    Unlike review_diff, this reviews full source code rather than changes.
    Useful for auditing existing code, onboarding to a codebase, or reviewing
    files that weren't part of a recent diff.

    Args:
        file_paths: List of file paths (for context/reporting).
        file_contents: List of file contents (parallel to file_paths).
        memories_json: Optional JSON string of recent memory objects.
        focus_areas: Optional comma-separated focus areas to prioritize.
        on_progress: Optional callback for streaming progress updates.
    """
    if not file_paths or not file_contents:
        return ReviewReport(summary="No files provided for review.")

    if len(file_paths) != len(file_contents):
        return ReviewReport(
            summary=f"Mismatch: {len(file_paths)} paths but {len(file_contents)} contents."
        )

    # Build file descriptors
    files = [
        {"path": path, "content": content}
        for path, content in zip(file_paths, file_contents)
    ]

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

    # ── PASS 1: Regex checks on each file ────────────────────────────────
    regex_findings: list[ReviewFinding] = []
    total_lines = 0

    for f in files:
        lines = f["content"].splitlines()
        total_lines += len(lines)
        path = f["path"]

        # Use dataclasses.replace to avoid mutating objects returned by
        # check functions — they may share instances across calls.
        for check_fn, check_arg in [
            (check_snippet_anti_patterns, f["content"]),
            (check_snippet_nesting, lines),
            (check_snippet_length, lines),
            (check_snippet_repeated_calls, lines),
            (check_snippet_hardcoded_numerics, lines),
            (check_snippet_mixed_io, lines),
        ]:
            for finding in check_fn(check_arg):
                regex_findings.append(replace(finding, file_path=path))

    # ── PASS 2: LLM adversarial analysis (chunked by file) ──────────────
    llm_findings: list[ReviewFinding] = []
    llm_summaries: list[str] = []
    llm_chunk_failures = 0

    # Chunk files to stay within token limits
    chunks = _chunk_source_files(files)
    num_chunks = len(chunks)

    if num_chunks > 1:
        logger.info(
            "Source files split into %d chunks for LLM review (%d files total)",
            num_chunks,
            len(files),
        )

    for chunk_idx, chunk_files_list in enumerate(chunks):
        try:
            # Regex findings relevant to this chunk
            chunk_paths = {f["path"] for f in chunk_files_list}
            chunk_regex = [
                f
                for f in regex_findings
                if f.file_path is None or f.file_path in chunk_paths
            ]
            chunk_regex_json = json.dumps([f.to_dict() for f in chunk_regex], indent=2)

            user_message = build_review_files_message(
                files=chunk_files_list,
                regex_findings_json=chunk_regex_json,
                memories_json=memories_json,
                focus_areas=focus_areas,
            )

            tool_label = (
                f"review_files[{chunk_idx + 1}/{num_chunks}]"
                if num_chunks > 1
                else "review_files"
            )
            chunk_file_names = ", ".join(f["path"] for f in chunk_files_list)
            chunk_chars = sum(len(f["content"]) for f in chunk_files_list)
            logger.info(
                "LLM chunk %d/%d: files=[%s] (~%d chars)",
                chunk_idx + 1,
                num_chunks,
                chunk_file_names,
                chunk_chars,
            )

            response = llm.invoke(
                system_prompt=REVIEW_FILES_SYSTEM,
                user_message=user_message,
                tool=tool_label,
                on_progress=on_progress,
            )
            response_text, stop_reason = response
            was_truncated = stop_reason == "max_tokens"
            chunk_findings = parse_llm_findings(
                response_text, was_truncated=was_truncated
            )
            chunk_summary = parse_llm_summary(response_text)

            llm_findings.extend(chunk_findings)
            if chunk_summary:
                llm_summaries.append(chunk_summary)

        except RuntimeError as e:
            llm_chunk_failures += 1
            logger.error(
                "LLM pass failed for chunk %d/%d: %s", chunk_idx + 1, num_chunks, e
            )
            llm_findings.append(
                ReviewFinding(
                    severity=Severity.WARNING,
                    category=Category.CORRECTNESS,
                    title="LLM review unavailable for chunk",
                    description=(
                        f"Bedrock inference failed for files: "
                        f"{', '.join(f['path'] for f in chunk_files_list)}. "
                        f"Error: {e}. Showing regex-only results for these files."
                    ),
                )
            )

    llm_summary = " ".join(llm_summaries)

    # Surface total LLM failure if all chunks failed
    if llm_chunk_failures == num_chunks and num_chunks > 0:
        llm_findings.append(
            ReviewFinding(
                severity=Severity.WARNING,
                category=Category.CORRECTNESS,
                title="LLM analysis completely unavailable",
                description=(
                    f"All {num_chunks} LLM chunk(s) failed. "
                    f"Review contains regex-based findings only — "
                    f"deep analysis (logic errors, edge cases, design issues) was not performed."
                ),
            )
        )

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
        f"Reviewed {len(files)} file(s), {total_lines} total lines.",
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

    return ReviewReport(
        findings=all_findings,
        summary=" ".join(summary_parts),
        diff_stats={
            "files_changed": len(files),
            "lines_added": total_lines,
            "lines_removed": 0,
        },
        memory_context_used=bool(memories),
    )


def _chunk_source_files(
    files: list[dict[str, str]],
) -> list[list[dict[str, str]]]:
    """Split source files into chunks that fit within token limits.

    Each chunk contains one or more files. Oversized single files
    get their own chunk (the LLM will handle truncation).
    """
    if not files:
        return []

    max_chars = DIFF_CHUNK_MAX_CHARS
    chunks: list[list[dict[str, str]]] = []
    current_chunk: list[dict[str, str]] = []
    current_chars = 0

    for f in files:
        file_chars = len(f["content"]) + 200  # overhead for path/formatting

        if current_chunk and (current_chars + file_chars) > max_chars:
            chunks.append(current_chunk)
            current_chunk = [f]
            current_chars = file_chars
        else:
            current_chunk.append(f)
            current_chars += file_chars

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


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
        llm_findings = parse_llm_findings(response_text, was_truncated=was_truncated)
        llm_summary = parse_llm_summary(response_text)
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
