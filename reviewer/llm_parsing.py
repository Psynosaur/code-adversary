"""LLM response parsing — extract structured findings from JSON/partial responses."""

from __future__ import annotations

import json
import logging

from reviewer.models import (
    Category,
    ReviewFinding,
    Severity,
)

logger = logging.getLogger(__name__)


def _strip_code_fence(text: str) -> str:
    """Strip markdown code fences (```...```) from LLM response text.

    Handles: opening fence with language tag, closing fence,
    both fences, neither fence, and edge cases like missing newline.
    Only strips the closing fence if an opening fence was also found.
    """
    cleaned = text.strip()

    had_opening = False
    if cleaned.startswith("```"):
        had_opening = True
        newline_pos = cleaned.find("\n")
        if newline_pos == -1:
            # Opening fence with no newline — entire string is just the fence
            return ""
        cleaned = cleaned[newline_pos + 1 :]

    if had_opening and cleaned.endswith("```"):
        cleaned = cleaned[:-3].rstrip()

    return cleaned


def parse_llm_findings(
    response_text: str, was_truncated: bool = False
) -> list[ReviewFinding]:
    """Parse LLM JSON response into ReviewFinding objects.

    If the response was truncated (max_tokens hit), attempts to salvage
    individual finding objects from the partial JSON.
    """
    findings: list[ReviewFinding] = []

    try:
        cleaned = _strip_code_fence(response_text)

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


def parse_llm_summary(response_text: str) -> str:
    """Extract summary from LLM JSON response."""
    try:
        cleaned = _strip_code_fence(response_text)
        data = json.loads(cleaned)
        return data.get("summary", "")
    except (json.JSONDecodeError, KeyError, TypeError):
        return ""


def _finding_from_dict(f: dict) -> ReviewFinding:
    """Convert a raw dict to a ReviewFinding, with safe enum parsing."""
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
