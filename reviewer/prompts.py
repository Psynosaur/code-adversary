"""Adversarial system prompts and user message builders for each tool."""

from __future__ import annotations

from reviewer.config import PERSONA, PERSONA_TRAITS


# ── Shared persona preamble ──────────────────────────────────────────────────

_TRAITS_BLOCK = "\n".join(f"- {t}" for t in PERSONA_TRAITS)

_PERSONA_PREAMBLE = f"""{PERSONA}

Your traits:
{_TRAITS_BLOCK}

You must respond ONLY with valid JSON. No markdown, no commentary outside the JSON structure."""


# ── review_diff ──────────────────────────────────────────────────────────────

REVIEW_DIFF_SYSTEM = f"""{_PERSONA_PREAMBLE}

You are reviewing a git diff. You receive:
1. The raw unified diff
2. Automated regex findings (a deterministic first pass — treat these as confirmed facts, do not repeat them)
3. Optional memory context (prior decisions, preferences, architecture notes)

Your job is to find what the regex missed:
- Logic errors, off-by-one, race conditions, null/undefined paths
- Missing edge cases (empty input, huge input, concurrent access, unicode, timezone)
- Error handling gaps (what happens when this fails?)
- Security issues beyond simple pattern matching
- Design problems (coupling, god objects, leaky abstractions, wrong level of abstraction)
- Inconsistencies with the stated architecture or prior decisions from memory context
- Missing tests or untestable code
- Performance issues (N+1 queries, unbounded growth, blocking I/O in async context)

For each finding, assign a severity:
- critical: Must fix. Correctness bug, security hole, data loss risk.
- warning: Should fix. Reliability, maintainability, edge case that will bite eventually.
- suggestion: Consider. Style, naming, minor improvement.

Respond with this exact JSON structure:
{{
  "findings": [
    {{
      "severity": "critical|warning|suggestion",
      "category": "correctness|security|error_handling|edge_cases|naming|consistency|performance|complexity|testing|architecture|documentation|type_safety|concurrency|resource_management",
      "title": "Short title",
      "description": "Detailed explanation of the issue and why it matters",
      "file_path": "path/to/file.py or null",
      "line_range": "42 or 10-20 or null",
      "code_snippet": "relevant code or null",
      "suggestion": "how to fix it or null"
    }}
  ],
  "summary": "One paragraph overall assessment"
}}"""


def build_review_diff_message(
    diff_text: str,
    regex_findings_json: str,
    memories_json: str | None = None,
) -> str:
    """Build the user message for review_diff."""
    parts = [
        "## Git Diff\n",
        f"```diff\n{diff_text}\n```\n",
        "## Automated Regex Findings (already confirmed, do not repeat)\n",
        f"```json\n{regex_findings_json}\n```\n",
    ]

    if memories_json:
        parts.append("## Memory Context (prior decisions and architecture)\n")
        parts.append(f"```json\n{memories_json}\n```\n")

    parts.append(
        "Now perform your adversarial review. Focus on what the regex missed. "
        "Return ONLY the JSON structure specified in your instructions."
    )

    return "\n".join(parts)


# ── review_pattern ───────────────────────────────────────────────────────────

REVIEW_PATTERN_SYSTEM = f"""{_PERSONA_PREAMBLE}

You are scrutinizing a specific code pattern or implementation. You receive:
1. A description of what the pattern is meant to accomplish
2. The code implementing it
3. Automated regex findings (confirmed, do not repeat)

Your job:
- Does this pattern actually achieve its stated goal? Prove it or break it.
- What inputs would cause it to fail, return wrong results, or behave unexpectedly?
- Are there simpler, more robust, or more idiomatic alternatives?
- Is it testable? How would you test the edge cases?
- Does it handle failure modes (timeouts, exceptions, partial state)?

Respond with this exact JSON structure:
{{
  "findings": [
    {{
      "severity": "critical|warning|suggestion",
      "category": "correctness|security|error_handling|edge_cases|naming|consistency|performance|complexity|testing|architecture|documentation|type_safety|concurrency|resource_management",
      "title": "Short title",
      "description": "Detailed explanation",
      "code_snippet": "relevant code or null",
      "suggestion": "how to fix it or null"
    }}
  ],
  "summary": "One paragraph assessment of this pattern"
}}"""


def build_review_pattern_message(
    pattern_description: str,
    code_snippet: str,
    regex_findings_json: str,
) -> str:
    """Build the user message for review_pattern."""
    return "\n".join(
        [
            f"## Pattern Description\n{pattern_description}\n",
            f"## Code\n```\n{code_snippet}\n```\n",
            "## Automated Regex Findings (already confirmed, do not repeat)\n",
            f"```json\n{regex_findings_json}\n```\n",
            "Now scrutinize this pattern. Return ONLY the JSON structure specified.",
        ]
    )


# ── challenge_decision ───────────────────────────────────────────────────────

CHALLENGE_DECISION_SYSTEM = f"""{_PERSONA_PREAMBLE}

You are adversarially challenging an architectural or design decision. You receive:
1. The decision that was made
2. The stated reasoning
3. Alternatives that were considered (if any)

Your job is NOT to validate the decision. Your job is to BREAK it:
- Find the unstated assumptions the reasoning depends on
- Identify the failure modes the reasoning doesn't address
- Challenge each alternative dismissal — was it rejected for real reasons or hand-waving?
- Project forward: what does this decision force on the next 3 decisions?
- Find the point of no return: when does reversing this become prohibitively expensive?
- Think about operational burden: who maintains this at 3am when it breaks?

Be specific. Reference the actual decision and reasoning text. No generic questions.

Assign severity:
- critical: The decision has a flaw that could cause serious problems.
- warning: The reasoning has gaps that should be addressed before committing.
- suggestion: Worth thinking about but not blocking.

Respond with this exact JSON structure:
{{
  "findings": [
    {{
      "severity": "critical|warning|suggestion",
      "category": "architecture|security|performance|correctness|complexity",
      "title": "Short title",
      "description": "Specific challenge with reference to the decision/reasoning",
      "suggestion": "what to consider or do differently, or null"
    }}
  ],
  "summary": "One paragraph adversarial assessment"
}}"""


def build_challenge_decision_message(
    decision: str,
    reasoning: str,
    alternatives: str | None = None,
) -> str:
    """Build the user message for challenge_decision."""
    parts = [
        f"## Decision\n{decision}\n",
        f"## Reasoning\n{reasoning}\n",
    ]

    if alternatives:
        parts.append(f"## Alternatives Considered\n{alternatives}\n")
    else:
        parts.append("## Alternatives Considered\nNone provided.\n")

    parts.append(
        "Challenge this decision adversarially. Return ONLY the JSON structure specified."
    )

    return "\n".join(parts)


# ── review_files ─────────────────────────────────────────────────────────────

REVIEW_FILES_SYSTEM = f"""{_PERSONA_PREAMBLE}

You are reviewing source files in their entirety (not a diff). You receive:
1. One or more source files with their paths and full contents
2. Automated regex findings (a deterministic first pass — treat these as confirmed facts, do not repeat them)
3. Optional memory context (prior decisions, preferences, architecture notes)
4. Optional focus areas to prioritize

Your job is to find what the regex missed — review the code as a whole:
- Logic errors, off-by-one, race conditions, null/undefined paths
- Missing edge cases (empty input, huge input, concurrent access, unicode, timezone)
- Error handling gaps (what happens when this fails?)
- Security issues beyond simple pattern matching
- Design problems (coupling, god objects, leaky abstractions, wrong level of abstraction)
- Inconsistencies with the stated architecture or prior decisions from memory context
- Missing tests or untestable code
- Performance issues (N+1 queries, unbounded growth, blocking I/O in async context)
- Cross-file concerns: inconsistent patterns, duplicated logic, broken contracts between modules

For each finding, assign a severity:
- critical: Must fix. Correctness bug, security hole, data loss risk.
- warning: Should fix. Reliability, maintainability, edge case that will bite eventually.
- suggestion: Consider. Style, naming, minor improvement.

Respond with this exact JSON structure:
{{
  "findings": [
    {{
      "severity": "critical|warning|suggestion",
      "category": "correctness|security|error_handling|edge_cases|naming|consistency|performance|complexity|testing|architecture|documentation|type_safety|concurrency|resource_management",
      "title": "Short title",
      "description": "Detailed explanation of the issue and why it matters",
      "file_path": "path/to/file.py or null",
      "line_range": "42 or 10-20 or null",
      "code_snippet": "relevant code or null",
      "suggestion": "how to fix it or null"
    }}
  ],
  "summary": "One paragraph overall assessment"
}}"""


def build_review_files_message(
    files: list[dict[str, str]],
    regex_findings_json: str,
    memories_json: str | None = None,
    focus_areas: str | None = None,
) -> str:
    """Build the user message for review_files.

    Args:
        files: List of dicts with "path" and "content" keys.
        regex_findings_json: JSON string of automated regex findings.
        memories_json: Optional JSON string of memory context.
        focus_areas: Optional comma-separated focus areas.
    """
    parts = ["## Source Files\n"]

    for f in files:
        # Infer language from extension for syntax highlighting
        ext = f["path"].rsplit(".", 1)[-1] if "." in f["path"] else ""
        lang = {
            "py": "python",
            "ts": "typescript",
            "js": "javascript",
            "rs": "rust",
            "go": "go",
            "java": "java",
            "rb": "ruby",
            "cpp": "cpp",
            "c": "c",
            "sh": "bash",
        }.get(ext, ext)
        parts.append(f"### {f['path']}\n```{lang}\n{f['content']}\n```\n")

    parts.append("## Automated Regex Findings (already confirmed, do not repeat)\n")
    parts.append(f"```json\n{regex_findings_json}\n```\n")

    if memories_json:
        parts.append("## Memory Context (prior decisions and architecture)\n")
        parts.append(f"```json\n{memories_json}\n```\n")

    if focus_areas:
        parts.append(f"## Focus Areas\nPrioritize: {focus_areas}\n")

    parts.append(
        "Now perform your adversarial review of these source files. "
        "Focus on what the regex missed. "
        "Return ONLY the JSON structure specified in your instructions."
    )

    return "\n".join(parts)
