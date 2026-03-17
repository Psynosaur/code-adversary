"""Adversarial system prompts and user message builders for each tool.

System prompts are returned as Bedrock content-block arrays so that
the shared persona block can carry ``cache_control`` for prompt caching.

User messages are returned as content-block arrays so that large
static sections (e.g. findings, memories) are separate blocks — cleaner
audit logs and future-ready for per-block caching.
"""

from __future__ import annotations

import json
import logging

from reviewer.config import PERSONA, PERSONA_TRAITS

logger = logging.getLogger(__name__)

# ── Type aliases ─────────────────────────────────────────────────────────────

# Bedrock content block: {"type": "text", "text": "...", ...}
ContentBlock = dict[str, object]

# ── Compact encoding helpers ─────────────────────────────────────────────────


def _encode_findings(findings: list[dict]) -> str:
    """Encode regex findings as compact JSON for LLM input."""
    if not findings:
        return "(none)"
    return json.dumps(findings, separators=(",", ":"))


def _encode_memories(memories_json: str | None) -> str | None:
    """Parse memories JSON and re-encode as compact JSON for LLM input.

    Returns None if the input is empty or malformed (so the caller
    can skip the memory section entirely).

    Handles the memory MCP wrapper format: if the parsed JSON is a dict
    with a "data" key, unwraps to just the data array.
    """
    if not memories_json:
        return None
    try:
        parsed = json.loads(memories_json)
        if isinstance(parsed, list):
            return json.dumps(parsed, separators=(",", ":"))
        elif isinstance(parsed, dict) and "data" in parsed:
            return json.dumps(parsed["data"], separators=(",", ":"))
        return json.dumps(parsed, separators=(",", ":"))
    except json.JSONDecodeError:
        logger.warning("Malformed memories JSON — skipping memory context")
        return None
    except TypeError:
        logger.warning(
            "Non-serializable memories data — skipping memory context",
            exc_info=True,
        )
        return None


# ── Shared persona & output schema ───────────────────────────────────────────

_TRAITS_BLOCK = "\n".join(f"- {t}" for t in PERSONA_TRAITS)

# The persona block is identical across all tools — it gets cache_control
# so Bedrock can cache it (requires ≥1,024 tokens cumulative before the
# checkpoint, so the tool-instruction block is included in the same
# cache-eligible prefix).
_PERSONA_TEXT = f"""{PERSONA}

Your traits:
{_TRAITS_BLOCK}

You must respond ONLY with valid JSON. No markdown, no commentary outside the JSON structure."""

# Full findings JSON schema — shared across review_diff, review_files
_FINDINGS_SCHEMA_FULL = """{
  "findings": [
    {
      "severity": "critical|warning|suggestion",
      "category": "correctness|security|error_handling|edge_cases|naming|consistency|performance|complexity|testing|architecture|documentation|type_safety|concurrency|resource_management",
      "title": "Short title",
      "description": "Detailed explanation of the issue and why it matters",
      "file_path": "path/to/file.py or null",
      "line_range": "42 or 10-20 or null",
      "code_snippet": "relevant code or null",
      "suggestion": "how to fix it or null"
    }
  ],
  "summary": "One paragraph overall assessment"
}"""

# Compact findings schema — for review_pattern, challenge_decision
_FINDINGS_SCHEMA_COMPACT = """{
  "findings": [
    {
      "severity": "critical|warning|suggestion",
      "category": "correctness|security|error_handling|edge_cases|naming|consistency|performance|complexity|testing|architecture|documentation|type_safety|concurrency|resource_management",
      "title": "Short title",
      "description": "Detailed explanation",
      "code_snippet": "relevant code or null",
      "suggestion": "how to fix it or null"
    }
  ],
  "summary": "One paragraph assessment"
}"""

_CHALLENGE_SCHEMA = """{
  "findings": [
    {
      "severity": "critical|warning|suggestion",
      "category": "architecture|security|performance|correctness|complexity",
      "title": "Short title",
      "description": "Specific challenge with reference to the decision/reasoning",
      "suggestion": "what to consider or do differently, or null"
    }
  ],
  "summary": "One paragraph adversarial assessment"
}"""


def _system_blocks(tool_instructions: str, schema: str) -> list[ContentBlock]:
    """Build the system prompt as a Bedrock content-block array.

    Block 1: persona + traits (shared, cache-eligible)
    Block 2: tool-specific instructions + output schema

    A cache_control checkpoint is placed on the end of block 2 so the
    entire system prompt prefix is cached once cumulative tokens ≥ 1,024.
    """
    return [
        {"type": "text", "text": _PERSONA_TEXT},
        {
            "type": "text",
            "text": f"{tool_instructions}\n\nRespond with this exact JSON structure:\n{schema}",
            "cache_control": {"type": "ephemeral"},
        },
    ]


# ── review_diff ──────────────────────────────────────────────────────────────

_REVIEW_DIFF_INSTRUCTIONS = """You are reviewing a git diff. You receive:
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
- suggestion: Consider. Style, naming, minor improvement."""

REVIEW_DIFF_SYSTEM = _system_blocks(_REVIEW_DIFF_INSTRUCTIONS, _FINDINGS_SCHEMA_FULL)


def build_review_diff_message(
    diff_text: str,
    regex_findings: list[dict],
    memories_json: str | None = None,
) -> list[ContentBlock]:
    """Build the user message as content blocks for review_diff."""
    findings_json = _encode_findings(regex_findings)
    blocks: list[ContentBlock] = [
        {"type": "text", "text": f"## Git Diff\n\n```diff\n{diff_text}\n```"},
        {
            "type": "text",
            "text": (
                "## Automated Regex Findings (already confirmed, do not repeat)\n\n"
                f"```json\n{findings_json}\n```"
            ),
        },
    ]

    memories_compact = _encode_memories(memories_json)
    if memories_compact:
        blocks.append(
            {
                "type": "text",
                "text": (
                    "## Memory Context (prior decisions and architecture)\n\n"
                    f"```json\n{memories_compact}\n```"
                ),
            }
        )

    blocks.append(
        {
            "type": "text",
            "text": (
                "Now perform your adversarial review. Focus on what the regex missed. "
                "Return ONLY the JSON structure specified in your instructions."
            ),
        }
    )

    return blocks


# ── review_pattern ───────────────────────────────────────────────────────────

_REVIEW_PATTERN_INSTRUCTIONS = """You are scrutinizing a specific code pattern or implementation. You receive:
1. A description of what the pattern is meant to accomplish
2. The code implementing it
3. Automated regex findings (confirmed, do not repeat)

Your job:
- Does this pattern actually achieve its stated goal? Prove it or break it.
- What inputs would cause it to fail, return wrong results, or behave unexpectedly?
- Are there simpler, more robust, or more idiomatic alternatives?
- Is it testable? How would you test the edge cases?
- Does it handle failure modes (timeouts, exceptions, partial state)?"""

REVIEW_PATTERN_SYSTEM = _system_blocks(
    _REVIEW_PATTERN_INSTRUCTIONS, _FINDINGS_SCHEMA_COMPACT
)


def build_review_pattern_message(
    pattern_description: str,
    code_snippet: str,
    regex_findings: list[dict],
) -> list[ContentBlock]:
    """Build the user message as content blocks for review_pattern."""
    findings_json = _encode_findings(regex_findings)
    return [
        {
            "type": "text",
            "text": f"## Pattern Description\n{pattern_description}\n\n## Code\n```\n{code_snippet}\n```",
        },
        {
            "type": "text",
            "text": (
                "## Automated Regex Findings (already confirmed, do not repeat)\n\n"
                f"```json\n{findings_json}\n```"
            ),
        },
        {
            "type": "text",
            "text": "Now scrutinize this pattern. Return ONLY the JSON structure specified.",
        },
    ]


# ── challenge_decision ───────────────────────────────────────────────────────

_CHALLENGE_DECISION_INSTRUCTIONS = """You are adversarially challenging an architectural or design decision. You receive:
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
- suggestion: Worth thinking about but not blocking."""

CHALLENGE_DECISION_SYSTEM = _system_blocks(
    _CHALLENGE_DECISION_INSTRUCTIONS, _CHALLENGE_SCHEMA
)


def build_challenge_decision_message(
    decision: str,
    reasoning: str,
    alternatives: str | None = None,
) -> list[ContentBlock]:
    """Build the user message as content blocks for challenge_decision."""
    alt_text = alternatives if alternatives else "None provided."
    return [
        {
            "type": "text",
            "text": f"## Decision\n{decision}\n\n## Reasoning\n{reasoning}\n\n## Alternatives Considered\n{alt_text}",
        },
        {
            "type": "text",
            "text": "Challenge this decision adversarially. Return ONLY the JSON structure specified.",
        },
    ]


# ── review_files ─────────────────────────────────────────────────────────────

_REVIEW_FILES_INSTRUCTIONS = """You are reviewing source files in their entirety (not a diff). You receive:
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
- suggestion: Consider. Style, naming, minor improvement."""

REVIEW_FILES_SYSTEM = _system_blocks(_REVIEW_FILES_INSTRUCTIONS, _FINDINGS_SCHEMA_FULL)

# Language extension map for syntax-highlighted code blocks
_EXT_TO_LANG = {
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
}


def build_review_files_message(
    files: list[dict[str, str]],
    regex_findings: list[dict],
    memories_json: str | None = None,
    focus_areas: str | None = None,
) -> list[ContentBlock]:
    """Build the user message as content blocks for review_files.

    Args:
        files: List of dicts with "path" and "content" keys.
        regex_findings: List of regex finding dicts.
        memories_json: Optional JSON string of memory context.
        focus_areas: Optional comma-separated focus areas.
    """
    # Build source file text — one block for all files
    file_parts = ["## Source Files\n"]
    for f in files:
        ext = f["path"].rsplit(".", 1)[-1] if "." in f["path"] else ""
        lang = _EXT_TO_LANG.get(ext, ext)
        file_parts.append(f"### {f['path']}\n```{lang}\n{f['content']}\n```\n")

    blocks: list[ContentBlock] = [
        {"type": "text", "text": "\n".join(file_parts)},
    ]

    findings_json = _encode_findings(regex_findings)
    blocks.append(
        {
            "type": "text",
            "text": (
                "## Automated Regex Findings (already confirmed, do not repeat)\n\n"
                f"```json\n{findings_json}\n```"
            ),
        }
    )

    memories_compact = _encode_memories(memories_json)
    if memories_compact:
        blocks.append(
            {
                "type": "text",
                "text": (
                    "## Memory Context (prior decisions and architecture)\n\n"
                    f"```json\n{memories_compact}\n```"
                ),
            }
        )

    if focus_areas:
        blocks.append(
            {"type": "text", "text": f"## Focus Areas\nPrioritize: {focus_areas}"}
        )

    blocks.append(
        {
            "type": "text",
            "text": (
                "Now perform your adversarial review of these source files. "
                "Focus on what the regex missed. "
                "Return ONLY the JSON structure specified in your instructions."
            ),
        }
    )

    return blocks
