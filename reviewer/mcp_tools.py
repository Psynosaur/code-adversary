"""MCP tool definitions for the adversarial code reviewer."""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from typing import Optional

from fastmcp import Context, FastMCP

from reviewer.analyzer import (
    challenge_decision as _challenge_decision,
    review_diff as _review_diff,
    review_pattern as _review_pattern,
)
from reviewer.config import PERSONA, PERSONA_TRAITS

logger = logging.getLogger(__name__)


def _error_response(tool_name: str, error: Exception) -> str:
    """Build a structured JSON error response for MCP tool failures."""
    logger.error("Tool %s failed: %s\n%s", tool_name, error, traceback.format_exc())
    return json.dumps(
        {
            "verdict": "ERROR",
            "summary": f"Tool '{tool_name}' failed: {error}",
            "stats": {"critical": 0, "warnings": 0, "suggestions": 0, "total": 0},
            "findings": [],
            "error": str(error),
        },
        indent=2,
    )


def _make_progress_bridge(ctx: Context, loop: asyncio.AbstractEventLoop):
    """Create a sync callback that sends MCP progress notifications.

    The callback is called from the blocking LLM streaming thread.
    It uses run_coroutine_threadsafe to bridge to the async MCP context.
    """
    call_count = 0

    def on_progress(chars_so_far: int, elapsed: float, message: str) -> None:
        nonlocal call_count
        call_count += 1
        try:
            future = asyncio.run_coroutine_threadsafe(
                ctx.report_progress(
                    progress=chars_so_far,
                    total=None,
                    message=message,
                ),
                loop,
            )
            # Don't block long -- just fire and forget with a short timeout
            future.result(timeout=1.0)
        except Exception:
            pass  # Never let progress errors kill the review

    return on_progress


def register_tools(mcp: FastMCP) -> None:
    """Register all review tools on the given FastMCP server instance."""

    @mcp.tool()
    async def review_diff(
        diff: str,
        ctx: Context,
        memories: Optional[str] = None,
        focus_areas: Optional[str] = None,
    ) -> str:
        """Adversarial code review of a git diff.

        Takes a unified diff (from `git diff`) and optionally recent memory
        context (as a JSON string) to cross-reference against prior decisions.

        Returns a structured review report with findings sorted by severity.

        Args:
            diff: The unified diff output (e.g., from `git diff` or `git diff HEAD~1`)
            memories: Optional JSON string of recent memory objects for context
            focus_areas: Optional comma-separated areas to focus on
                         (e.g., "security,error_handling,correctness")
        """
        try:
            loop = asyncio.get_running_loop()
            on_progress = _make_progress_bridge(ctx, loop)

            report = await asyncio.to_thread(
                _review_diff,
                diff_text=diff,
                memories_json=memories,
                focus_areas=focus_areas,
                on_progress=on_progress,
            )
            return report.to_json()
        except Exception as e:
            return _error_response("review_diff", e)

    @mcp.tool()
    async def review_pattern(
        pattern_description: str,
        code_snippet: str,
        ctx: Context,
    ) -> str:
        """Scrutinize a specific code pattern or implementation against best practices.

        Use this to get adversarial feedback on a particular pattern, algorithm,
        or approach — independent of a full diff.

        Args:
            pattern_description: What the pattern is meant to accomplish
            code_snippet: The code implementing the pattern
        """
        try:
            loop = asyncio.get_running_loop()
            on_progress = _make_progress_bridge(ctx, loop)

            report = await asyncio.to_thread(
                _review_pattern,
                pattern_description=pattern_description,
                code_snippet=code_snippet,
                on_progress=on_progress,
            )
            return report.to_json()
        except Exception as e:
            return _error_response("review_pattern", e)

    @mcp.tool()
    async def challenge_decision(
        decision: str,
        reasoning: str,
        ctx: Context,
        alternatives: Optional[str] = None,
    ) -> str:
        """Adversarial challenge of an architectural or design decision.

        Generates structured challenge questions from multiple angles to stress-test
        the decision's soundness.

        Args:
            decision: The decision that was made (e.g., "Use SQLite for storage")
            reasoning: Why this decision was made
            alternatives: Optional comma-separated alternatives that were considered
                          (e.g., "PostgreSQL, Redis, flat files")
        """
        try:
            loop = asyncio.get_running_loop()
            on_progress = _make_progress_bridge(ctx, loop)

            report = await asyncio.to_thread(
                _challenge_decision,
                decision=decision,
                reasoning=reasoning,
                alternatives=alternatives,
                on_progress=on_progress,
            )
            return report.to_json()
        except Exception as e:
            return _error_response("challenge_decision", e)

    @mcp.tool()
    def get_persona() -> str:
        """Get the reviewer's persona description and traits.

        Returns the adversarial reviewer's persona and behavioral traits
        so the calling agent knows what to expect.
        """
        lines = [PERSONA, "", "Traits:"]
        for trait in PERSONA_TRAITS:
            lines.append(f"  - {trait}")
        return "\n".join(lines)
