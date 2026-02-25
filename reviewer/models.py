"""Data models for the code reviewer MCP server."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class Severity(str, Enum):
    """Severity levels for review findings."""

    CRITICAL = "critical"  # Must fix — correctness, security, data loss
    WARNING = "warning"  # Should fix — reliability, maintainability, edge cases
    SUGGESTION = "suggestion"  # Consider — style, naming, minor improvements

    def __str__(self) -> str:
        return self.value


class Category(str, Enum):
    """Categories for review findings."""

    CORRECTNESS = "correctness"
    SECURITY = "security"
    ERROR_HANDLING = "error_handling"
    EDGE_CASES = "edge_cases"
    NAMING = "naming"
    CONSISTENCY = "consistency"
    PERFORMANCE = "performance"
    COMPLEXITY = "complexity"
    TESTING = "testing"
    ARCHITECTURE = "architecture"
    DOCUMENTATION = "documentation"
    TYPE_SAFETY = "type_safety"
    CONCURRENCY = "concurrency"
    RESOURCE_MANAGEMENT = "resource_management"

    def __str__(self) -> str:
        return self.value


@dataclass
class ReviewFinding:
    """A single finding from the adversarial code review."""

    severity: Severity
    category: Category
    title: str
    description: str
    file_path: Optional[str] = None
    line_range: Optional[str] = None  # e.g., "12-18" or "42"
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = str(self.severity)
        d["category"] = str(self.category)
        # Drop None values for cleaner output
        return {k: v for k, v in d.items() if v is not None and v != []}


@dataclass
class ReviewReport:
    """Complete review report containing all findings."""

    findings: list[ReviewFinding] = field(default_factory=list)
    summary: str = ""
    reviewed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    diff_stats: Optional[dict] = None
    memory_context_used: bool = False

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.WARNING)

    @property
    def suggestion_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.SUGGESTION)

    @property
    def verdict(self) -> str:
        if self.critical_count > 0:
            return "CHANGES REQUESTED — critical issues must be addressed"
        if self.warning_count > 2:
            return "NEEDS ATTENTION — multiple warnings found"
        if self.warning_count > 0:
            return "ACCEPTABLE WITH RESERVATIONS — warnings should be reviewed"
        if self.suggestion_count > 0:
            return "APPROVED — minor suggestions only"
        return "APPROVED — no issues found"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "summary": self.summary,
            "stats": {
                "critical": self.critical_count,
                "warnings": self.warning_count,
                "suggestions": self.suggestion_count,
                "total": len(self.findings),
            },
            "findings": [f.to_dict() for f in self.findings],
            "reviewed_at": self.reviewed_at,
            "diff_stats": self.diff_stats,
            "memory_context_used": self.memory_context_used,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
