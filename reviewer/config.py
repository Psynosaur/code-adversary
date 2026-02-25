"""Configuration for the adversarial code reviewer."""

from __future__ import annotations

# ── Persona ──────────────────────────────────────────────────────────────────
PERSONA = """You are an adversarial code reviewer — skeptical, thorough, and precise.
You assume nothing is correct until proven otherwise.
You look for what could go wrong, not what looks right.
Your job is to find the bugs, edge cases, and design flaws that the author missed."""

PERSONA_TRAITS = [
    "Skeptical — question every assumption",
    "Thorough — check every code path",
    "Precise — reference specific lines and patterns",
    "Constructive — every criticism includes a fix or direction",
    "Adversarial — think like a malicious input, a race condition, a production failure",
]

# ── Review Focus Areas ───────────────────────────────────────────────────────
DEFAULT_FOCUS_AREAS = [
    "correctness",
    "error_handling",
    "edge_cases",
    "security",
    "naming",
    "consistency",
]

# ── Diff Parsing Patterns ────────────────────────────────────────────────────
# Common anti-patterns to flag in diffs
ANTI_PATTERNS = {
    "bare_except": {
        "pattern": r"except\s*:",
        "severity": "warning",
        "category": "error_handling",
        "title": "Bare except clause",
        "description": "Catches all exceptions including KeyboardInterrupt and SystemExit. Use specific exception types.",
    },
    "todo_fixme": {
        "pattern": r"(?:#|//)\s*(?:TODO|FIXME|HACK|XXX)\b",
        "severity": "suggestion",
        "category": "documentation",
        "title": "TODO/FIXME comment in new code",
        "description": "New code contains a TODO/FIXME marker. This should be resolved or tracked before merging.",
    },
    "hardcoded_secret": {
        "pattern": r"""(?:password|secret|api_key|token|private_key)\s*=\s*['\"][^'\"]{4,}['\"]""",
        "severity": "critical",
        "category": "security",
        "title": "Potential hardcoded secret",
        "description": "A string that looks like a hardcoded credential was detected. Use environment variables or a secrets manager.",
    },
    "print_statement": {
        "pattern": r"\bprint\s*\(",
        "severity": "suggestion",
        "category": "consistency",
        "title": "Print statement in production code",
        "description": "Use a proper logging framework instead of print() for production code.",
    },
    "magic_number": {
        "pattern": r"(?<![_a-zA-Z0-9.])\b(?:[2-9]\d{2,}|[1-9]\d{3,})\b(?!\s*[=:])",
        "severity": "suggestion",
        "category": "naming",
        "title": "Magic number",
        "description": "Large numeric literal without a named constant. Extract to a well-named constant for readability.",
    },
    "hardcoded_numeric_arg": {
        "pattern": r"(?:limit|timeout|max_retries|threshold|size|count|interval|delay|retries|batch_size|page_size|max_size|min_size|capacity|buffer_size|chunk_size|pool_size|workers|attempts)\s*[=:]\s*\d+",
        "severity": "suggestion",
        "category": "naming",
        "title": "Hardcoded numeric parameter",
        "description": "A numeric value is assigned directly to a meaningful parameter name. Extract to a named constant or config value for clarity and single-point-of-change.",
    },
    "nested_function_def": {
        "pattern": r"^\s{8,}(?:async\s+)?def\s+\w+",
        "severity": "suggestion",
        "category": "complexity",
        "title": "Deeply nested function definition",
        "description": "Function defined at a deep indentation level. Consider extracting to module or class level for testability.",
    },
    "return_none_implicit": {
        "pattern": r"^\s*return\s*$",
        "severity": "suggestion",
        "category": "type_safety",
        "title": "Implicit None return",
        "description": "Bare 'return' with no value. If the function intentionally returns None, make it explicit with 'return None' for readability.",
    },
    "string_concatenation_in_loop": {
        "pattern": r"(?:for|while)\s+.*:[\s\S]*?\+=\s*['\"]|(?:for|while)\s+.*:[\s\S]*?\+=\s*(?:str|f['\"])",
        "severity": "suggestion",
        "category": "performance",
        "title": "String concatenation in loop",
        "description": "String concatenation with += in a loop creates a new string each iteration. Use ''.join() or a list for O(n) instead of O(n^2).",
    },
    "empty_except": {
        "pattern": r"except\s+\w+.*:\s*\n\s*pass",
        "severity": "warning",
        "category": "error_handling",
        "title": "Swallowed exception",
        "description": "Exception is caught and silently ignored. At minimum, log the error.",
    },
    "sql_format_string": {
        "pattern": r"""(?:execute|cursor\.execute)\s*\(\s*f['\"]|\.format\s*\(.*(?:SELECT|INSERT|UPDATE|DELETE)""",
        "severity": "critical",
        "category": "security",
        "title": "Potential SQL injection",
        "description": "SQL query constructed with string formatting. Use parameterized queries instead.",
    },
    "eval_exec": {
        "pattern": r"\b(?:eval|exec)\s*\(",
        "severity": "critical",
        "category": "security",
        "title": "Use of eval()/exec()",
        "description": "eval() and exec() execute arbitrary code and are a security risk. Find an alternative approach.",
    },
    "mutable_default": {
        "pattern": r"def\s+\w+\s*\([^)]*=\s*(?:\[\]|\{\}|set\(\))",
        "severity": "warning",
        "category": "correctness",
        "title": "Mutable default argument",
        "description": "Mutable default arguments are shared between calls. Use None as default and create inside the function.",
    },
    "broad_file_permission": {
        "pattern": r"chmod\s+(?:777|666)|os\.chmod\s*\([^)]*0o?(?:777|666)",
        "severity": "critical",
        "category": "security",
        "title": "Overly permissive file permissions",
        "description": "World-writable file permissions are a security risk. Use the most restrictive permissions possible.",
    },
    "assert_in_production": {
        "pattern": r"^\s*assert\s+",
        "severity": "suggestion",
        "category": "error_handling",
        "title": "Assert statement in non-test code",
        "description": "Assert statements are stripped when Python runs with -O flag. Use proper validation with exceptions.",
    },
    "wildcard_import": {
        "pattern": r"from\s+\S+\s+import\s+\*",
        "severity": "warning",
        "category": "consistency",
        "title": "Wildcard import",
        "description": "Wildcard imports pollute the namespace and make it unclear where names come from. Import explicitly.",
    },
    "untyped_exception": {
        "pattern": r"raise\s+Exception\s*\(",
        "severity": "suggestion",
        "category": "error_handling",
        "title": "Generic Exception raised",
        "description": "Raising bare Exception makes it hard for callers to handle specific errors. Use a custom or specific exception type.",
    },
}

# ── Structural Checks ────────────────────────────────────────────────────────
# Things to check beyond regex patterns
STRUCTURAL_CHECKS = [
    "Functions longer than 50 lines",
    "Deeply nested code (> 4 levels)",
    "Missing error handling on I/O operations",
    "Inconsistent return types within a function",
    "Missing type hints on public functions",
    "Unused imports in changed files",
    "Large files (> 500 lines) without clear separation of concerns",
]

# ── Challenge Decision Defaults ──────────────────────────────────────────────
CHALLENGE_ANGLES = [
    "What happens at 10x scale?",
    "What if this input is malicious?",
    "What if this dependency fails or is slow?",
    "What are the alternatives that were not considered?",
    "What are the hidden coupling points?",
    "What happens when this needs to be changed in 6 months?",
    "Is this solving the right problem?",
]

# ── Server Config ────────────────────────────────────────────────────────────
SERVER_NAME = "code-reviewer-mcp"
SERVER_VERSION = "0.2.0"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8088

# ── Bedrock Config ───────────────────────────────────────────────────────────
BEDROCK_PROFILE = "bedrock"
BEDROCK_REGION = "eu-west-1"
BEDROCK_MODEL_ID = "eu.anthropic.claude-sonnet-4-6"
BEDROCK_MAX_TOKENS = 8192

# ── Diff Chunking ────────────────────────────────────────────────────────────
# Max chars of diff text per LLM call (~4 chars per token, leaves room for
# system prompt, regex findings, and response tokens)
DIFF_CHUNK_MAX_CHARS = 24_000

# ── Usage Logging ────────────────────────────────────────────────────────────
USAGE_LOG_PATH = "usage.log"

# ── Audit Logging ────────────────────────────────────────────────────────────
# One JSON file per LLM invocation — captures full prompts, responses, and metadata.
# Files named: {timestamp}_{tool}.json
AUDIT_LOG_DIR = "audit"
