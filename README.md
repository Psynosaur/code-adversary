# code-reviewer-mcp

Adversarial code reviewer exposed as an MCP server. Two-pass review: deterministic regex checks followed by LLM-powered analysis via AWS Bedrock (Claude Opus 4.6). Designed to run alongside an AI coding agent and scrutinize its work.

## How it works

```
                          code-reviewer-mcp
                    +--------------------------+
   git diff ------->|  PASS 1: Regex (free)    |
   memories ------->|  19 anti-patterns        |
                    |  structural checks       |
                    +-----------+--------------+
                                |
                    +-----------v--------------+
                    |  PASS 2: LLM (Bedrock)   |
                    |  adversarial persona     |
                    |  receives diff + regex   |
                    |  findings as context     |
                    +-----------+--------------+
                                |
                    +-----------v--------------+
                    |  Merged report           |
                    |  sorted by severity      |
   <----------------|  JSON response           |
                    +--------------------------+
```

Pass 1 (regex) runs in milliseconds, costs nothing, and catches objective patterns (hardcoded secrets, eval, SQL injection, bare except). Pass 2 (LLM) receives the diff plus the regex findings and focuses on what regex missed: logic errors, design flaws, edge cases, missing error handling.

Severity levels:
- **critical** -- must fix (security, correctness, data loss)
- **warning** -- should fix (reliability, maintainability, edge cases)
- **suggestion** -- consider (style, naming, minor improvements)

## Tools

| Tool | Args | Description |
|------|------|-------------|
| `review_diff` | `diff` (str), `memories` (str, optional), `focus_areas` (str, optional) | Main review tool. Two-pass: regex + LLM adversarial analysis of a unified diff. |
| `review_files` | `file_paths` (JSON array str), `file_contents` (JSON array str), `memories` (str, optional), `focus_areas` (str, optional) | Adversarial review of complete source files (not diffs). Same two-pass approach. Use for auditing existing code. |
| `review_pattern` | `pattern_description` (str), `code_snippet` (str) | Scrutinize a specific code pattern against best practices. Two-pass: regex + LLM. |
| `challenge_decision` | `decision` (str), `reasoning` (str), `alternatives` (str, optional) | LLM-powered adversarial challenge of an architectural decision. |
| `get_persona` | -- | Returns the reviewer persona and behavioral traits. |

## Regex anti-patterns (19)

bare_except, todo_fixme, hardcoded_secret, print_statement, magic_number, hardcoded_numeric_arg, empty_except, sql_format_string, eval_exec, mutable_default, broad_file_permission, assert_in_production, wildcard_import, untyped_exception, nested_function_def, return_none_implicit, string_concatenation_in_loop

## Structural checks

- Functions longer than 50 lines
- New files larger than 500 lines
- I/O operations without error handling
- Missing return type annotations on Python functions
- Duplicate/redundant function calls within same block
- Missing input validation on public functions
- Mixed I/O and business logic in same function
- Memory context cross-referencing (contradictions with prior decisions)

## Project structure

```
code-advesary/
  server.py              # FastMCP entry point (streamable HTTP, port 8088)
  reviewer/
    __init__.py
    config.py            # Persona, anti-patterns, Bedrock config, server config
    models.py            # Severity, Category enums, ReviewFinding, ReviewReport
    mcp_tools.py         # 5 MCP tool definitions + progress bridge
    analyzer.py          # Orchestrator: review_diff, review_files, review_pattern, challenge_decision
    diff_parser.py       # DiffHunk, DiffFile dataclasses, parse_diff(), diff_stats()
    chunking.py          # Diff and source file chunking for LLM context windows
    checks.py            # 19 regex anti-patterns + structural checks
    llm_parsing.py       # LLM JSON response parsing + partial-response salvage
    llm.py               # Bedrock client (boto3 Session, AWS profile), token usage logging
    prompts.py           # Adversarial system prompts and message builders
  pyproject.toml
  requirements.txt
  .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires an AWS profile named `bedrock` with access to Claude on Bedrock in `eu-west-1`:

```bash
aws configure --profile bedrock
```

## Run

```bash
.venv/bin/python server.py
```

Starts a streamable HTTP MCP server on `http://0.0.0.0:8088/mcp`.

Server logs (stderr) show inference calls with token usage:

```
10:23:05 reviewer.llm INFO Bedrock client initialized: profile=bedrock region=eu-west-1
10:23:31 reviewer.llm INFO Bedrock usage [review_diff]: input=1344 output=1426 total=2770 latency=26113ms
```

Token usage is also appended to `usage.log` (TSV format).

## OpenCode MCP config

Add to your `opencode.json` (global or project-level):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "code-reviewer": {
      "type": "remote",
      "url": "http://localhost:8088/mcp",
      "enabled": true,
      "timeout": 300000
    }
  }
}
```

The `timeout` is set to 5 minutes (300000ms) because Bedrock LLM inference can take 60-90 seconds per chunk. Without it, OpenCode's default timeout will kill the tool call before the review completes.

## OpenCode plugin integration

An OpenCode plugin at `~/.config/opencode/plugins/code-reviewer.ts` enforces the review workflow automatically. The plugin works together with the long-term-memory plugin:

```
  Memory Plugin                     Review Plugin
       |                                 |
       |  1. Agent edits files           |
       |                                 +-- tracks filesEdited (passive)
       |                                 |
       |  2. Memory plugin enforces      |
       |     long-term-memory_remember   |
       |          |                      |
       |          +--------------------->+-- detects memory store after edits
       |                                 |-- review done? no -> sets reminder
       |                                 |
       |  3. System prompt tells agent   |
       |     to run git diff + call      |
       |     code-reviewer_review_diff   |
       |                                 |
       |  4. Agent calls review_diff --->+-- marks reviewCalled = true
       |                                 |
       |  5. Agent stores review         |
       |     findings in memory          |
       |     (tag: "review")             |
```

The review is NOT triggered on every file edit. The memory store is the signal that work is done.

Plugin hooks:
1. **System prompt injection** -- tells agent about review workflow in every LLM call
2. **tool.execute.after** -- tracks file edits and review tool calls
3. **tool.execute.before** -- detects memory stores after edits, triggers review reminder
4. **Compaction** -- preserves review state across session compaction
5. **Session cleanup** -- cleans up state on session delete

## Tool use examples

### review_diff

The primary workflow. Pass a git diff and optionally recent memories as JSON context.

```
Review my recent changes for issues. Here is the diff:

<git diff output>

And here is the recent memory context:

<JSON array of memory objects>

Use the code-reviewer review_diff tool.
```

The tool returns a JSON report:

```json
{
  "verdict": "CHANGES REQUESTED -- critical issues must be addressed",
  "summary": "Reviewed 1 file(s): +15/-0 lines. Found 3 critical, 6 warning(s), 3 suggestion(s). [Regex: 5, LLM: 7]",
  "stats": {
    "critical": 3,
    "warnings": 6,
    "suggestions": 3,
    "total": 12
  },
  "findings": [
    {
      "severity": "critical",
      "category": "security",
      "title": "Potential hardcoded secret",
      "description": "A string that looks like a hardcoded credential was detected.",
      "file_path": "app.py",
      "line_range": "4",
      "code_snippet": "password = \"hunter2\""
    }
  ]
}
```

### review_pattern

Scrutinize a specific piece of code without a full diff.

```
I'm using this retry pattern in my API client. Check it for issues.

Use the code-reviewer review_pattern tool with:
- pattern_description: "Exponential backoff retry for HTTP requests"
- code_snippet: <the code>
```

### review_files

Audit existing source files without needing a diff. Pass file paths and contents as parallel JSON arrays.

```
Review these files for security issues and design flaws.

Use the code-reviewer review_files tool with:
- file_paths: '["src/auth.py", "src/database.py"]'
- file_contents: '["<contents of auth.py>", "<contents of database.py>"]'
- focus_areas: "security,error_handling"
```

Returns the same JSON report format as `review_diff`.

### challenge_decision

Stress-test an architectural decision.

```
Challenge this decision using the code-reviewer challenge_decision tool:
- decision: "Use SQLite for the memory database"
- reasoning: "Single-user local app, no need for a server database"
- alternatives: "PostgreSQL, Redis, flat JSON files"
```

Returns specific adversarial challenges based on the decision content, not generic templates.

## Requirements

- Python >= 3.12
- fastmcp >= 2.0.0
- uvicorn >= 0.34.0
- sse-starlette >= 2.2.1
- boto3 >= 1.35.0
- AWS profile `bedrock` with Bedrock access in eu-west-1
