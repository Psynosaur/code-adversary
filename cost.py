#!/usr/bin/env python3
"""Parse usage.log and estimate AWS Bedrock costs.

Supports multiple models with different pricing tiers.

Usage:
  python cost.py                # reads usage.log in current directory
  python cost.py /path/to/usage.log
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# ── Model Pricing (USD per token) ────────────────────────────────────────────


@dataclass
class ModelPricing:
    label: str
    input_per_token: float
    output_per_token: float


MODELS: dict[str, ModelPricing] = {
    # Opus 4.6 — $5/M input, $25/M output
    "eu.anthropic.claude-opus-4-6-v1": ModelPricing(
        "Opus 4.6", 5.00 / 1_000_000, 25.00 / 1_000_000
    ),
    "anthropic.claude-opus-4-6-v1": ModelPricing(
        "Opus 4.6", 5.00 / 1_000_000, 25.00 / 1_000_000
    ),
    # Sonnet 4.6 — $3/M input, $15/M output
    "eu.anthropic.claude-sonnet-4-6": ModelPricing(
        "Sonnet 4.6", 3.00 / 1_000_000, 15.00 / 1_000_000
    ),
    "anthropic.claude-sonnet-4-6": ModelPricing(
        "Sonnet 4.6", 3.00 / 1_000_000, 15.00 / 1_000_000
    ),
}

_FALLBACK = ModelPricing("unknown", 3.00 / 1_000_000, 15.00 / 1_000_000)


def _get_pricing(model_id: str) -> ModelPricing:
    return MODELS.get(model_id, _FALLBACK)


# ── Stats ────────────────────────────────────────────────────────────────────


@dataclass
class Stats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

    @property
    def avg_latency_s(self) -> float:
        return (self.total_latency_ms / self.calls / 1000) if self.calls else 0

    def add(self, inp: int, out: int, latency: int, model: str) -> None:
        pricing = _get_pricing(model)
        self.calls += 1
        self.input_tokens += inp
        self.output_tokens += out
        self.input_cost += inp * pricing.input_per_token
        self.output_cost += out * pricing.output_per_token
        self.total_latency_ms += latency


def parse_usage_log(
    path: Path,
) -> tuple[
    dict[str, Stats],  # per-tool
    dict[str, Stats],  # per-model
    Stats,  # grand total
]:
    """Parse TSV usage log into per-tool and per-model stats."""
    by_tool: dict[str, Stats] = {}
    by_model: dict[str, Stats] = {}
    grand = Stats()

    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("timestamp"):
                continue

            parts = line.split("\t")
            if len(parts) < 7:
                print(
                    f"  [skip] line {line_no}: expected 7+ columns, got {len(parts)}",
                    file=sys.stderr,
                )
                continue

            _ts, model, tool, inp, out, _total, latency = parts[:7]

            try:
                inp_tokens = int(inp)
                out_tokens = int(out)
                lat_ms = int(latency)
            except ValueError:
                print(
                    f"  [skip] line {line_no}: non-numeric token/latency values",
                    file=sys.stderr,
                )
                continue

            if tool not in by_tool:
                by_tool[tool] = Stats()
            by_tool[tool].add(inp_tokens, out_tokens, lat_ms, model)

            pricing = _get_pricing(model)
            label = pricing.label
            if label not in by_model:
                by_model[label] = Stats()
            by_model[label].add(inp_tokens, out_tokens, lat_ms, model)

            grand.add(inp_tokens, out_tokens, lat_ms, model)

    return by_tool, by_model, grand


def _print_table(title: str, stats: dict[str, Stats]) -> None:
    """Print a stats table with the given title."""
    print(f"  {title}")
    print(
        f"  {'Name':<30} {'Calls':>5} {'Input':>9} {'Output':>9} {'Total':>9} {'In $':>8} {'Out $':>8} {'Cost $':>9} {'Avg lat':>8}"
    )
    print("  " + "-" * 107)

    for name, s in sorted(stats.items()):
        print(
            f"  {name:<30} {s.calls:>5} {s.input_tokens:>9,} {s.output_tokens:>9,} "
            f"{s.total_tokens:>9,} {s.input_cost:>8.4f} {s.output_cost:>8.4f} "
            f"{s.total_cost:>9.4f} {s.avg_latency_s:>7.1f}s"
        )


def print_report(
    by_tool: dict[str, Stats],
    by_model: dict[str, Stats],
    grand: Stats,
) -> None:
    """Print a formatted cost report with per-model and per-tool breakdowns."""
    if grand.calls == 0:
        print("No usage data found.")
        return

    # Model pricing reference
    print("  Model pricing:")
    seen = set()
    for model_id, p in MODELS.items():
        if p.label in seen:
            continue
        seen.add(p.label)
        print(
            f"    {p.label:<20} input=${p.input_per_token * 1_000_000:.2f}/M  output=${p.output_per_token * 1_000_000:.2f}/M"
        )
    print()

    # Per-model breakdown
    _print_table("By Model", by_model)
    print()

    # Per-tool breakdown
    _print_table("By Tool", by_tool)
    print()

    # Grand total
    print("  " + "=" * 107)
    print(
        f"  {'TOTAL':<30} {grand.calls:>5} {grand.input_tokens:>9,} {grand.output_tokens:>9,} "
        f"{grand.total_tokens:>9,} {grand.input_cost:>8.4f} {grand.output_cost:>8.4f} "
        f"{grand.total_cost:>9.4f} {grand.avg_latency_s:>7.1f}s"
    )
    print()

    # Summary
    print(f"  Input tokens:  {grand.input_tokens:>10,}  (${grand.input_cost:.4f})")
    print(f"  Output tokens: {grand.output_tokens:>10,}  (${grand.output_cost:.4f})")
    print(f"  Total tokens:  {grand.total_tokens:>10,}")
    print(f"  Total cost:    ${grand.total_cost:.4f}")
    print(f"  Total calls:   {grand.calls}")
    print(f"  Total time:    {grand.total_latency_ms / 1000:.1f}s")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("usage.log")
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"  Parsing: {path}")
    print()

    by_tool, by_model, grand = parse_usage_log(path)
    print_report(by_tool, by_model, grand)


if __name__ == "__main__":
    main()
