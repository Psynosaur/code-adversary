"""FastMCP entry point for the adversarial code reviewer."""

from __future__ import annotations

import logging

from fastmcp import FastMCP

from reviewer.config import SERVER_HOST, SERVER_NAME, SERVER_PORT, SERVER_VERSION
from reviewer.mcp_tools import register_tools

# ── Logging ──────────────────────────────────────────────────────────────────
# Configure logging for all reviewer.* modules so inference calls,
# token usage, and errors show up in the server's stderr output.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
# Keep boto noise at WARNING
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def create_server() -> FastMCP:
    """Create and configure the FastMCP server instance."""
    mcp = FastMCP(
        name=SERVER_NAME,
        version=SERVER_VERSION,
    )
    register_tools(mcp)
    return mcp


# Module-level server instance (used by FastMCP CLI and stdio transport)
mcp = create_server()


def main() -> None:
    """Entry point — runs as streamable HTTP MCP server."""
    logging.getLogger(__name__).info(
        "Starting %s v%s on %s:%d",
        SERVER_NAME,
        SERVER_VERSION,
        SERVER_HOST,
        SERVER_PORT,
    )
    try:
        mcp.run(
            transport="streamable-http",
            host=SERVER_HOST,
            port=SERVER_PORT,
        )
    except OSError as e:
        logging.getLogger(__name__).error("Server failed to start: %s", e)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
