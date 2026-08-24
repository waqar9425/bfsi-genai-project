"""
Milestone 12: MCP client bootstrap.

Fetches tool objects from the Argus MCP server (mcp_server.py, launched
as a stdio subprocess) at IMPORT time -- one synchronous call per
specialist file, matching the existing "build once at module load, reuse
every request" pattern used for LLM clients (llm.py) and the harness
throughout this project.

Known limitation, named not hidden: each specialist calling this
independently spins up its OWN short-lived server subprocess just for
tool discovery -- four separate discovery round-trips instead of one
shared bootstrap. A one-time startup cost, not a per-request one; fine
for a learning project, worth sharing a client if this were headed to
production.

Second known limitation: `asyncio.run()` at import time requires that
nothing importing this module is ALREADY inside a running event loop.
True for every entry point in this project so far (plain scripts, pytest).
Will matter again once Milestone 13 wraps this in a FastAPI app --
imports happening inside an async framework's startup would need a
different bootstrap strategy (fetch inside an async startup hook instead
of at bare import time).
"""

import asyncio
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

_SERVER_SCRIPT = str(Path(__file__).parent / "mcp_server.py")


async def _fetch_tools() -> list:
    client = MultiServerMCPClient(
        {
            "argus": {
                "command": sys.executable,
                "args": [_SERVER_SCRIPT],
                "transport": "stdio",
            }
        }
    )
    return await client.get_tools()


def get_mcp_tools(names: list[str] | None = None) -> list:
    """Synchronously fetch tool objects from the Argus MCP server.
    names: if given, filter down to only these tool names -- a specialist
    only wants ITS tools, not every tool on the server.
    """
    all_tools = asyncio.run(_fetch_tools())
    if names is None:
        return all_tools
    return [t for t in all_tools if t.name in names]
