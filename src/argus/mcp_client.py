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

`asyncio.run()` at import time requires that nothing importing this
module is ALREADY inside a running event loop -- resolved for real in
Milestone 13 (api.py imports at true top-level, before uvicorn's loop
exists) and Milestone 14 (eval_gate.py needed the same fix applied again).

Milestone 15 fix, root-caused rather than patched blind: the MCP SDK's
stdio launcher does NOT inherit the parent process's full environment --
by design, a security allowlist (mcp/client/stdio/__init__.py's
DEFAULT_INHERITED_ENV_VARS: just HOME/LOGNAME/PATH/SHELL/TERM/USER on
Linux). This is WHY mcp_server.py needed the sys.path self-sufficiency
fix back in Milestone 12 (PYTHONPATH was never going to be inherited
either) -- same root cause, different symptom, not understood as such
until this milestone's Docker-simulation testing surfaced it for real
(every earlier test had a real .env file on disk, which the subprocess's
OWN load_dotenv() call found independently, masking that inheritance
was never happening). Fixed by passing GOOGLE_API_KEY explicitly via
StdioServerParameters' `env` field, which MERGES ON TOP of the safe
default set rather than replacing it -- deliberately NOT passing
os.environ wholesale, which would defeat the entire point of the
allowlist. Pass only what's actually needed, nothing more.
"""

import asyncio
import os
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
                # Merged ON TOP of the SDK's safe allowlist -- doesn't
                # replace it. The server needs this specifically because
                # tools/policy_tools.py builds an embeddings client at
                # IMPORT time (rag.py), which needs API access to embed
                # search queries -- a real, load-bearing need, not
                # incidental.
                "env": {"GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY", "")},
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
