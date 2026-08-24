# Milestone 15: containerizes the FastAPI app (api.py) AND the MCP
# server (mcp_server.py) TOGETHER, in one image -- not two. This is a
# direct consequence of a decision made back in Milestone 12: MCP uses
# stdio transport, meaning mcp_server.py is a plain CHILD PROCESS that
# mcp_client.py launches via sys.executable, not a network service. A
# subprocess lives happily inside whatever container its parent runs in
# -- no Docker Compose, no inter-container network needed. If MCP used
# SSE/HTTP transport instead (a "remote" MCP server), THAT would need two
# containers and a network between them. The transport choice, not "how
# many logical processes exist," is what determines container topology.

FROM python:3.12-slim

WORKDIR /app

# requirements.txt copied and installed BEFORE the rest of the source --
# a deliberate Docker layer-caching optimization, not accidental
# ordering. Docker caches each instruction as a layer; a layer is only
# rebuilt if ITS OWN inputs changed. requirements.txt changes far less
# often than src/. Copying everything first, then installing, would
# invalidate the expensive "pip install" layer on every single code
# change, even a one-line docstring edit -- forcing a full dependency
# reinstall every rebuild. This ordering means editing src/ only ever
# rebuilds the cheap COPY step below, not this one.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# A dedicated, UNPRIVILEGED user to actually RUN the app -- not root.
# Reasoning: if this process is ever compromised (a dependency CVE, any
# exploitable code path), running as root means an attacker has root
# INSIDE the container -- and container root has a more direct path to
# host-level trouble than people assume (kernel exploits, a misconfigured
# volume mount, etc.). Installing dependencies still happens as root
# (needs to write into system site-packages); only the actual running
# process drops to this user, via USER below.
RUN useradd --create-home --shell /bin/bash appuser

COPY --chown=appuser:appuser src/ ./src/

# Matches this project's existing dev convention (PYTHONPATH=src, used
# in every command since Milestone 0) rather than introducing a
# pyproject.toml/pip-installable-package structure this project has
# never had. Consistent with how the codebase already works, not a new
# packaging decision made just for Docker.
ENV PYTHONPATH=/app/src

USER appuser

# Documents the port for humans/tooling reading the Dockerfile -- does
# NOT actually publish it. Publishing happens at `docker run -p`.
EXPOSE 8000

# Lets `docker ps` / any orchestrator (Compose, Kubernetes, ECS) see
# whether this container is actually SERVING, not just "hasn't crashed
# yet" -- a process can be alive and completely wedged (e.g. stuck on a
# hung MCP subprocess) and a bare process-liveness check would never
# catch that. Reuses api.py's own /health endpoint (Milestone 13) for a
# second real purpose. Plain python instead of curl -- avoids an extra
# apt-get layer just for this one check.
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" || exit 1

# --host 0.0.0.0 is not optional, it's the whole point: uvicorn's
# DEFAULT host is 127.0.0.1 (loopback-only), which inside a container
# means "only reachable from within this container's own network
# namespace" -- completely unreachable from outside even with a port
# published via -p, since 127.0.0.1 in the container isn't the same
# 127.0.0.1 the host maps a published port to. This exact mistake is
# THE classic first-time Docker+webserver bug.
CMD ["uvicorn", "argus.api:app", "--host", "0.0.0.0", "--port", "8000"]
