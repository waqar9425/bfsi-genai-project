# Containers & CI/CD — Explained From Scratch

Same purpose as `LANGGRAPH_CONCEPTS.md`, different domain: every container/
CI/CD idea used in this project, explained assuming zero prior knowledge,
each section self-contained. **Updated whenever this area of the project
grows.**

---

## 1. What a container actually is (not the marketing description)

A container is **not** a lightweight VM. There's no hypervisor, no second
kernel. A container is an ordinary Linux **process**, made to look
isolated using three kernel features that predate Docker by years:

- **Namespaces** — the kernel gives a process its own private view of
  something normally global: its own process tree (PID namespace — inside
  the container you're PID 1, not PID 47291), its own network stack
  (interfaces, routes, ports), its own filesystem mount tree, its own
  hostname. Docker doesn't invent isolation, it just asks the kernel to
  create a fresh set of these for the process it launches.
- **cgroups** (control groups) — kernel-enforced resource *limits*: CPU,
  memory, I/O. The mechanism behind `docker run --memory=512m`.
- **A layered filesystem (OverlayFS)** — an image is a stack of read-only
  layers; a running container adds ONE thin writable layer on top. Every
  layer below is shared and reusable across any number of containers from
  the same image — why spinning up ten containers from one image is
  cheap: nine of them are just new thin writable layers on top of
  filesystem content already in the kernel's page cache.

**The sharp interview answer to "container vs. VM":** a VM virtualizes
hardware and boots a full second kernel; a container is an ordinary
process, isolated by namespaces, resource-bounded by cgroups, given its
filesystem via an overlay of image layers — same kernel as the host,
always. That's why containers start in milliseconds and VMs take
seconds-to-minutes: there's no kernel to boot.

## 2. Images vs. containers — the class/instance distinction, precisely

An **image** is a read-only template: a stack of layers (each the
filesystem diff from one Dockerfile instruction) plus metadata (`CMD`,
`ENV`, `EXPOSE`). It does nothing by itself — bytes on disk,
content-addressed by hash (`sha256:...`), which is why `docker pull` can
skip layers you already have (it compares hashes, not filenames).

A **container** is what you get when the runtime is told "start a
process from this image": takes the layer stack, adds a writable layer on
top (anything the running process writes lands here), creates fresh
namespaces, sets cgroup limits, execs the image's `CMD` inside all of it.
Delete the container, that writable layer is gone; the image layers are
untouched and can spawn an identically-clean new container instantly.
This is the entire mechanism behind "containers are disposable, images
are durable" — not a philosophy, a direct consequence of the layered
filesystem.

## 3. Layer caching, concretely, from our actual Dockerfile

```dockerfile
FROM python:3.12-slim          → base OS + Python layers
COPY requirements.txt .        → +1 layer (one small file)
RUN pip install ...            → +1 layer (all installed packages -- the BIG one)
RUN useradd appuser             → +1 layer
COPY --chown=... src/ ./src/   → +1 layer (our code)
```

Every `docker build` after the first, **if `requirements.txt` hasn't
changed**, skips re-running `pip install` entirely — reuses the cached
layer by hash, re-executing only from `COPY src/` downward. This is why
instruction order matters: put what changes rarely (`requirements.txt`)
before what changes constantly (`src/`), or every code edit forces a full
dependency reinstall.

## 4. Multi-stage builds (not needed here, but commonly asked about)

Our image doesn't need this — pure Python, no compilation — but it's
worth being precise on. Uses more than one `FROM` in one Dockerfile:

```dockerfile
FROM python:3.12 AS builder          # full image, has compilers
RUN pip install --user some-package-with-c-extensions

FROM python:3.12-slim                # final image, NO compilers
COPY --from=builder /root/.local /root/.local   # only the BUILT artifacts cross over
```

The `builder` stage's entire bulk (compilers, headers, intermediate
files) is discarded — only what's explicitly `COPY --from=builder`'d
survives. How you get a small final image even when *building* something
needs a heavy toolchain: compile in a fat stage, ship only the result in
a thin one.

## 5. When you need Docker Compose -- and why THIS project doesn't (yet)

Compose runs **multiple containers that need to talk to each other** as
one coordinated unit, giving them a shared network where they reach each
other **by service name** via Compose's built-in DNS — no manual IP
management.

**Argus doesn't need this**, for a specific, deliberate reason: MCP
(Milestone 12) uses **stdio transport**, so `mcp_server.py` is a
subprocess of the same container, not a separate network service. If MCP
used SSE/HTTP transport instead (a "remote" MCP server), THAT would
genuinely need two containers and a network between them. **The
transport choice, not "how many logical processes exist," is what
determines container topology.**

We already know where this changes: `MemorySaver` (Milestone 11) is
dev-only; a real deployment would swap in `PostgresSaver`, and Postgres
IS a genuinely separate service. Concretely, what that would look like:

```yaml
services:
  argus:
    build: .
    ports: ["8000:8000"]
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - DATABASE_URL=postgresql://argus:argus@db:5432/argus   # "db" resolves via Compose DNS
    depends_on: [db]
  db:
    image: postgres:16
    environment:
      - POSTGRES_USER=argus
      - POSTGRES_PASSWORD=argus
      - POSTGRES_DB=argus
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

`db` in the connection string isn't a placeholder — it's Compose's actual
internal DNS resolving the service name to that container's current
address, no hardcoded IPs, surviving container restarts with new IPs.

## 6. Volumes vs. bind mounts

Containers are disposable — so how does a database survive a restart? A
**named volume** (`volumes: pgdata:` above) is storage managed by the
Docker/Podman engine itself, living *outside* any single container's
writable layer, mounted into whichever container needs it. Delete and
recreate the `db` container pointed at the same volume, the data's still
there. A **bind mount** (`- ./local-folder:/data`) instead maps a
specific *host* directory directly in — useful for local dev (e.g.
mounting source code live without rebuilding an image), but ties you to
that exact host path, which a named volume doesn't.

## 7. Docker vs. Podman — the real architectural difference

**Docker requires a persistent background daemon (`dockerd`)**,
traditionally running as root; the `docker` CLI talks to it over a socket
(`/var/run/docker.sock`). Every container is a child of the daemon, not
of your shell. Real security consequence: anyone who can write to that
socket effectively has root on the host (a known privilege-escalation
vector — mount the socket into a container, and that container can launch
new arbitrary privileged containers on the host).

**Podman is daemonless.** `podman run` forks/execs the container process
**directly as a child of your own shell** — no background daemon, no
socket, no single elevated-privilege chokepoint. Built from the start
around **rootless containers as the default expectation**, using Linux
user namespaces so an unprivileged user can run something that *looks*
like root *inside* the container while genuinely being unprivileged on
the host. Docker has added rootless mode too, but retrofitted; Podman's
whole architecture assumes it.

CLI is deliberately near-identical (`podman build`/`run`/`ps`, often
literally `alias docker=podman`) — intentional: "same workflow, safer
architecture," not "learn something new."

**Podman's name is literally "Pod Manager"** — borrows Kubernetes' **pod**
concept (a group of containers sharing a network namespace) as a
first-class *local* construct (`podman pod create`). Docker Compose
"services" don't map onto a k8s Pod the same clean way. If you already
know Podman pods, Kubernetes Pods aren't a new concept, just a
distributed version of one you've already used locally.

**Why this matters industry-wise:** Red Hat/OpenShift standardized on
Podman; Docker's 2021+ Docker Desktop licensing changes (usage fees for
larger companies) pushed real adoption toward Podman/daemonless tooling;
many CI runners default to Podman or `buildah` specifically to avoid
running a privileged daemon in shared infrastructure.

## 8. Registries

A built image on your machine isn't deployable anywhere else. A
**registry** (Docker Hub, GitHub Container Registry, AWS ECR, Google
Artifact Registry) is where a built image gets **pushed** so another
machine can **pull** it. Naming convention:
`registry-host/namespace/repo:tag` — e.g. `ghcr.io/waqar9425/argus:v1.2.0`
— `docker build -t <that-full-name> .` tags at build time so `docker push`
knows the destination. Not yet used in this project (images built
locally/in-CI, never pushed anywhere) -- the natural next step if a real
deploy target is chosen later.

---

## 9. CI/CD — the two-tier design, and what it actually caught

`ci.yml` (fast: every push/PR) and `eval-gate.yml` (slow: manual +
nightly) are the literal GitHub Actions materialization of the two-tier
philosophy `eval_gate.py` was built around back in Milestone 14 -- not a
new design made for CI, the CI is what that design was always for.

**GitHub Actions mechanics used here:**
- `on:` — triggers (`push`, `pull_request`, `workflow_dispatch` for a
  manual button, `schedule` with cron syntax for nightly runs).
- `jobs:` — independent units that run in PARALLEL by default (unless
  one declares `needs:` on another) -- faster feedback, clearer failure
  attribution than one monolithic job.
- `secrets.GOOGLE_API_KEY` — referenced via the `secrets` context, never
  written directly in the workflow; GitHub auto-redacts any secret value
  that happens to appear in logs. Set once via the repo's Settings →
  Secrets UI, never through a command-line tool or chat.
- Third-party actions pinned to a version tag (`actions/checkout@v4`),
  not `@main`/`@latest` -- an action maintainer's default branch is a
  real supply-chain vector; pinning (ideally to a commit SHA for
  anything security-sensitive) is standard practice.

**Every one of the following was a REAL failure from an actual CI run,
not designed in from the start:**
- A smoke test asserting `"paused":false` -- a specific business
  outcome, not deployment health. A real transient tool failure made the
  harness correctly escalate instead of crash (working exactly as
  designed since Milestone 4) -- and the smoke test failed anyway,
  because it checked the wrong thing. Fixed: assert the response is
  well-formed (`thread_id`/`reply` present), accept either valid outcome.
- `eval-gate.yml` firing on every push, concurrently with `ci.yml`, both
  hammering the same free-tier key at once -- directly caused the
  failure above. Fixed: manual + nightly only.
- An unauthenticated polling loop against GitHub's REST API (checking
  run status) burned the 60-req/hour rate limit in one loop. Should have
  used one longer-interval check, or just pointed at the Actions UI
  directly instead of polling from a script at all.
- 4 test files (import-time API cost, not actual test need) were slowing
  down the fast tier 3x for no real benefit -- moved to the slow tier,
  nothing lost, ~22s → ~7.5s for the tier that runs on every push.

---

## 10. Getting Docker running for real: WSL2 without systemd

A genuinely common real-world scenario with surprisingly thin official
guidance: installing Docker Engine directly inside a WSL2 distro that
does **not** have `systemd` as PID 1 (check: `ps -p 1 -o comm=` — if it
says `init` rather than `systemd`, this applies).

**What doesn't work:** `systemctl start docker` -- no systemd to talk to.
**What also doesn't necessarily work:** `service docker start` -- depends
on whether `docker.io`'s postinstall actually registered
`/etc/init.d/docker`; in this project's case it didn't, because an
UNRELATED pre-existing broken package (`azcmagent`, Azure's hybrid-server
agent) failed during the same `apt install` transaction and interrupted
trigger processing partway through -- `docker.io` itself still installed
fine (`dpkg -s docker.io` showed `install ok installed`), just without
getting its init script registered as a side effect of the earlier
package's failure.

**What actually works: start the daemon binary directly.**
```
sudo dockerd > /tmp/dockerd.log 2>&1 &
```
Backgrounds `dockerd` (the actual daemon process) directly, no init
system involved at all. Needs `sudo` -- the daemon manages low-level
namespace/cgroup/network operations, genuinely requires root.

**After that, add yourself to the `docker` group** (one-time, needs
sudo) so future `docker` commands don't need `sudo` at all:
```
sudo usermod -aG docker $USER
```
Requires a NEW shell session to take effect (group membership is read at
login) -- either open a new terminal, or `newgrp docker` in the current one.

**Verified working, real output from this exact setup:**
```
$ docker ps
CONTAINER ID   IMAGE         STATUS                        PORTS
573d404e4948   argus:local   Up About a minute (healthy)   0.0.0.0:8000->8000/tcp
```
That `(healthy)` is our own `HEALTHCHECK` directive (Milestone 15)
actually running -- starts as `(health: starting)` during the
`--start-period` grace window, transitions to `(healthy)` once `/health`
actually responds correctly.
