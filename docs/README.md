# Documentation

The tt-xla documentation is built with [Sphinx](https://www.sphinx-doc.org/) and the `sphinx_rtd_theme`, with
[MyST](https://myst-parser.readthedocs.io/) as the markdown parser so pages stay in `.md`. The theme CSS and fonts
are served from `https://docs.tenstorrent.com/_static/`, so styling stays aligned with the canonical Tenstorrent docs site
and tt-xla doesn't need to bundle its own copies. Only the project logo and favicon are kept locally (Sphinx requires
`html_logo` and `html_favicon` to be local file paths).

## Folder structure

```
docs/
|-- README.md               # this file
|-- CMakeLists.txt          # CMake wrappers around the Makefile (see "Building" below)
|-- Makefile                # sphinx-build orchestration (html, clean, server)
|-- build_docs.sh           # install deps + build (used by CI and first-time local builds)
|-- requirements-docs.txt   # pinned Sphinx + extension deps (not in venv - see below)
|-- build/                  # generated output (gitignored)
`-- source/                 # Sphinx source root - everything here is published
    |-- conf.py             # Sphinx configuration
    |-- index.md            # toctree root
    |-- _static/            # tt_logo.svg, favicon.png (everything else served from docs.tenstorrent.com)
    |-- _templates/         # layout.html (sidebar override)
    |-- imgs/               # in-page images
    `-- *.md                # the docs pages themselves
```

To add a new page: drop a `.md` file under `docs/source/`, then list it in the appropriate toctree in
`docs/source/index.md`. The build runs with `-W`, so any page left out of a toctree or referenced by a broken link
will fail CI.

## Dependencies

Docs deps are pinned in `docs/requirements-docs.txt` and are **not** installed by `venv/activate`. They're kept out
of the main venv on purpose: docs is a niche workflow (CI and the occasional contributor), and there's no reason to
add ~20MB of Sphinx machinery to every developer's environment when most people never build docs. Install them once
when you need them, the same way tt-metal does (`tests/scripts/run_build_docs.sh`).

The easiest one-time setup:

```bash
bash docs/build_docs.sh
```

This installs the requirements and builds the docs. Re-running it is fine but the install step is wasteful - once
it's done, prefer the direct entry points below.

## Building

Three entry points, all producing HTML in `docs/build/html/`:

| Command | When to use |
|---|---|
| `bash docs/build_docs.sh` | First-time local build; what CI runs. Installs deps + does a clean build. |
| `cmake --build build -- docs` | When you're already working in the CMake build flow. Assumes deps are installed. |
| `make -C docs html` | Fast local iteration. Assumes deps are installed. |

The CMake target `docs` (defined in `docs/CMakeLists.txt`) is a thin wrapper around `make -C docs html`. It exists
so `cmake --build build -- docs` works the same way it did under the old mdBook setup - the docs build stays
discoverable from the project's primary build system, but the actual work happens in the Makefile. There are
matching `docs-clean` and `docs-serve` CMake targets that wrap `make clean` and `make server`.

The Makefile sets `SPHINXOPTS = -W` by default, so any Sphinx warning (broken cross-reference, unknown Pygments
lexer, heading starting below H1, orphan page, ...) fails the build. CI uses the same setting via `build_docs.sh`,
so local and CI builds either both pass or both fail.

## Previewing locally

```bash
make -C docs server
```

This builds first if needed, then serves `docs/build/html/` on port 8888 (`http://localhost:8888`). Override the
port with `PORT=9090 make -C docs server`. The CMake equivalent is `cmake --build build -- docs-serve`.

### Viewing from your laptop when the build runs on a remote dev machine

You'll want SSH local port forwarding. The shape is:

```bash
ssh -N -L 8888:127.0.0.1:8888 <user>@<dev-machine>
```

`-N` skips opening a shell (the SSH connection just holds the tunnel open). `-L 8888:127.0.0.1:8888` says: listen
on port 8888 on the laptop, forward each connection to `127.0.0.1:8888` *as seen from the SSH server*. Once the
tunnel is up, `http://localhost:8888` in your laptop's browser reaches the `make server` process on the dev
machine. Keep `make -C docs server` running in its own shell on the dev machine.

### When the docs build runs inside a container on a dev machine

If your build environment is a container (e.g. one reserved via `ird`), the container has its own network
namespace. SSH'ing to the host puts you outside it, so a tunnel to the host's `127.0.0.1:8888` finds nothing. You
need to SSH *into the container* instead.

Find the container's SSH port on the host (look for the `->22/tcp` mapping):

```bash
docker ps
# e.g. 0.0.0.0:49805->22/tcp  -> container's sshd is on host port 49805
```

Then tunnel through that port:

```bash
ssh -N -L 8888:127.0.0.1:8888 -p 49805 <user>@<dev-machine>
```

The `-p 49805` makes SSH connect to the container's sshd (not the host's), so `127.0.0.1:8888` resolves inside the
container - exactly where `make server` is listening. Open `http://localhost:8888` in your laptop browser as
before.
