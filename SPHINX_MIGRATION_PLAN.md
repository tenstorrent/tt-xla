# Sphinx Documentation Migration Plan

This document describes the plan for migrating tt-xla documentation from
mdBook to Sphinx. The motivation is to align tt-xla docs with the rest of
Tenstorrent's documentation (notably tt-metal), which is published under
`docs.tenstorrent.com` using Sphinx + `sphinx_rtd_theme`.

A prior attempt (PR #1165, `dimitri/rebuild_with_sphinx`) is **not** being
used as a base. That PR has rotted since August 2025 (deleted-page
references in the toctree, dropped Doxygen with no replacement, broken
`versions.html`, unpinned dependencies, and the workflow file it edits no
longer exists on `main`). Starting fresh is cleaner than rebasing.

## Decisions

These have been settled and shape the rest of the plan:

| Topic | Decision |
|---|---|
| Versioning | Single rolling "latest" — no multi-version hosting |
| Deployment URL | `docs.tenstorrent.com/tt-xla` |
| Source format | Markdown via `myst_parser` (no RST conversion) |
| Jupyter notebooks | **Not** included now. Noted as future work — see "Future work" below |
| Warnings-as-errors | `-W` enabled from day one |
| CMake integration | Keep thin wrapper targets in `docs/CMakeLists.txt` that shell out to the Makefile |
| Local preview | `make server` target included |
| Directory layout | `docs/source/` (Sphinx convention) |
| Theme & static assets | Copy tt-metal's `_static/` wholesale for cross-site visual consistency |
| Analytics (PostHog) | Skipped for now |
| Intersphinx | Enabled, with mappings for tt-mlir and tt-metal |
| Sitemap (`sphinx_sitemap`) | Skipped |
| C++ API docs (Doxygen / breathe) | **Not** needed — tt-xla exposes no public C++ API |

## Target layout

```
docs/
├── CMakeLists.txt              # thin wrappers: `docs`, `docs-clean`, `docs-serve`
├── Makefile                    # sphinx-build orchestration (build, clean, server)
├── requirements-docs.txt       # pinned, mirrors tt-metal versions
└── source/
    ├── conf.py
    ├── index.md                # MyST markdown root (toctree)
    ├── _static/                # copied from tt-metal: fonts, tt_theme.css, logo, favicon
    ├── _templates/
    │   └── layout.html         # copied from tt-metal common/_templates
    ├── getting_started.md
    ├── getting_started_debugging.md
    ├── performance.md
    ├── mixed_precision.md
    ├── test_infra.md
    ├── fusing_and_composite_ops.md
    ├── model_auto_discovery_tests.md
    ├── getting_started_codegen.md
    ├── emitpy_tutorial.md
    ├── troubleshooting_codegen.md
    ├── torch_xla_build.md
    ├── tools.md
    ├── tt_explorer.md
    ├── bisect_improvements.md
    └── imgs/                   # existing image assets, moved from docs/src/imgs/
```

Files to delete:
- `docs/src/` (after pages are moved to `docs/source/`)
- `docs/src/SUMMARY.md` (replaced by `index.md` toctree)
- `docs/book.toml` (mdBook config)
- `docs/doxygen.cfg.in` (Doxygen template, no longer used)

## Implementation steps

### Step 1 — `docs/requirements-docs.txt` (pinned, isolated)

Mirror tt-metal's pins for cross-project consistency. Minimum set for our
chosen extensions:

```
sphinx==7.1.2
sphinx-rtd-theme==1.3.0
sphinxcontrib-jquery==4.1
sphinxcontrib-email==0.3.5
myst-parser==3.0.0
docutils==0.18.1
```

`nbsphinx`, `breathe`, `lxml`, `pandoc`, `nbconvert`, `ipython`,
`tabulate` are intentionally omitted (no notebooks, no Doxygen, no
op-table generation).

**Isolation:** docs deps are **not** wired into `venv/activate`. tt-xla
already auto-installs `python_package/requirements.txt` and
`venv/requirements-dev.txt` on activation; adding ~20MB of Sphinx
machinery to every dev's venv for a workflow only docs contributors and
CI care about is unnecessary. The deps are installed on-demand by the
build wrapper script (see Step 6.5), matching tt-metal's
`run_build_docs.sh` pattern.

### Step 2 — `docs/source/conf.py`

Key configuration choices:

- `project = "TT-XLA"`, `author = "Tenstorrent"`, `copyright = "Tenstorrent"`.
- `extensions`: `sphinx.ext.autodoc`, `sphinx.ext.autosummary`,
  `sphinx.ext.napoleon`, `sphinx.ext.mathjax`, `sphinx.ext.intersphinx`,
  `sphinxcontrib.email`, `myst_parser`.
- `source_suffix = {".rst": "restructuredtext", ".md": "markdown"}`.
- `html_theme = "sphinx_rtd_theme"`.
- `html_baseurl = "/tt-xla/"` (path under `docs.tenstorrent.com`).
- `html_logo = "_static/tt_logo.svg"`, `html_favicon = "_static/favicon.png"` (local files, **not** URLs — Sphinx requires this).
- `html_static_path = ["_static"]`.
- `templates_path = ["_templates"]`.
- `html_css_files = ["tt_theme.css"]`.
- `html_context = {"logo_link_url": "https://docs.tenstorrent.com/"}`.
- `intersphinx_mapping`:
  - `"tt-mlir": ("https://docs.tenstorrent.com/tt-mlir/", None)`
  - `"tt-metal": ("https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/", None)`
  - `"ttnn": ("https://docs.tenstorrent.com/tt-metal/latest/ttnn/", None)`
- Napoleon settings copied from tt-metal (NumPy style).

No PostHog, no breathe, no sitemap. No `REQUESTED_DOCS_PKG` switching
(tt-xla is a single docs package).

### Step 3 — `docs/source/_static/` and `_templates/`

Copy from `/localdev/acicovic/tt-metal/docs/source/common/`:

- `_static/tt_theme.css`
- `_static/fonts/` (Degular*, RMMono*)
- `_static/tt_logo.svg` (or pull from canonical location)
- `_static/favicon.png`
- `_templates/layout.html` (no edits needed — it's parametric on `project`)

**Not** copying: `posthog.js`, `versions.html`, screenshot PNGs/MP4s
specific to tt-metal tutorials.

### Step 4 — `docs/source/index.md`

MyST markdown root with a single toctree covering all current pages. Use
the existing `docs/src/SUMMARY.md` as the source of truth for grouping
and ordering. Tentative structure:

```markdown
# TT-XLA Documentation

```{toctree}
:caption: Getting Started
:maxdepth: 2

getting_started
getting_started_debugging
```

```{toctree}
:caption: Performance
:maxdepth: 2

performance
mixed_precision
```

```{toctree}
:caption: Testing
:maxdepth: 2

test_infra
model_auto_discovery_tests
```

```{toctree}
:caption: Compiler Internals
:maxdepth: 2

fusing_and_composite_ops
```

```{toctree}
:caption: Code Generation
:maxdepth: 2

getting_started_codegen
emitpy_tutorial
troubleshooting_codegen
```

```{toctree}
:caption: Build & Tools
:maxdepth: 2

torch_xla_build
tools
tt_explorer
bisect_improvements
```
```

### Step 5 — Move existing markdown pages

`git mv docs/src/*.md docs/source/` and `git mv docs/src/imgs docs/source/imgs`.
Fix image references inside each `.md` if they used a `src/`-relative
path (most use `./imgs/...` which is fine after the move).

Sweep each page for any `[link](other_page.md)` cross-references and
confirm they still resolve under the new flat layout. MyST handles
`.md` → `.md` links natively, so most should just work.

### Step 6 — `docs/Makefile`

The Makefile assumes deps are already installed and stays minimal — fast
iteration for repeat builds, no pip overhead per invocation.

```make
SPHINXOPTS    ?= -W
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
PORT          ?= 8888

.PHONY: help default clean html server

default: html

clean:
	@rm -rf $(BUILDDIR)

html:
	@$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

server: html
	@echo "Navigate to: http://localhost:$(PORT)/index.html"
	@cd $(BUILDDIR)/html && python -m http.server $(PORT)
```

`-W` is set by default so CI and local builds catch warnings the same way.

### Step 6.5 — `docs/build_docs.sh` (install + build wrapper)

Mirrors tt-metal's `tests/scripts/run_build_docs.sh`. This is the entry
point CI calls and the recommended path for first-time local builds.

```bash
#!/bin/bash
set -eo pipefail

DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing docs requirements..."
pip install -r "${DOCS_DIR}/requirements-docs.txt"

echo "Building docs..."
cd "${DOCS_DIR}"
make clean
make html
```

`chmod +x docs/build_docs.sh` when committing.

Developers doing repeat local builds skip this script and just run
`make -C docs html` after the one-time install. Document this in a short
"Building the docs" section we'll add to one of the existing pages (or
in the `docs/CMakeLists.txt` comment block).

### Step 7 — `docs/CMakeLists.txt` (thin wrappers)

The CMake wrappers stay so `cmake --build build -- docs` keeps working
and the docs build remains discoverable from the project's primary
build system. They are intentionally lightweight — they delegate to the
Makefile and **assume docs deps are already installed** (one-time setup
via `bash docs/build_docs.sh` or `pip install -r docs/requirements-docs.txt`).

Replace the existing mdBook+Doxygen contents with:

```cmake
# All docs work is driven by docs/Makefile. These CMake targets are
# thin wrappers so `cmake --build build -- docs` keeps working.
# Assumes docs deps are installed — see docs/build_docs.sh for the
# install + build path used by CI and first-time local builds.

add_custom_target(docs
  COMMAND ${CMAKE_COMMAND} -E env make -C ${CMAKE_CURRENT_SOURCE_DIR} html
  USES_TERMINAL
  COMMENT "Building Sphinx documentation"
)

add_custom_target(docs-clean
  COMMAND ${CMAKE_COMMAND} -E env make -C ${CMAKE_CURRENT_SOURCE_DIR} clean
  USES_TERMINAL
  COMMENT "Cleaning Sphinx documentation build"
)

add_custom_target(docs-serve
  COMMAND ${CMAKE_COMMAND} -E env make -C ${CMAKE_CURRENT_SOURCE_DIR} server
  USES_TERMINAL
  COMMENT "Serving Sphinx documentation locally"
)
```

Output lives in `docs/build/html/` (not `build/docs/book/` as before).
This will require updating the CI workflow upload path.

**Three entry points, one source of truth:**

| Entry point | When | What it does |
|---|---|---|
| `bash docs/build_docs.sh` | CI; local first-time | `pip install` deps + `make clean html` |
| `cmake --build build -- docs` | Project-build flows | `make -C docs html` (deps assumed installed) |
| `make -C docs html` | Local repeat builds | direct Sphinx build (fastest iteration) |

### Step 8 — CI workflows

Two workflow files currently call into mdBook:

- `.github/workflows/call-check-docs-build.yml` — PR-time build verification.
- `.github/workflows/call-build-and-deploy-docs.yml` — push-to-main build + deploy.

Edits (both files):

- Remove `MDBOOK_VERSION` env and the `Install system deps (Doxygen, Rust, Cargo)` step.
- Add a `Set up Python` step (`actions/setup-python@v5`, `python-version: 3.12`).
- Replace the `cmake --build build -- docs` step with `bash docs/build_docs.sh`.
  This handles both `pip install` and the Sphinx build in one step,
  matching tt-metal's CI pattern and keeping the workflow YAML thin.
- Update `upload-pages-artifact` `path:` from `./build/docs/book` to
  `./docs/build/html`.

Deployment mechanism (current: `actions/configure-pages` →
`upload-pages-artifact` → `deploy-pages` to GitHub Pages) stays the same
for the rolling-latest model. The migration to `docs.tenstorrent.com/tt-xla`
is a DNS/CNAME concern handled outside this PR — see "Out of scope" below.

### Step 9 — Pre-commit / formatting

Verify the Sphinx-generated build directory (`docs/build/`) is in
`.gitignore`. Add it if not.

Check that `pre-commit` does not try to format `.md` files inside
`docs/source/` in a way that breaks MyST directives (the existing config
should already handle this — confirm during testing).

## Local validation checklist

Before opening the PR:

1. `pip install -r docs/requirements-docs.txt` in a clean venv.
2. `make -C docs clean html` — must succeed with `-W` enabled.
3. `make -C docs server` — open in browser, visually confirm:
   - Tenstorrent logo + theme CSS load correctly.
   - Sidebar lists all pages with the intended grouping.
   - At least one image renders (sanity check for the `imgs/` move).
   - At least one intra-doc link (e.g. `getting_started` → `test_infra`) works.
4. `cmake -G Ninja -B build && cmake --build build -- docs` — confirm
   the thin CMake wrapper still produces the same output.
5. Verify all `.md` files referenced in `index.md`'s toctree exist; verify
   no `.md` file in `docs/source/` is orphaned (Sphinx will warn under `-W`).

## Out of scope for this PR

- DNS / CNAME setup for `docs.tenstorrent.com/tt-xla`. That is an
  infrastructure change owned by whoever runs `docs.tenstorrent.com`.
  Until it's wired up, docs continue to publish at the existing GitHub
  Pages URL — only the `html_baseurl` in `conf.py` anticipates the
  future location.
- Multi-version hosting (`published_versions.json`, version dropdown).
- Jupyter notebook support via `nbsphinx`.
- Doxygen / `breathe` integration (no public C++ API).
- PostHog analytics.
- Sitemap generation.

## Future work

These are explicitly deferred but worth tracking:

- **Notebook tutorials via `nbsphinx`.** Desired but no `.ipynb` content
  exists today. When the first notebook lands, add `nbsphinx` to
  requirements + extensions, and add `nbconvert` / `pandoc` /
  `ipython` to requirements.
- **PostHog analytics.** Skipped pending a decision on whether tt-xla
  docs should match tt-metal's instrumentation.
- **Multi-version hosting.** Add if and when tt-xla begins shipping
  versioned releases that warrant frozen doc snapshots.
- **API autodoc.** If `jax_plugin_tt` / `torch_plugin_tt` ever grow a
  public Python API surface worth documenting, wire up `autodoc` +
  `autosummary` against those packages.
