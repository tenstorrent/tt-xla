# TT-XLA Setup Options on a Developer Machine

This doc records two concrete paths to get a TT-XLA Python environment up on a
shared developer host, so branch work (for example the quetzal FX rewrite
branch, which changes only Python) can be tested end-to-end without rebuilding
the world.

The official, canonical setup is in `docs/src/getting_started.md`. This doc is
narrower: it is a practical cheat sheet for a host where you do NOT have sudo,
you DO have a Tenstorrent card, and you want to exercise the Python frontend
against a prebuilt PJRT plugin.

## When Option A vs Option B

- **Option A (wheel path)** — Use when the branch touches only Python
  (`git diff --stat` shows no `*.cpp`, `*.h`, or `CMakeLists.txt` changes).
  The prebuilt `pjrt-plugin-tt` wheel provides the C++ PJRT plugin, and your
  checkout's `python_package/tt_torch` is overlaid via `PYTHONPATH`. No
  toolchain build required, no `sudo apt install` required, no tt-mlir build
  required.
- **Option B (source build)** — Use when the branch touches C++ in `src/`,
  changes the `CMakeLists.txt`, or needs a tt-mlir commit newer than the one
  frozen in the current public wheel. Requires `protobuf-compiler` +
  `libprotobuf-dev` (system install, typically needs sudo) and a tt-mlir
  toolchain. See `docs/src/getting_started.md` for the full procedure.

## Host Prerequisites

These must be present regardless of path:

- Linux x86_64
- Python 3.12 binary available somewhere on the system (or installable via
  `uv python install 3.12` — no sudo needed)
- A Tenstorrent device node at `/dev/tenstorrent/0` (or equivalent) if you
  intend to run IR comparisons on real hardware. Compile-only runs can use a
  saved `.ttsys` descriptor instead.

Quick inventory commands:

```bash
ls /dev/tenstorrent                                  # cards visible
ls /usr/bin/python3.12 /proj_sw/**/python3.12 2>/dev/null  # existing 3.12
command -v uv                                        # uv fallback
```

## Option A: Wheel Path (Python-only branch)

### 1. Python 3.12

If the system already exposes `python3.12`, use it. Otherwise, with `uv`:

```bash
uv python install 3.12
```

### 2. Create an isolated venv outside the repo

Using a venv under `/tmp` (or anywhere non-destructive) keeps the repo
untouched so the in-tree `venv/activate` script is unaffected:

```bash
uv venv --python 3.12 /tmp/quetzal-venv
source /tmp/quetzal-venv/bin/activate
```

### 3. Install TT-XLA wheels from the Tenstorrent index

```bash
pip install pjrt-plugin-tt \
  --extra-index-url https://pypi.eng.aws.tenstorrent.com/
```

This installs `pjrt_plugin_tt` (the PJRT `.so` + its tt-mlir/tt-metal runtime
dependencies), plus the thin `jax_plugin_tt` and `torch_plugin_tt` wrappers.

### 4. Install the project's Python requirements

Pin JAX/Torch to match the wheel, plus test tooling:

```bash
pip install -r python_package/requirements.txt
pip install -r venv/requirements-dev.txt
```

If `torch-xla` is not pulled transitively, install Tenstorrent's custom build
from the same index:

```bash
pip install torch-xla \
  --extra-index-url https://pypi.eng.aws.tenstorrent.com/
```

### 5. Put the branch's `tt_torch` on PYTHONPATH

The prebuilt wheel installs `pjrt_plugin_tt`, `jax_plugin_tt`,
`torch_plugin_tt`. It does **not** contain the in-repo `tt_torch` package,
which holds the torch backend logic including this branch's additions
(`quetzal_rewrite.py`, `quetzal_analysis.py`). Overlay it:

```bash
export PYTHONPATH="$(pwd)/python_package:$(pwd):$(pwd)/tests"
```

Order matters: `python_package` must come first so `import tt_torch` resolves
to the checkout, not site-packages.

### 6. Smoke tests

Device visibility:

```bash
python -c "import jax; print(jax.devices('tt'))"
python -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices('tt'))"
```

Focused quetzal tests (no device required):

```bash
pytest -svv \
  tests/torch/test_quetzal_rewrite.py \
  tests/torch/test_quetzal_analysis.py \
  tests/torch/test_quetzal_ir_compare.py
```

End-to-end IR comparison on a real device:

```bash
python scripts/compare_quetzal_rewrite_ir.py \
  --case all \
  --output-dir /tmp/quetzal-ir \
  --strict
```

No-device, compile-only mode requires a `.ttsys` system descriptor produced
on a machine with real hardware (see `examples/pytorch/system_desc.py`):

```bash
python scripts/compare_quetzal_rewrite_ir.py \
  --case all \
  --output-dir /tmp/quetzal-ir \
  --system-desc /path/to/system_desc.ttsys \
  --strict
```

### Caveats for Option A

- The wheel's `pjrt_plugin_tt.so` is pinned to a specific tt-mlir commit. If
  your branch relies on tt-mlir behavior newer than that pin, Option A will
  succeed at compile but may produce IR that diverges from what the HEAD
  compiler would emit. In that case switch to Option B.
- The wheel's Python wrappers (`torch_plugin_tt`, `jax_plugin_tt`) are
  independent from the in-repo `tt_torch` package — do not confuse the two.

## Option B: Source Build (when C++ or tt-mlir version changes)

This path follows `docs/src/getting_started.md` §"Building from Source" and is
only summarized here.

Additional requirements on top of Option A:

- `sudo apt install protobuf-compiler libprotobuf-dev ccache libnuma-dev \
  libhwloc-dev libboost-all-dev libnsl-dev`
- A built tt-mlir toolchain at `$TTMLIR_TOOLCHAIN_DIR`. If you already have
  one from a previous tt-xla or tt-mlir build on the same host, reuse it; it
  contains llvm/clang/lld and tt-mlir libraries and is the most expensive
  artifact to rebuild.
- `TTXLA_ENV_ACTIVATED=1` (set by `venv/activate`).

Build:

```bash
source venv/activate
cmake -G Ninja -B build
cmake --build build
```

The `venv/activate` script creates `./venv` in-place on first activation and
bakes the absolute path into its `bin/activate`. That venv is **not
relocatable**: if the workspace is later bind-mounted or copied to a different
absolute path, recreate the venv.

### Common gotchas

- `clang-20` is required. `clang++` must resolve to the matching version.
- The tt-mlir toolchain's `venv/bin/python` symlink is often pinned to a
  specific host's `/usr/bin/python3.12`. Cross-host clones will show a broken
  symlink and activation will fail; recreate the venv.
- Only the wheel path avoids `sudo` entirely. A source build without sudo is
  possible but requires building protobuf into a local prefix and exporting
  `CMAKE_PREFIX_PATH` — not recommended.

## Minimal Testing Matrix for Python-only Changes

For a branch that only touches `python_package/tt_torch/**`, the minimum test
set under Option A is:

1. `pytest -svv tests/torch/test_quetzal_rewrite.py` (or the focused file for
   the branch).
2. `python scripts/compare_quetzal_rewrite_ir.py --case all --strict` on a
   real device.
3. Inspect the generated `summary.json` and the exported `.mlir` artifacts
   under `--output-dir` to confirm expected StableHLO / TTIR / TTNN deltas.

If all three succeed, the Python-level behavior is validated end-to-end. The
C++ PJRT plugin itself has not been re-exercised by this run; for C++ changes
use Option B and run the full pytest suite in `tests/`.
