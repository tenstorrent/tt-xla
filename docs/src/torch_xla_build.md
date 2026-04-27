# Building PyTorch XLA from Source

This guide covers building PyTorch and PyTorch-XLA (Tenstorrent fork) from source for development on tt-xla.

## Overview

- **Build Method**: Official PyTorch XLA contributing guide workflow
- **PyTorch Version**: 2.9.1
- **XLA Source**: [Tenstorrent fork](https://github.com/tenstorrent/pytorch-xla.git)
- **Python**: 3.12
- **Bazel**: 7.4.1
- **Total Time**: ~2-2.5 hours (first build)

## Build

The [`scripts/build_torch_xla.sh`](../../scripts/build_torch_xla.sh) script automates the entire process — installing dependencies, cloning repos, building, and integrating into the tt-xla venv. Each step is documented with comments in the script itself.

```bash
./scripts/build_torch_xla.sh            # Release build (default)
./scripts/build_torch_xla.sh --debug    # Debug build
```

Subsequent runs skip builds if the source hasn't changed.

## Incremental Rebuilds

After making changes to the `torch-xla` repo, you can do an incremental build:

```bash
# Activate the build venv (not tt-xla venv)
source temp/torch_dev_env/bin/activate

# Go to the torch-xla location
cd temp/pytorch/xla/

# Incremental build
python setup.py develop
```

For Python-only changes, no rebuild is needed (development mode).

## Troubleshooting

### `_XLAC_cuda_functions` Python version mismatch

If you see:
```
ImportError: Python version mismatch: module was compiled for Python 3.10
```

Bazel compiled extensions against the system Python instead of 3.12. Fix:

```bash
rm -rf /tmp/$USER/bazel_cache
cd temp/pytorch/xla
rm -rf build/
export HERMETIC_PYTHON_VERSION=3.12
python setup.py develop
```

### `TTMLIR_TOOLCHAIN_DIR: unbound variable`

The tt-xla `venv/activate` script expects this variable. Either set it before running the build script, or ensure tt-xla is properly set up first.
