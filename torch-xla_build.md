# Torch-XLA Build Instructions

This document describes how to build torch-xla from source and connect tt-xla to use your custom build.

## Building Torch-XLA from Source

1. Clone and build torch-xla:
```bash
cd /localdev/vkovinic/temp/pytorch/xla
python setup.py develop
```

## How to Connect tt-xla to Use Custom Torch-XLA Build

To ensure tt-xla uses your custom torch-xla build instead of the system-installed version, you need to set the `PYTHONPATH` environment variable to prioritize your custom build.

### Method 1: Using PYTHONPATH (Recommended)

When running tt-xla commands or scripts, prefix them with the custom torch-xla path:

```bash
cd /localdev/vkovinic/tt-xla
PYTHONPATH=/localdev/vkovinic/temp/pytorch/xla:$PYTHONPATH python your_script.py
```

### Method 2: Export PYTHONPATH in Shell

For persistent use in your current shell session:

```bash
export PYTHONPATH=/localdev/vkovinic/temp/pytorch/xla:$PYTHONPATH
```

Add this to your `~/.bashrc` or `~/.zshrc` for permanent effect.

### Verification

To verify that tt-xla is using your custom torch-xla build:

```bash
cd /localdev/vkovinic/tt-xla
PYTHONPATH=/localdev/vkovinic/temp/pytorch/xla:$PYTHONPATH python -c "import torch_xla; print('torch-xla path:', torch_xla.__file__); print('torch-xla version:', getattr(torch_xla, '__version__', 'unknown'))"
```

Expected output should show:
- Path: `/localdev/vkovinic/temp/pytorch/xla/torch_xla/__init__.py` (your custom build)
- Version: `2.9.0+git1adbe97` (or similar git hash)

If you see path `/usr/local/lib/python3.11/dist-packages/torch_xla/__init__.py`, then it's using the system version instead of your custom build.

### Running Tests with Custom Build

When running tt-xla tests or executing models:

```bash
cd /localdev/vkovinic/tt-xla
PYTHONPATH=/localdev/vkovinic/temp/pytorch/xla:$PYTHONPATH python tests/torch/single_chip/test_nested_model.py
```

This ensures your custom torch-xla changes (including debug logging and semantic location enhancements) are active.