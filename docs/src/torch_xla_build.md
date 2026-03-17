# Building PyTorch XLA from Source

This guide covers building PyTorch and PyTorch-XLA (Tenstorrent fork) from source for development on tt-xla.

## Overview

- **Build Method**: Official PyTorch XLA contributing guide workflow
- **PyTorch Version**: 2.9.1
- **XLA Source**: [Tenstorrent fork](https://github.com/tenstorrent/pytorch-xla.git)
- **Python**: 3.12
- **Bazel**: 7.4.1
- **Total Time**: ~2-2.5 hours (first build)

## Automated Build

A script automates the entire process:

```bash
./scripts/build_torch_xla.sh            # Release build (default)
./scripts/build_torch_xla.sh --debug    # Debug build
```

The script handles cloning, building, and integrating into the tt-xla venv. Subsequent runs skip builds if the source hasn't changed.

## Manual Build

### Phase 1: System Setup

#### 1. Install System Dependencies

```bash
sudo apt update

# Install build tools
sudo apt install -y build-essential cmake git curl wget unzip

# Install development libraries
sudo apt install -y libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev
sudo apt install -y libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev
```

#### 2. Install Python 3.12

```bash
# Add Python repository
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# Install Python 3.12 with development headers
sudo apt install python3.12 python3.12-venv python3.12-dev -y

# Verify installation
python3.12 --version
```

#### 3. Install Bazel 7.4.1

```bash
# Download and install specific Bazel version required by XLA
wget https://github.com/bazelbuild/bazel/releases/download/7.4.1/bazel-7.4.1-installer-linux-x86_64.sh
chmod +x bazel-7.4.1-installer-linux-x86_64.sh
sudo ./bazel-7.4.1-installer-linux-x86_64.sh

# Verify installation
bazel --version  # Should show 7.4.1
```

### Phase 2: Python Environment Setup

#### 4. Create Workspace Directory

```bash
mkdir -p /localdev/$USER/temp
cd /localdev/$USER/temp
```

#### 5. Set Up Virtual Environment

```bash
# Create Python 3.12 virtual environment
python3.12 -m venv torch_dev_env
source torch_dev_env/bin/activate

# Verify correct Python version
python --version  # Should show 3.12.x
which python      # Should point to torch_dev_env
```

#### 6. Install Python Build Dependencies

```bash
# Upgrade core tools
pip install --upgrade pip setuptools wheel

# Install build dependencies
pip install numpy pyyaml cmake ninja typing_extensions six requests \
    dataclasses astunparse expecttest hypothesis psutil
```

### Phase 3: Repository Setup

#### 7. Clone PyTorch 2.9.1

```bash
# Ensure we're in the temp directory
cd /localdev/$USER/temp

# Clone specific PyTorch version with submodules
git clone --recursive --branch v2.9.1 https://github.com/pytorch/pytorch
cd pytorch/

# Verify we have the correct version
cat version.txt  # Should show 2.9.1a0
```

#### 8. Clone Tenstorrent PyTorch XLA Fork

```bash
# Clone XLA fork as 'xla' inside pytorch/ (correct structure)
git clone --recursive https://github.com/tenstorrent/pytorch-xla.git xla
```

**Final Directory Structure:**
```
/localdev/$USER/temp/
├── torch_dev_env/          # Virtual environment
└── pytorch/                # PyTorch 2.9.1 source
    └── xla/                # Tenstorrent XLA fork
```

### Phase 4: Build Process

#### 9. Build PyTorch

```bash
# Navigate to PyTorch root
cd /localdev/$USER/temp/pytorch

# Set build configuration
export USE_CUDA=0      # Building for TPU/CPU (not GPU)
export BUILD_TEST=0    # Skip tests for faster build

# Build PyTorch wheel (required by XLA build process)
python setup.py bdist_wheel

# Install PyTorch in development mode
python setup.py develop
```

#### 10. Build PyTorch XLA

```bash
# Navigate to XLA directory
cd /localdev/$USER/temp/pytorch/xla/

# Clean any existing build
rm -rf build/

# Set environment variables for XLA build
export PYTORCH_REPO_PATH=/localdev/$USER/temp/pytorch
export TEST_TMPDIR=/localdev/$USER/temp/bazel_tmp
export HERMETIC_PYTHON_VERSION=3.12

# Create Bazel temp directory
mkdir -p /localdev/$USER/temp/bazel_tmp

# Build XLA in development mode
python setup.py develop

# Build XLA in DEBUG mode
# DEBUG=1 python setup.py develop
```

> **Note**: `HERMETIC_PYTHON_VERSION=3.12` is required when the system `python3` is not 3.12
> (e.g., Ubuntu 22.04 where `/usr/bin/python3` points to 3.10). Without it, Bazel compiles
> native extensions against the wrong Python version.

### Phase 5: Verification and Development

#### 11. Verify Installation

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Test XLA installation
python -c "import torch_xla; print('PyTorch XLA imported successfully')"

# Check available XLA devices (if hardware is available)
python -c "import torch_xla.core.xla_model as xm; print(f'XLA devices: {xm.get_xla_supported_devices()}')"
```

#### 12. Development Workflow

```
# Your development workspace structure:
/localdev/$USER/temp/
├── torch_dev_env/          # Virtual environment
├── bazel_tmp/              # Bazel cache and temporary files
└── pytorch/                # PyTorch source (editable install)
    ├── dist/               # PyTorch wheel package
    └── xla/                # XLA source code (YOUR DEVELOPMENT CODE HERE)

# For iterative development:
# 1. Make code changes in /localdev/$USER/temp/pytorch/xla/
# 2. For Python changes: No rebuild needed (development mode)
# 3. For C++/Bazel changes: python setup.py develop (incremental)
# 4. Test changes immediately
```

### Environment Management

#### Essential Environment Variables

```bash
# Always set these when working on XLA
export PYTORCH_REPO_PATH=/localdev/$USER/temp/pytorch
export TEST_TMPDIR=/localdev/$USER/temp/bazel_tmp
export USE_CUDA=0
export BUILD_TEST=0
export HERMETIC_PYTHON_VERSION=3.12

# Optional: Enable debug mode for enhanced debugging
# export DEBUG=1
```

## Integration with tt-xla

If you have tt-xla set up, you can make it use your development version instead of downloading from PyPI:

```bash
# Activate the tt-xla virtual environment
cd /localdev/$USER/tt-xla
source venv/activate

# Remove any existing torch-xla installation
pip uninstall torch_xla -y

# Install your development XLA in editable mode
cd /localdev/$USER/temp/pytorch/xla
pip install -e .

# Install missing dependencies
pip install "Jinja2>=3.1" "Pygments>=2.17" torch==2.9.1

# Verify your development version is being used
python -c "import torch_xla; print(f'XLA version: {torch_xla.__version__}'); print(f'XLA path: {torch_xla.__file__}')"
```

**Expected output path**: `/localdev/$USER/temp/pytorch/xla/torch_xla/__init__.py` (points to your source)

After making changes to the `torch-xla` repo, for incremental build you can just use:

```bash
# Activate the build venv (not tt-xla venv)
source /localdev/$USER/temp/torch_dev_env/bin/activate

# Go to the torch-xla location
cd /localdev/$USER/temp/pytorch/xla/

# Incremental build
python setup.py develop

# Go back to your previous location
cd -
```

## Troubleshooting

### `_XLAC_cuda_functions` Python version mismatch

If you see:
```
ImportError: Python version mismatch: module was compiled for Python 3.10
```

Bazel compiled extensions against the system Python instead of 3.12. Fix:

```bash
# Clean bazel cache and rebuild
rm -rf /tmp/$USER/bazel_cache   # or wherever your bazel output_base is
cd /localdev/$USER/temp/pytorch/xla
rm -rf build/
export HERMETIC_PYTHON_VERSION=3.12
python setup.py develop
```

### `TTMLIR_TOOLCHAIN_DIR: unbound variable`

The tt-xla `venv/activate` script expects this variable. Either set it before running the build script, or ensure tt-xla is properly set up first.
