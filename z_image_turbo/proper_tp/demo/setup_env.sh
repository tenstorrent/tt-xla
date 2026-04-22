#!/bin/bash
# Setup environment for the Z-Image-Turbo demo.
#
# The full tt-xla build hasn't been done in this clone, so we borrow
# pre-built native libraries from existing builds on this machine:
#   - ttnn + shared libs from abogdanovic/tt-xla
#   - tt_dit model library from aknezevic/tt-mlir
#
# Usage:
#   source setup_env.sh          # sets up env vars
#   python3 generate.py "your prompt here" --output out.png

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABOG_DIR=/home/ttuser/abogdanovic/tt-xla
AKNEZ_MODELS=/home/ttuser/aknezevic/tt-mlir/third_party/tt-metal/src/tt-metal/models

# Sanity checks
if [ ! -d "$ABOG_DIR/third_party/tt-mlir/install" ]; then
    echo "ERROR: $ABOG_DIR build not found." >&2
    return 1 2>/dev/null || exit 1
fi
if [ ! -d "$AKNEZ_MODELS/tt_dit" ]; then
    echo "ERROR: tt_dit not found at $AKNEZ_MODELS" >&2
    return 1 2>/dev/null || exit 1
fi

# Symlink tt_dit models into our repo tree (model_ttnn.py resolves it via relative path)
TT_METAL_MODELS="$DEMO_DIR/../../../third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/models"
if [ ! -e "$TT_METAL_MODELS" ]; then
    mkdir -p "$(dirname "$TT_METAL_MODELS")"
    ln -sf "$AKNEZ_MODELS" "$TT_METAL_MODELS"
    echo "Created symlink: tt_dit models"
fi

# Python interpreter
export PYTHON="$ABOG_DIR/venv/bin/python3"

# Native libraries
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain
export LD_LIBRARY_PATH="$ABOG_DIR/third_party/tt-mlir/install/lib:$TTMLIR_TOOLCHAIN_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Python path: original ttnn (not the wrapper), tt_dit, demo dir
export PYTHONPATH="$AKNEZ_MODELS:$ABOG_DIR/third_party/tt-mlir/install/tt-metal/ttnn:$ABOG_DIR/third_party/tt-mlir/install/tt-metal:$DEMO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# TT environment
export TT_METAL_HOME="$ABOG_DIR/third_party/tt-mlir/install/tt-metal"
export TT_METAL_LOGGER_LEVEL=ERROR
export ARCH_NAME=blackhole

cd "$DEMO_DIR"

echo "Environment ready. Python: $PYTHON"
echo ""
echo "Run the demo:"
echo "  \$PYTHON generate.py \"a misty mountain lake at dawn\""
echo "  \$PYTHON generate.py \"prompt 1\" \"prompt 2\" --output-dir results/"
