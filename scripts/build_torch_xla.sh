#!/usr/bin/env bash
#
# Build PyTorch + PyTorch-XLA (Tenstorrent fork) from source
# and integrate into tt-xla's venv.
#
# Based on: https://gist.github.com/vkovinicTT/dce22ed9d94065d3c64c61cf1805ef14
#
# Usage:
#   ./scripts/build_torch_xla.sh [--debug]   # default is Release
#   ./scripts/build_torch_xla.sh --release
#   ./scripts/build_torch_xla.sh --debug
#
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PYTHON_VERSION="3.12"
PYTORCH_TAG="v2.9.1"
PYTORCH_XLA_REPO="https://github.com/tenstorrent/pytorch-xla.git"
PYTORCH_XLA_BRANCH="master"
BAZEL_VERSION="7.4.1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMP_DIR="${PROJECT_ROOT}/temp"
PYTORCH_DIR="${TEMP_DIR}/pytorch"
XLA_DIR="${PYTORCH_DIR}/xla"
BAZEL_TMP="${TEMP_DIR}/bazel_tmp"
VENV_DIR="${TEMP_DIR}/torch_dev_env"
TTXLA_VENV="${PROJECT_ROOT}/venv"

# ── Parse arguments ──────────────────────────────────────────────────────────
BUILD_TYPE="Release"
for arg in "$@"; do
    case "$arg" in
        --debug)  BUILD_TYPE="Debug" ;;
        --release) BUILD_TYPE="Release" ;;
        *) echo "Unknown argument: $arg"; echo "Usage: $0 [--debug|--release]"; exit 1 ;;
    esac
done

echo "=== Build configuration ==="
echo "  Python:     ${PYTHON_VERSION}"
echo "  PyTorch:    ${PYTORCH_TAG}"
echo "  XLA branch: ${PYTORCH_XLA_BRANCH}"
echo "  Build type: ${BUILD_TYPE}"
echo "  Temp dir:   ${TEMP_DIR}"
echo ""

# ── Helper functions ─────────────────────────────────────────────────────────
info()  { echo -e "\n\033[1;34m>>> $*\033[0m"; }
warn()  { echo -e "\n\033[1;33mWARN: $*\033[0m"; }
fail()  { echo -e "\n\033[1;31mERROR: $*\033[0m"; exit 1; }

check_command() {
    command -v "$1" &>/dev/null || return 1
}

# ── Step 1: Verify prerequisites ────────────────────────────────────────────
info "Checking prerequisites..."

PYTHON_BIN="python${PYTHON_VERSION}"
check_command "${PYTHON_BIN}" || fail "python${PYTHON_VERSION} not found."
check_command bazel          || fail "Bazel not found."
for cmd in cmake git ninja gcc g++; do
    check_command "$cmd"     || fail "${cmd} not found."
done

ACTUAL_BAZEL_VERSION=$(bazel --version | awk '{print $2}')
if [[ "${ACTUAL_BAZEL_VERSION}" != "${BAZEL_VERSION}" ]]; then
    warn "Bazel version mismatch: expected ${BAZEL_VERSION}, got ${ACTUAL_BAZEL_VERSION}"
fi

echo "  All prerequisites OK."

# ── Step 2: Create temp directory and venv ───────────────────────────────────
mkdir -p "${TEMP_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
    info "Creating Python ${PYTHON_VERSION} virtual environment..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

info "Installing Python build dependencies..."
pip install --upgrade pip setuptools wheel -q
pip install numpy pyyaml cmake ninja typing_extensions six requests \
    astunparse expecttest hypothesis psutil -q

# ── Step 3: Clone/verify PyTorch ─────────────────────────────────────────────
if [[ -d "${PYTORCH_DIR}" ]]; then
    info "PyTorch directory exists. Verifying version..."
    cd "${PYTORCH_DIR}"
    CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "unknown")
    if [[ "${CURRENT_TAG}" == "${PYTORCH_TAG}" ]]; then
        echo "  PyTorch ${PYTORCH_TAG} already cloned."
    else
        fail "PyTorch directory exists but is at '${CURRENT_TAG}', not '${PYTORCH_TAG}'.
  Remove ${PYTORCH_DIR} and re-run."
    fi
else
    info "Cloning PyTorch ${PYTORCH_TAG}..."
    git clone --recursive --branch "${PYTORCH_TAG}" https://github.com/pytorch/pytorch "${PYTORCH_DIR}"
fi

# ── Step 4: Clone/verify PyTorch-XLA (Tenstorrent fork) ─────────────────────
if [[ -d "${XLA_DIR}" ]]; then
    info "PyTorch-XLA directory exists. Verifying remote..."
    cd "${XLA_DIR}"
    CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "unknown")
    if [[ "${CURRENT_REMOTE}" == "${PYTORCH_XLA_REPO}" ]]; then
        echo "  Tenstorrent pytorch-xla already cloned."
        info "Pulling latest changes on ${PYTORCH_XLA_BRANCH}..."
        git fetch origin
        git checkout "${PYTORCH_XLA_BRANCH}"
        git pull origin "${PYTORCH_XLA_BRANCH}"
        git submodule update --init --recursive
    else
        fail "XLA directory exists but points to '${CURRENT_REMOTE}', not '${PYTORCH_XLA_REPO}'.
  Remove ${XLA_DIR} and re-run."
    fi
else
    info "Cloning Tenstorrent pytorch-xla (${PYTORCH_XLA_BRANCH})..."
    git clone --recursive --branch "${PYTORCH_XLA_BRANCH}" "${PYTORCH_XLA_REPO}" "${XLA_DIR}"
fi

# ── Step 5: Build PyTorch ────────────────────────────────────────────────────
info "Building PyTorch (${BUILD_TYPE})..."
cd "${PYTORCH_DIR}"

export USE_CUDA=0
export BUILD_TEST=0

if [[ "${BUILD_TYPE}" == "Debug" ]]; then
    export DEBUG=1
fi

TORCH_INSTALLED=0
if python -c "import torch; assert torch.version.git_version.startswith('$(git rev-parse --short HEAD)')" 2>/dev/null; then
    TORCH_INSTALLED=1
    echo "  PyTorch already built and installed for this commit. Skipping rebuild."
fi

if [[ "${TORCH_INSTALLED}" -eq 0 ]]; then
    python setup.py bdist_wheel
    python setup.py develop
    echo "  PyTorch build complete."
fi

python -c "import torch; print(f'  PyTorch {torch.__version__} (git: {torch.version.git_version})')"

# ── Step 6: Build PyTorch-XLA ────────────────────────────────────────────────
info "Building PyTorch-XLA (${BUILD_TYPE})..."
cd "${XLA_DIR}"

export PYTORCH_REPO_PATH="${PYTORCH_DIR}"
export TEST_TMPDIR="${BAZEL_TMP}"
# System python3 is 3.10 but we need 3.12; tell bazel's hermetic Python resolver.
export HERMETIC_PYTHON_VERSION="${PYTHON_VERSION}"
mkdir -p "${BAZEL_TMP}"

if [[ "${BUILD_TYPE}" == "Debug" ]]; then
    export XLA_DEBUG=1
fi

XLA_COMMIT=$(git rev-parse HEAD)
XLA_STAMP_FILE="${TEMP_DIR}/.torch_xla_built_commit"

if [[ -f "${XLA_STAMP_FILE}" ]] && [[ "$(cat "${XLA_STAMP_FILE}")" == "${XLA_COMMIT}" ]] && \
   python -c "import torch_xla" 2>/dev/null; then
    echo "  PyTorch-XLA already built for commit ${XLA_COMMIT:0:7}. Skipping rebuild."
else
    rm -rf build/
    python setup.py develop
    echo "${XLA_COMMIT}" > "${XLA_STAMP_FILE}"
    echo "  PyTorch-XLA build complete."
fi

# ── Step 7: Integrate into tt-xla venv ───────────────────────────────────────
info "Integrating into tt-xla venv at ${TTXLA_VENV}..."

if [[ ! -d "${TTXLA_VENV}" ]]; then
    fail "tt-xla venv not found at ${TTXLA_VENV}. Set up tt-xla first."
fi

deactivate
cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
set +u
source "${TTXLA_VENV}/activate"
set -u

pip uninstall torch_xla -y 2>/dev/null || true

cd "${XLA_DIR}"
pip install -e .

pip install "Jinja2>=3.1" "Pygments>=2.17" "torch==${PYTORCH_TAG#v}" -q

# ── Step 8: Final verification ───────────────────────────────────────────────
info "Verifying installation in tt-xla venv..."

python -c "
import torch
print(f'  PyTorch:   {torch.__version__} (git: {torch.version.git_version})')
import torch_xla
print(f'  torch_xla: {torch_xla.__file__}')
print('All good!')
"

info "Done! Build type: ${BUILD_TYPE}"
echo ""
echo "To use in tt-xla:"
echo "  cd ${PROJECT_ROOT}"
echo "  source venv/activate"
echo ""
