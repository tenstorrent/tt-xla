#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

project_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")/../.." rev-parse --show-toplevel)"
cd "$project_root"

venv_dir="/opt/tt-triage-venv"

python3.12 -m venv "$venv_dir"
source $venv_dir/bin/activate

TT_MLIR_VERSION=$(grep 'set(TT_MLIR_VERSION' third_party/CMakeLists.txt | sed 's/.*"\(.*\)".*/\1/')
echo "TT_MLIR_VERSION=${TT_MLIR_VERSION}"

mlir_temp="$(mktemp)"
tt_mlir_tp_url="https://raw.githubusercontent.com/tenstorrent/tt-mlir/${TT_MLIR_VERSION}/third_party/CMakeLists.txt"
echo "Downloading ${tt_mlir_tp_url}"
curl -fL "${tt_mlir_tp_url}" -o "${mlir_temp}"
TT_METAL_VERSION=$(grep 'set(TT_METAL_VERSION' "${mlir_temp}" | sed 's/.*"\(.*\)".*/\1/')
echo "TT_METAL_VERSION=${TT_METAL_VERSION}"

cd $(mktemp -d)

wget -O install_debugger.sh "https://raw.githubusercontent.com/tenstorrent/tt-metal/${TT_METAL_VERSION}/scripts/install_debugger.sh"
wget -O ttexalens_ref.txt "https://raw.githubusercontent.com/tenstorrent/tt-metal/${TT_METAL_VERSION}/scripts/ttexalens_ref.txt"
wget -O requirements.txt "https://raw.githubusercontent.com/tenstorrent/tt-metal/${TT_METAL_VERSION}/tools/triage/requirements.txt"

chmod u+x install_debugger.sh
./install_debugger.sh
pip install --no-cache-dir -r requirements.txt

cd -

tt_triage_dir=/opt/tt-triage
mkdir -p $tt_triage_dir
curl -L "https://github.com/tenstorrent/tt-metal/archive/${TT_METAL_VERSION}.tar.gz" \
  | tar -xz -C $tt_triage_dir --strip-components=1 \
      tt-metal-${TT_METAL_VERSION}/scripts/ttexalens_ref.txt \
      tt-metal-${TT_METAL_VERSION}/tools/tt-triage.py \
      tt-metal-${TT_METAL_VERSION}/tools/triage \
      tt-metal-${TT_METAL_VERSION}/tt_metal

tt_triage_exec="$tt_triage_dir/tools/tt-triage.py"
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="source ${venv_dir}/bin/activate && python ${tt_triage_exec} 1>&2"
