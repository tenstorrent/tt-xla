#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

TT_METAL_VERSION=$1
if [[ -z "$TT_METAL_VERSION" ]]; then
  echo "Usage: $0 <TT_METAL_VERSION>" >&2
  exit 1
fi

mkdir temp
cd temp

wget -O install_debugger.sh "https://raw.githubusercontent.com/tenstorrent/tt-metal/${TT_METAL_VERSION}/scripts/install_debugger.sh"
wget -O ttexalens_ref.txt "https://raw.githubusercontent.com/tenstorrent/tt-metal/${TT_METAL_VERSION}/scripts/ttexalens_ref.txt"
wget -O requirements.txt "https://raw.githubusercontent.com/tenstorrent/tt-metal/${TT_METAL_VERSION}/tools/triage/requirements.txt"

chmod u+x install_debugger.sh
./install_debugger.sh

pip install --no-cache-dir -r requirements.txt

cd ..
rm -rf ./temp
