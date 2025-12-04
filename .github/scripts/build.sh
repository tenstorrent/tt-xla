#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
set -e -o pipefail

# clean build artifacts and tt-mlir submodule
rm -rf build
rm -rf third_party/tt-mlir

cd python_package
python setup.py bdist_wheel --build-type release
cd ..
