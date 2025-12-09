# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Usage:
# . tests/torch/single_chip/operators/bash_functions.sh
# cd-xla-tests
# cd-xla-third-party-sweeps


XLA_ROOT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

XLA_ROOT_DIR=$(realpath "${XLA_ROOT_DIR}/../../../../")

function cd-xla-tests {
    cd "${XLA_ROOT_DIR}/tests" || exit 1
}

function cd-xla-third-party-sweeps {
    cd "${XLA_ROOT_DIR}/third_party/tt_forge_sweeps" || exit 1
}
