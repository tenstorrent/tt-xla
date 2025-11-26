

XLA_ROOT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

XLA_ROOT_DIR=$(realpath "${XLA_ROOT_DIR}/../../../../")

function cd-xla-tests {
    cd "${XLA_ROOT_DIR}/tests" || exit 1
}

function cd-xla-third-party-sweeps {
    cd "${XLA_ROOT_DIR}/third_party/tt_forge_sweeps" || exit 1
}
