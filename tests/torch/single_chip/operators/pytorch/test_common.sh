# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Common pytest helper functions


function _set_pytest_opts {
    local PYTEST_PARAMS="${DEFAULT_PYTEST_ADDOPTS}"
    if [ "${_PYTEST_FUNCTION}" != "" ]; then
        PYTEST_PARAMS="${PYTEST_PARAMS} ${_PYTEST_FUNCTION}"
    fi
    export PYTEST_ADDOPTS="${PYTEST_PARAMS}"
    print_pytest_params
}

function reset_pytest_opts {
    _set_pytest_opts
}

function print_pytest_params {
    echo "Pytest options:"
    echo "  PYTEST_ADDOPTS = $PYTEST_ADDOPTS"
}

function reset_test_config_params {
    unset SKIP_FORGE_VERIFICATION
}

function print_test_config_params {
    echo "Test configuration:"
    echo "  SKIP_FORGE_VERIFICATION=$SKIP_FORGE_VERIFICATION"
}

function print_pytest_commands {
    echo "Setup pytest options:"
    echo "  reset_pytest_opts   - Reset pytest options."
    echo "Setup test configuration:"
    echo "  reset_test_config_params   - Reset test configuration parameters."
}


function unset_pytest_function {
    unset _PYTEST_FUNCTION
    _set_pytest_opts
}

function set_pytest_function {
    if [ "${_PYTEST_FUNCTION}" != "$1" ]; then
        _PYTEST_FUNCTION="$1"
        _set_pytest_opts
    fi
}

function with-params {
    print_params
    "$@"
    print_params
}
