# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Usage:
# source forge/test/operators/pytorch/test_commands.sh


function print_help {
    echo "Help:"
    echo "  print_help              - Print commands and current query parameters."
    echo "  print_query_docs        - Print docs for all available query parameters."
    echo "  print_pytest_commands   - Print pytest helper commands."
    echo "Query parameters:"
    echo "  print_params            - Print current query parameters values."
    echo "  reset_params            - Reset all query and test configuration parameters."
    echo "Setup commands:"
    echo "  select_test_query       - Select test_query pytest."
    echo "  select_test_push        - Select test_push pytest."
    echo "Example commands:"
    echo "  OPERATORS=add with-params pytest"
    echo "Run commands:"
    echo "  pytest                  - Run all tests or subset of test plan based on a query parameters."
    echo "  with-params pytest      - Print params before and after test run."
    echo "  export_tests            - Export tests from test plan to JSON file based on a query parameters."
    print_params
}

function print_params {
    echo "Query Params:"
    echo "  OPERATORS=$OPERATORS"
    echo "  FILTERS=$FILTERS"
    echo "  INPUT_SOURCES=$INPUT_SOURCES"
    echo "  INPUT_SHAPES=\"$INPUT_SHAPES\""
    echo "  DEV_DATA_FORMATS=$DEV_DATA_FORMATS"
    echo "  MATH_FIDELITIES=$MATH_FIDELITIES"
    echo "  KWARGS=\"$KWARGS\""
    echo "  FAILING_REASONS=$FAILING_REASONS"
    echo "  SKIP_REASONS=$SKIP_REASONS"
    echo "  RANDOM_SEED=$RANDOM_SEED"
    echo "  SAMPLE=$SAMPLE"
    echo "  RANGE=$RANGE"
    echo "  TEST_ID=$TEST_ID"
    echo "  ID_FILES=$ID_FILES"
    echo "  ID_FILES_IGNORE=$ID_FILES_IGNORE"
    print_test_config_params
    print_pytest_params
}

function reset_params {
    reset_query_params
    reset_test_config_params
    # reset_pytest_opts
}

function print_query_docs {
    max_width=$(tput cols)
    max_width=$((max_width * 80 / 100))

    pushd ${SWEEPS_SCRIPT_DIR}/../../../../
    python3 -c "from test.operators.pytorch.test_all import InfoUtils; InfoUtils.print_query_params(max_width=${max_width})"
    popd
}

function export_tests {
    local file_name=$1
    local logs_dir=${LOGS_DIR}

    if [ "${logs_dir}" == "" ]; then
        logs_dir=$(pwd)
    fi

    if [ "${file_name}" == "" ]; then
        file_name="test_vectors_$(filters_to_string).json"
    fi

    if [[ ! "${file_name}" == */* ]]; then
        file_name="${logs_dir}/${file_name}"
    fi

    pushd ${SWEEPS_SCRIPT_DIR}/../../../../
    python3 -c "from test.operators.pytorch.test_all import InfoUtils; InfoUtils.export(file_name=\"${file_name}\")"
    popd
}


function reset_query_params {
    unset OPERATORS
    unset FILTERS
    unset INPUT_SOURCES
    unset INPUT_SHAPES
    unset DEV_DATA_FORMATS
    unset MATH_FIDELITIES
    unset KWARGS
    unset FAILING_REASONS
    unset SKIP_REASONS
    unset RANDOM_SEED
    unset SAMPLE
    unset RANGE
    unset TEST_ID
    unset ID_FILES
    unset ID_FILES_IGNORE
    print_params
}


function filters_to_string {
    filters=""
    if [ "${TEST_NAME}" != "" ]; then
        filters="${filters}_TEST_${TEST_NAME}"
    fi
    if [ "${OPERATORS}" != "" ]; then
        filters="${filters}_OPERATORS_${OPERATORS}"
    fi
    if [ "${FILTERS}" != "" ]; then
        filters="${filters}_FILTERS_${FILTERS}"
    fi
    if [ "${INPUT_SOURCES}" != "" ]; then
        filters="${filters}_INPUT_SOURCES_${INPUT_SOURCES}"
    fi
    if [ "${INPUT_SHAPES}" != "" ]; then
        filters="${filters}_INPUT_SHAPES_${INPUT_SHAPES}"
    fi
    if [ "${DEV_DATA_FORMATS}" != "" ]; then
        filters="${filters}_DEV_DATA_FORMATS_${DEV_DATA_FORMATS}"
    fi
    if [ "${MATH_FIDELITIES}" != "" ]; then
        filters="${filters}_MATH_FIDELITIES_${MATH_FIDELITIES}"
    fi
    if [ "${KWARGS}" != "" ]; then
        filters="${filters}_KWARGS_${KWARGS}"
    fi
    if [ "${FAILING_REASONS}" != "" ]; then
        filters="${filters}_FAILING_REASONS_${FAILING_REASONS}"
    fi
    if [ "${SKIP_REASONS}" != "" ]; then
        filters="${filters}_SKIP_REASONS_${SKIP_REASONS}"
    fi
    if [ "${RANDOM_SEED}" != "" ]; then
        filters="${filters}_RANDOM_SEED_${RANDOM_SEED}"
    fi
    if [ "${SAMPLE}" != "" ]; then
        filters="${filters}_SAMPLE_${SAMPLE}"
    fi
    if [ "${RANGE}" != "" ]; then
        filters="${filters}_RANGE_${RANGE}"
    fi
    if [ "${TEST_ID}" != "" ]; then
        filters="${filters}_TEST_ID_${TEST_ID}"
    fi
    if [ "${ID_FILES}" != "" ]; then
        filters="${filters}_ID_FILES_${ID_FILES}"
    fi
    if [ "${ID_FILES_IGNORE}" != "" ]; then
        filters="${filters}_ID_FILES_IGNORE_${ID_FILES_IGNORE}"
    fi
    filters=$(
        echo "${filters}" \
        | sed 's/[ {}'"'"']//g' \
        | sed 's/[ ,:\(\)\/]/_/g' \
    )
    echo "${filters}"
}


function select_test_query {
    set_pytest_function "${SWEEPS_SCRIPT_DIR}/test_query.py::test_query"
    # set_pytest_function "forge/test/operators/pytorch/test_query.py::test_query"
}

function select_test_push {
    set_pytest_function "${SWEEPS_SCRIPT_DIR}/test_push.py::test_push"
}


SWEEPS_SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

. ${SWEEPS_SCRIPT_DIR}/test_common.sh

select_test_query
# _set_pytest_opts

print_help
