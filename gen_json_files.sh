LABEL="after"

# Push Tests
TT_XLA_TEST_METADATA_DIR=/localdev/kmabee/tt-xla/dump_${LABEL}/ pytest -vv tests/runner/test_models.py --arch p150 -m "p150 and expected_passing and push" |& tee p150_push_${LABEL}.log
TT_XLA_TEST_METADATA_DIR=/localdev/kmabee/tt-xla/dump_${LABEL}/ pytest -vv tests/runner/test_models.py --arch n150 -m "n150 and expected_passing and push" |& tee n150_push_${LABEL}.log

# XFAIL Tests
TT_XLA_TEST_METADATA_DIR=/localdev/kmabee/tt-xla/dump_${LABEL}/ pytest -vv tests/runner/test_models.py --arch n150 -m "(n150 and (known_failure_xfail or not_supported_skip)) or placeholder" |& tee n150_xfail_${LABEL}.log
TT_XLA_TEST_METADATA_DIR=/localdev/kmabee/tt-xla/dump_${LABEL}/ pytest -vv tests/runner/test_models.py --arch p150 -m "(p150 and (known_failure_xfail or not_supported_skip)) or placeholder" |& tee p150_xfail_${LABEL}.log

# Expected Passing Tests
TT_XLA_TEST_METADATA_DIR=/localdev/kmabee/tt-xla/dump_${LABEL}/ pytest -vv tests/runner/test_models.py --arch n150 -m "n150 and expected_passing" |& tee n150_expected_passing_${LABEL}.log
TT_XLA_TEST_METADATA_DIR=/localdev/kmabee/tt-xla/dump_${LABEL}/ pytest -vv tests/runner/test_models.py --arch p150 -m "p150 and expected_passing" |& tee p150_expected_passing_${LABEL}.log
TT_XLA_TEST_METADATA_DIR=/localdev/kmabee/tt-xla/dump_${LABEL}/ pytest -vv tests/runner/test_models.py --arch n300-llmbox -m "tensor_parallel and n300-llmbox and expected_passing" |& tee n300-llmbox_expected_passing_${LABEL}.log
TT_XLA_TEST_METADATA_DIR=/localdev/kmabee/tt-xla/dump_${LABEL}/ pytest -vv tests/runner/test_models.py --arch n300 -m "data_parallel and n300 and expected_passing" |& tee n300_expected_passing_${LABEL}.log
