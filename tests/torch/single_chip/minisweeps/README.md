# minisweeps

A lightweight, file-driven harness for tt-xla op precision tests.

Inspired by **[tt-forge-sweeps](https://github.com/tenstorrent/tt-forge-sweeps)** —
test IDs are in the same sweeps format, env vars (`TEST_ID`, `ID_FILES`)
have the same names and semantics, and conf files from sweeps drop in
without modification. The difference: no submodule, no `TestPlan` /
`VerifyConfig` / `AutomaticValueChecker` machinery, no conftest hooks.
A test file owns its operator definition and points at a `.conf` of
test_ids next to it; `minisweeps.py` handles loading and verification.

For API details, internal layout, and how to add a new operator, see
[`development.md`](./development.md).

## Running the tests

```bash
source /localdev/<user>/venv/sweeps/xla/bin/activate     # or your venv
export PYTHONPATH="$PWD:$PWD/tests:$PWD/tests/torch/single_chip/minisweeps:$PYTHONPATH"
```

### Default run

Loads the test's own conf file (`test_matmul_mp_grid.conf` next to the
test):

```bash
pytest tests/torch/single_chip/minisweeps/test_matmul_mp.py
```

### Single test_id

```bash
TEST_ID="matmul_mp-FROM_ANOTHER_OP-{'compiler_config': 'mp_opt2_bf16_fp32accfalse_hifi2'}-((32, 128, 1024), (1024, 2048))-None-None" \
    pytest tests/torch/single_chip/minisweeps/test_matmul_mp.py -sv
```

### Custom conf file

```bash
# bare name resolves next to the test
ID_FILES=test_matmul_mp_pcc.conf \
    pytest tests/torch/single_chip/minisweeps/test_matmul_mp.py

# absolute or repo-relative paths work too; comma-separates several files
ID_FILES=tests/torch/single_chip/minisweeps/test_matmul_mp_pcc.conf,/tmp/scratch.conf \
    pytest tests/torch/single_chip/minisweeps/test_matmul_mp.py
```

### Input regime

```bash
MINISWEEPS_PROFILE=uniform pytest ...   # matches sweeps' ValueRanges.SMALL
MINISWEEPS_PROFILE=mixture pytest ...   # default; LLM-style with outliers
```

## Layout

```
tests/torch/single_chip/minisweeps/
├── README.md                    ← you are here
├── development.md               ← API surface, internals, "writing a new op"
├── minisweeps.py                ← TestVector, load_test_vectors, verify
├── conftest.py                  ← cache-clear fixture (replaces --forked)
├── test_matmul_mp.py            ← matmul_mp operator + pytest entry
├── test_matmul_mp_grid.conf     ← default test_ids for matmul_mp
└── test_matmul_mp_pcc.conf      ← legacy sweeps-failure ids (for replay)
```
