
# Sweeps operators tests

Sweeps operator tests resides in repo [TT-Forge-Sweeps](https://github.com/tenstorrent/tt-forge-sweeps). Execution of sweeps tests is fully available from TT-Xla via pytest function `tests/torch/single_chip/operators/test_query.py`.

---

## Setup git submodule

To initialize git submodule run following commands:

```bash
pushd third_party/tt_forge_sweeps
git submodule update --init .
git sparse-checkout init .
git sparse-checkout set src/sweeps
popd
```
---

## Remove git submodule

To remove tt-forge-sweeps submodule run following commands:

```bash
git submodule deinit -f third_party/tt_forge_sweeps
```

Sometimes additional steps needed to cleanup tt-forge-sweeps submodule completely

```bash
git rm --cached third_party/tt_forge_sweeps
git config --unset submodule.third_party/tt_forge_sweeps.active
git config -f .git/config --remove-section submodule.third_party/tt_forge_sweeps
```

---
## Execute sweeps tests

Example commands to run sweeps tests:

Run 0.2% of all sweeps tests
```bash
SAMPLE=0.2 pytest tests/torch/single_chip/operators/test_query.py -v
```

Run 1% of conv2d tests
```bash
SAMPLE=1 OPERATORS=conv2d pytest tests/torch/single_chip/operators/test_query.py -v
```

Run all conv2d tests with failing reason BAD_STATUS0R_ACCESS
```bash
FAILING_REASONS=BAD_STATUS0R_ACCESS OPERATORS=conv2d pytest tests/torch/single_chip/operators/test_query.py -v
```

List all conv2d tests with failing reason BAD_STATUS0R_ACCESS
```bash
FAILING_REASONS=BAD_STATUS0R_ACCESS OPERATORS=conv2d pytest tests/torch/single_chip/operators/test_query.py --collect-only
```

Print sweeps help
```bash
. third_party/tt_forge_sweeps/src/sweeps/operators/pytorch/test_commands.sh
PYTHONPATH=$(pwd):$(pwd)/tests print_query_docs
```
