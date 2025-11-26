
## Sweeps operators tests

Sweeps operator tests resides in tt-forge-sweeps

```bash
pushd third_party/tt_forge_sweeps
git submodule update --init .
git sparse-checkout init .
# git sparse-checkout init --cone
git sparse-checkout set sweeps/src/sweeps
popd
```

To remove tt-forge-sweeps submodule run

```bash
git submodule deinit -f third_party/tt_forge_sweeps
```

Sometimes additional steps needed to cleanup tt-forge-sweeps submodule

```bash
git rm --cached third_party/tt_forge_sweeps
git config --unset submodule.third_party/tt_forge_sweeps.active
git config -f .git/config --remove-section submodule.third_party/tt_forge_sweeps
```

Apply patches

```bash
git apply ../../tests/torch/single_chip/operators/sweeps_tags.patch
```

Revert patches

```bash
git apply --reverse ../../tests/torch/single_chip/operators/sweeps_tags.patch
```


To run sweeps tests

```bash
cd tests/
SAMPLE=0.05 pytest torch/single_chip/operators/test_query.py -v
```
